#!/usr/bin/env python3
"""
s11: Error Recovery — 三条恢复路径 + 指数退避。

AI 编程智能体的容错机制：
1. Path 1 (max_tokens): 输出截断 → 先升级 8K→64K（不追加截断输出）→ 再发续写提示（最多 3 次）
2. Path 2 (prompt_too_long): 上下文过长 → 紧急压缩 → 重试（1 次）
3. Path 3 (429/529): 速率/过载 → 指数退避 + 随机抖动（最多 10 次）→ 连续 529 切换备选模型
4. RecoveryState 追踪所有恢复状态（升级/压缩/529/模型切换）
5. with_retry 包装瞬态错误，非瞬态异常向上抛出给外层 handler

Run / 运行: python s11_error_recovery/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

Changes from s10 / 相对 s10 的变更:
  - LLM call wrapped in try/except with three recovery paths
  - Path 1: max_tokens -> escalate 8K->64K (no append on first escalation),
            then continuation prompt (max 3)
  - Path 2: prompt_too_long -> reactive compact -> retry (once)
  - Path 3: 429/529 -> exponential backoff with jitter (max 10),
            fallback model on consecutive 529
  - with_retry wrapper for transient errors
  - RecoveryState tracks escalation / compact / 529 / model

ASCII flow:
  messages -> prompt assembly -> [try] LLM [except] -> tools -> loop
                                      |          |
                                stop_reason   error type
                                max_tokens?   prompt_too_long? -> compact
                                escalate /    429/529? -> backoff
                                continue      other? -> log + exit
"""

import os, subprocess, time, random, json
from pathlib import Path

# ── 终端中文输入兼容性 ──────────────────────────────────
# macOS 的 libedit 在处理中文输入时有退格问题，readline 配置修复它
try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')   # 关闭特殊字符绑定
    readline.parse_and_bind('set input-meta on')                # UTF-8 输入
    readline.parse_and_bind('set output-meta on')               # UTF-8 输出
    readline.parse_and_bind('set convert-meta off')             # 保持原始字节
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

# ── 初始化 Anthropic 客户端 ────────────────────────────
load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()  # 工作目录
MEMORY_DIR = WORKDIR / ".memory"  # s09: 记忆目录
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"  # 记忆索引
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
PRIMARY_MODEL = os.environ["MODEL_ID"]  # s11: 主模型
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL_ID")  # s11: 备选模型（连续 529 时切换）

# ═══════════════════════════════════════════════════════════
#  s11 恢复常量
# ═══════════════════════════════════════════════════════════

ESCALATED_MAX_TOKENS = 64000   # 升级后的 max_tokens
DEFAULT_MAX_TOKENS = 8000      # 默认 max_tokens
MAX_RECOVERY_RETRIES = 3       # 续写提示最大尝试次数
MAX_RETRIES = 10               # 429/529 最大重试次数
BASE_DELAY_MS = 500            # 退避基础延迟（毫秒）
MAX_CONSECUTIVE_529 = 3        # 连续 529 触发模型切换的阈值
CONTINUATION_PROMPT = (        # 续写提示词
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought."
)

# ═══════════════════════════════════════════════════════════
#  FROM s10: Prompt 组装 — 来自 s10（同步）
# ═══════════════════════════════════════════════════════════

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    sections = [PROMPT_SECTIONS["identity"],
                PROMPT_SECTIONS["tools"],
                PROMPT_SECTIONS["workspace"]]
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")
    return "\n\n".join(sections)


_last_context_key, _last_prompt = None, None


def get_system_prompt(context: dict) -> str:
    global _last_context_key, _last_prompt
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        print("  \033[90m[cache hit] system prompt unchanged\033[0m")
        return _last_prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)

    loaded = ["identity", "tools", "workspace"]
    if context.get("memories"):
        loaded.append("memory")
    print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
    return _last_prompt


# ── 工具实现（未改动）────────────────────────────────
def safe_path(p: str) -> Path:
    """路径安全校验。防止模型读写工作目录之外的文件。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 shell 命令并返回输出。"""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容。可指定行数上限，超出则截断。"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件。自动创建父目录。"""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# ── 工具定义 ─────────────────────────────────────────
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
]

TOOL_HANDLERS = {"bash": run_bash, "read_file": run_read, "write_file": run_write}


# ═══════════════════════════════════════════════════════════
#  NEW in s11: 错误恢复系统（三条恢复路径 + 指数退避）
# ═══════════════════════════════════════════════════════════

class RecoveryState:
    """追踪循环中的恢复尝试状态。
    字段说明：
    - has_escalated: 是否已从 8K 升级到 64K max_tokens
    - recovery_count: 续写提示已发次数（最多 MAX_RECOVERY_RETRIES）
    - consecutive_529: 连续过载错误计数（达阈值时切换模型）
    - has_attempted_reactive_compact: 是否已尝试过紧急压缩
    - current_model: 当前使用的模型 ID"""
    def __init__(self):
        self.has_escalated = False
        self.recovery_count = 0
        self.consecutive_529 = 0
        self.has_attempted_reactive_compact = False
        self.current_model = PRIMARY_MODEL


def retry_delay(attempt, retry_after=None):
    """指数退避 + 随机抖动。API 的 Retry-After 头优先。"""
    if retry_after:
        return retry_after
    base = min(BASE_DELAY_MS * (2 ** attempt), 32000) / 1000 
    jitter = random.uniform(0, base * 0.25)  # 25% 随机抖动，避免惊群效应
    return base + jitter


def with_retry(fn, state: RecoveryState):
    """Path 3: 瞬态错误（429/529）的指数退避包装器。
        429（速率限制）→ 纯退避；
        529（过载）→ 退避 + 连续 3 次后切换备选模型。
    非瞬态异常不做退避，直接向上抛出给外层 handler。"""
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_529 = 0  # 成功后重置
            return result
        except Exception as e:
            name = type(e).__name__
            msg = str(e).lower()

            # 429 速率限制 → 指数退避
            if "ratelimit" in name.lower() or "429" in msg:
                delay = retry_delay(attempt)
                print(f"  \033[33m[429 rate limit] retry {attempt+1}/{MAX_RETRIES},"
                      f" wait {delay:.1f}s\033[0m")
                time.sleep(delay)
                continue

            # 529 过载 → 指数退避 + 连续 N 次后切换备选模型
            if "overloaded" in name.lower() or "529" in msg or "overloaded" in msg:
                state.consecutive_529 += 1
                if state.consecutive_529 >= MAX_CONSECUTIVE_529:
                    if FALLBACK_MODEL: # 环境变量配置了备选模型，切换过去
                        state.current_model = FALLBACK_MODEL
                        state.consecutive_529 = 0
                        print(f"  \033[31m[529 x{MAX_CONSECUTIVE_529}]"
                              f" switching to {FALLBACK_MODEL}\033[0m")
                    else: # 没有备选模型，记录日志后继续退避重试
                        state.consecutive_529 = 0
                        print(f"  \033[31m[529 x{MAX_CONSECUTIVE_529}]"
                              f" no FALLBACK_MODEL_ID configured, continuing retry\033[0m")
                delay = retry_delay(attempt)
                print(f"  \033[33m[529 overloaded] retry {attempt+1}/{MAX_RETRIES},"
                      f" wait {delay:.1f}s\033[0m")
                time.sleep(delay)
                continue

            # 非瞬态错误 → 不重试，直接向上抛出
            raise
    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded") # 达到最大重试次数仍未成功，抛出异常


def is_prompt_too_long_error(e: Exception) -> bool:
    """判断 API 异常是否为 prompt/上下文过长错误。"""
    msg = str(e).lower()
    return (("prompt" in msg and "long" in msg)
            or "prompt_is_too_long" in msg
            or "context_length_exceeded" in msg
            or "max_context_window" in msg)


def reactive_compact(messages: list) -> list:
    """Path 2: 紧急压缩。教学版保留最后 5 条消息。
    真正的 CC 会通过 LLM 生成压缩摘要后重试，教学版简化为尾部保留。
    s08/s09 已覆盖 LLM 压缩，这里不再重复。"""
    print("  \033[31m[reactive compact] trimming to last 5 messages\033[0m")
    tail = messages[-5:]
    return [{"role": "user",
             "content": "[Reactive compact] Earlier conversation trimmed. "
                        "Continue from where you left off."}, *tail]


# ═══════════════════════════════════════════════════════════
#  FROM s10: Context 系统 — 从真实状态推导
# ═══════════════════════════════════════════════════════════

def update_context(context: dict, messages: list) -> dict:
    """从真实文件系统状态推导 context。"""
    memories = ""
    if MEMORY_INDEX.exists():
        content = MEMORY_INDEX.read_text().strip()
        if content:
            memories = content
    return {
        "enabled_tools": list(TOOL_HANDLERS.keys()),
        "workspace": str(WORKDIR),
        "memories": memories,
    }


def _tool_input_summary(block) -> str:
    """提取工具调用的关键参数摘要，用于终端显示。"""
    inputs = block.input
    if block.name == "bash":
        return inputs.get("command", "")
    return inputs.get("path", "")


# ═══════════════════════════════════════════════════════════
#  agent_loop — s11 核心：LLM 调用外包装错误恢复
#  三层 try/except：with_retry（429/529）→ 外层 prompt_too_long → 兜底
# ═══════════════════════════════════════════════════════════

def agent_loop(messages: list, context: dict):
    """智能体主循环 + 错误恢复。
    流程：
    1. 从 context 组装 system prompt（带缓存）
    2. LLM 调用 → with_retry 处理 429/529（指数退避 + 模型切换）
    3. 外层 try/except 处理 prompt_too_long（紧急压缩，1 次）
    4. max_tokens 截断 → 先升级 8K→64K → 再发续写提示（最多 3 次）
    5. 不可恢复错误 → 记录错误文本并退出
    """
    system = get_system_prompt(context)
    state = RecoveryState()
    max_tokens = DEFAULT_MAX_TOKENS

    while True:
        # s11: LLM 调用——with_retry 处理 429/529，外层处理其余异常
        try:
            response = with_retry( # Path 3: 瞬态错误（429/529）的指数退避包装器
                lambda mt=max_tokens, mdl=state.current_model:
                    client.messages.create(
                        model=mdl, system=system, messages=messages,
                        tools=TOOLS, max_tokens=mt),
                state)
        except Exception as e:
            # Path 2: prompt_too_long → 紧急压缩（仅 1 次）
            if is_prompt_too_long_error(e):
                if not state.has_attempted_reactive_compact:
                    messages[:] = reactive_compact(messages)
                    state.has_attempted_reactive_compact = True
                    continue
                print("  \033[31m[unrecoverable] still too long after compact\033[0m")
                messages.append({"role": "assistant", "content": [
                    {"type": "text",
                     "text": "[Error] Context too large, cannot continue."}]})
                return

            # 不可恢复错误
            name = type(e).__name__
            print(f"  \033[31m[unrecoverable] {name}: {str(e)[:100]}\033[0m")
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": f"[Error] {name}: {str(e)[:200]}"}]})
            return

        # Path 1: max_tokens 截断 → 升级或续写
        if response.stop_reason == "max_tokens":
            # 第一次截断：不追加截断输出，直接升级 max_tokens 重试同一请求
            if not state.has_escalated:
                max_tokens = ESCALATED_MAX_TOKENS
                state.has_escalated = True
                print(f"  \033[33m[max_tokens] escalating"
                      f" {DEFAULT_MAX_TOKENS} -> {ESCALATED_MAX_TOKENS}\033[0m")
                continue
            # 64K 仍然截断：保存截断输出 + 续写提示
            messages.append({"role": "assistant", "content": response.content})
            if state.recovery_count < MAX_RECOVERY_RETRIES:
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                state.recovery_count += 1
                print(f"  \033[33m[max_tokens] continuation"
                      f" {state.recovery_count}/{MAX_RECOVERY_RETRIES}\033[0m")
                continue
            print("  \033[31m[max_tokens] recovery limit reached\033[0m")
            return

        # 正常完成：将助手回复追加到消息历史
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        # ── 工具执行 ──
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            # 显示工具名和关键参数（黄色）
            print(f"\033[33m$ {block.name}: {_tool_input_summary(block)}\033[0m")
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            # 工具结果：粗体品红标签 + 品红内容
            print(f"\033[1;35m[{block.name} -> Tool Calling Result]\033[0m \033[35m{str(output)[:200]}\033[0m")
            results.append({"type": "tool_result",
                            "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})

        # s10: 每轮后刷新 context + prompt
        context = update_context(context, messages)
        system = get_system_prompt(context)


# ── 入口：交互式 REPL ──────────────────────────────────
# 流程：读取输入 → agent_loop（含错误恢复）→ 打印助理回复 → 循环
if __name__ == "__main__":
    print("s11: error recovery")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []  # 消息历史
    context = update_context({}, [])  # s10: 启动时初始化 context
    while True:
        try:
            query = input("\033[36ms11 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        turn_start = len(history)
        history.append({"role": "user", "content": query})
        agent_loop(history, context)
        context = update_context(context, history)  # 刷新 context
        # 打印本轮新增的助理文本回复（蓝色）
        for msg in history[turn_start:]:
            if msg.get("role") != "assistant":
                continue
            for block in msg["content"]:
                if getattr(block, "type", None) == "text":
                    print(f"\033[34m{block.text}\033[0m")
        print()
