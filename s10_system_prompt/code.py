#!/usr/bin/env python3
"""
s10: System Prompt — 运行时按需组装 system prompt，带确定性缓存。

AI 编程智能体的 prompt 管理：
1. PROMPT_SECTIONS: 按主题分片存储 prompt 片段（identity/tools/workspace/memory）
2. assemble_system_prompt(context): 根据真实状态选择并拼接片段
3. get_system_prompt(context): 通过 json.dumps 做确定性缓存，状态不变则不重新拼接
4. agent_loop 使用 get_system_prompt(context) 替代硬编码的 SYSTEM 字符串
5. 记忆片段仅在 .memory/MEMORY.md 真实存在时才加载（基于真实状态，非关键词）

Run / 运行: python s10_system_prompt/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

Changes from s09 / 相对 s09 的变更:
  - PROMPT_SECTIONS: topic-keyed dict of prompt fragments
  - assemble_system_prompt(context): select + join sections by real state
  - get_system_prompt(context): deterministic cache via json.dumps
  - agent_loop uses get_system_prompt(context) instead of hardcoded SYSTEM

Memory section loads when .memory/MEMORY.md exists (real state, not keywords).
"""

import os, subprocess, json
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
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"  # s09: 记忆索引文件
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]


# ═══════════════════════════════════════════════════════════
#  NEW in s10: Prompt 片段系统 + 确定性缓存
#  PROMPT_SECTIONS: 按主题分片存储 → assemble 按状态拼接 → 缓存避免重复组装
# ═══════════════════════════════════════════════════════════

# Prompt 片段字典：按主题分片，每个片段是独立可替换的文本块
PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.", # 角色设定
    "tools": "Available tools: bash, read_file, write_file.", # 工具列表
    "workspace": f"Working directory: {WORKDIR}", # 工作目录
    "memory": "Relevant memories are injected below when available.", # 记忆提示（仅当 MEMORY.md 存在且有内容时加载）
}


def assemble_system_prompt(context: dict) -> str:
    """根据当前 context 选择并拼接 prompt 片段。
    始终加载：identity + tools + workspace。
    条件加载：memory（仅当 MEMORY.md 存在且有内容时）。"""
    sections = []

    # 始终加载：身份、工具、工作目录
    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(PROMPT_SECTIONS["tools"])
    sections.append(PROMPT_SECTIONS["workspace"])

    # 条件加载：仅当 .memory/MEMORY.md 存在且有内容时才注入记忆
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")

    return "\n\n".join(sections)


# 缓存：上次的 context_key 和拼接结果
_last_context_key = None
_last_prompt = None


def get_system_prompt(context: dict) -> str:
    """带缓存的 system prompt 获取。仅在 context 变化时重新拼接。
    使用 json.dumps 做确定性序列化（而非 Python 的 hash()，它有进程随机化问题）。
    此缓存只避免同一进程内的冗余字符串拼接。
    真正的 Claude Code 还通过稳定的 section 排序和 SYSTEM_PROMPT_DYNAMIC_BOUNDARY
    保护了 API 层的 prompt cache。"""
    global _last_context_key, _last_prompt
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        print("  \033[90m[cache hit] system prompt unchanged\033[0m")
        return _last_prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    # 终端显示加载了哪些部分（绿色）
    loaded = ["identity", "tools", "workspace"]
    if context.get("memories"):
        loaded.append("memory")
    print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
    return _last_prompt


# ═══════════════════════════════════════════════════════════
#  工具实现 — 精简版（3 个工具，聚焦 prompt 组装系统）
# ═══════════════════════════════════════════════════════════

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


# ── 工具定义（精简：3 个基础工具）───────────────────
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

# ── 工具分发映射 ─────────────────────────────────────
TOOL_HANDLERS = {"bash": run_bash, "read_file": run_read, "write_file": run_write}


# ═══════════════════════════════════════════════════════════
#  s10: Context 系统 — 从真实状态推导，驱动 prompt 组装
# ═══════════════════════════════════════════════════════════

def update_context(context: dict, messages: list) -> dict:
    """从真实状态推导 context：启用了哪些工具、记忆文件是否存在。
    基于真实文件系统状态，非关键词或猜测。"""
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
#  agent_loop — 使用组装式 system prompt 替代硬编码 SYSTEM
#  每轮工具执行后重新评估 context + 刷新 prompt
# ═══════════════════════════════════════════════════════════

def agent_loop(messages: list, context: dict):
    """智能体主循环 + 动态 prompt 组装。
    流程：
    1. 从 context 组装 system prompt（带缓存，context 不变则复用）
    2. LLM 调用 → 工具执行
    3. 每轮工具执行后 → update_context() 重新评估状态 → 刷新 prompt
    """
    system = get_system_prompt(context)
    while True:
        response = client.messages.create(
            model=MODEL, system=system, messages=messages,
            tools=TOOLS, max_tokens=8000)
        # 将助手回复追加到消息历史
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return

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

        # s10: 每轮工具执行后重新评估 context + 刷新 prompt
        context = update_context(context, messages)
        system = get_system_prompt(context)


# ── 入口：交互式 REPL ──────────────────────────────────
# 流程：初始化 context → 读取输入 → agent_loop（动态 prompt）→ 刷新 context → 打印回复 → 循环
if __name__ == "__main__":
    print("s10: system prompt — runtime assembly")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []  # 消息历史
    context = update_context({}, [])  # s10: 启动时从真实状态初始化 context
    while True:
        try:
            query = input("\033[36ms10 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history, context)
        context = update_context(context, history)  # s10: 每轮后刷新 context
        # 打印模型最终文本回复（蓝色）
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(f"\033[34m{block.text}\033[0m")
        print()
