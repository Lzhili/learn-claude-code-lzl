#!/usr/bin/env python3
"""
s08_context_compact.py - 上下文压缩（Context Compact）

在 LLM 调用前插入四层压缩管道（从廉价到昂贵）：

    L1: snip_compact       — 消息数超过 50 时修剪中间消息（0 API 调用）
    L2: micro_compact      — 旧 tool_result 替换为占位符（0 API 调用）
    L3: tool_result_budget — 大结果持久化到磁盘，只留预览（0 API 调用）
    L4: compact_history    — LLM 全文摘要（1 API 调用）
    Emergency: reactive_compact — API 返回 prompt_too_long 时紧急压缩

    核心原则：廉价的先跑，昂贵的最后跑。
    执行顺序与 CC 源码一致：budget → snip → micro → auto。

    ┌─────────────────────────────────────────────────────────────┐
    │  messages[]                                                 │
    │    ↓                                                        │
    │  L3 budget ─→ L1 snip ─→ L2 micro ─→ [token > threshold?]  │
    │                                      ├─ No  → LLM          │
    │                                      └─ Yes → L4 summary   │
    │                                              ↓              │
    │                                          LLM call           │
    │                                    [prompt_too_long?]        │
    │                                      └─ Yes → reactive      │
    └─────────────────────────────────────────────────────────────┘

Four-layer compaction pipeline inserted before LLM calls:

    L1: snip_compact      — trim middle messages when count > 50
    L2: micro_compact     — replace old tool_results with placeholders
    L3: tool_result_budget — persist large results to disk
    L4: compact_history   — LLM full summary (1 API call)

    Emergency: reactive_compact — when API still returns prompt_too_long

Core principle: cheap first, expensive last.
Execution order matches CC source: budget → snip → micro → auto.

Builds on s07 (skill loading). Usage / 用法:

    python s08_context_compact/code.py
    Needs: pip install anthropic python-dotenv + ANTHROPIC_API_KEY in .env
"""

import os, subprocess, json, time
from pathlib import Path

# ── 终端中文输入兼容性 ──────────────────────────────────
# macOS 的 libedit 在处理中文输入时有退格问题，readline 配置修复它
try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')   # 关闭特殊字符绑定，避免中文被误解析
    readline.parse_and_bind('set input-meta on')                # 开启输入 8-bit 元位，支持 UTF-8 多字节
    readline.parse_and_bind('set output-meta on')               # 开启输出 8-bit 元位
    readline.parse_and_bind('set convert-meta off')             # 不转换 meta 字符，保持原始字节
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

# ── 初始化 Anthropic 客户端 ────────────────────────────
# 若用户配置了自定义 BASE_URL（如代理/中转），需清理 AUTH_TOKEN 避免冲突
load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"): os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()  # 工作目录
SKILLS_DIR = WORKDIR / "skills"  # s07: 技能文件目录
TRANSCRIPT_DIR = WORKDIR / ".transcripts"  # s08: 压缩前对话归档目录
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"  # s08: 大结果持久化目录
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
CURRENT_TODOS: list[dict] = []  # s05: 内存中的任务列表

# ═══════════════════════════════════════════════════════════
#  FROM s07: 技能加载系统 — 来自 s07（未改动）
# ═══════════════════════════════════════════════════════════
def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 SKILL.md 中的 YAML frontmatter。返回 (meta, body)。"""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, parts[2].strip()

# 技能注册表：启动时构建，load_skill 通过查表安全获取
SKILL_REGISTRY: dict[str, dict] = {}

def _scan_skills():
    """启动时扫描 skills/ 目录，填入注册表。"""
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", d.name)
            desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            SKILL_REGISTRY[name] = {"name": name, "description": desc, "content": raw}

_scan_skills()  # 模块加载时立即扫描

def list_skills() -> str:
    """列出所有已注册技能的名称和一行描述。"""
    if not SKILL_REGISTRY:
        return "(no skills found)"
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())

def load_skill(name: str) -> str:
    """加载指定技能的完整 SKILL.md 内容。通过注册表查表获取。"""
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]

def build_system() -> str:
    """构建 SYSTEM prompt。启动时注入技能目录（廉价——仅名称+描述）。"""
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )

SYSTEM = build_system()  # s07: SYSTEM 在启动时动态生成

# s06: 子 Agent 独立系统提示词
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)


# ═══════════════════════════════════════════════════════════
#  FROM s02-s07 (unchanged): 基础工具 — 来自 s02/s07（未改动）
# ═══════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    """路径安全校验。防止模型读写工作目录之外的文件。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR): raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    """执行 shell 命令并返回输出。权限检查已上移到 PreToolUse 钩子。"""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired: return "Error: Timeout (120s)"

def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容。可指定行数上限，超出则截断。"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines): lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e: return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    """写入文件。自动创建父目录。"""
    try:
        file_path = safe_path(path); file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content); return f"Wrote {len(content)} bytes to {path}"
    except Exception as e: return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    """精确替换文件中的文本。仅替换首次出现。"""
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text: return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e: return f"Error: {e}"

def run_glob(pattern: str) -> str:
    """按 glob 模式匹配文件。结果限制在工作目录内。"""
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e: return f"Error: {e}"

def run_todo_write(todos: list) -> str:
    """todo_write 工具实现。管理内存中的任务列表并终端显示。"""
    global CURRENT_TODOS
    for i, t in enumerate(todos):
        if "content" not in t or "status" not in t:
            return f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return f"Error: todos[{i}] has invalid status '{t['status']}'"
    CURRENT_TODOS = todos
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "\033[36m▸\033[0m", "completed": "\033[32m✓\033[0m"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"

# s06: 辅助函数，从 message content 中提取纯文本
def extract_text(content) -> str:
    """从消息的 content 块中提取文本。处理 list 和 str 两种类型。"""
    if not isinstance(content, list): return str(content)
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")


# ═══════════════════════════════════════════════════════════
#  FROM s06-s07 (unchanged): 子智能体 — 来自 s06/s07（未改动）
# ═══════════════════════════════════════════════════════════

# 子 Agent 工具列表：只有 5 个基础工具，无 task（禁止递归）
SUB_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
]
# 子 Agent 工具分发映射
SUB_HANDLERS = {"bash": run_bash, "read_file": run_read, "write_file": run_write,
                "edit_file": run_edit, "glob": run_glob}

def spawn_subagent(task: str) -> str:
    """派生一个子智能体，拥有全新 messages[]，仅返回最终摘要。
    安全措施：上下文隔离 + 30 轮限制 + 无递归 + 钩子继承。"""
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    messages = [{"role": "user", "content": task}]  # 全新上下文
    for _ in range(30):  # 安全限制
        response = client.messages.create(model=MODEL, system=SUB_SYSTEM,
            messages=messages, tools=SUB_TOOLS, max_tokens=8000)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # 子 Agent 同样走钩子系统
                blocked = trigger_hooks("PreToolUse", block)
                if blocked:
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": str(blocked)})
                    continue
                handler = SUB_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
                trigger_hooks("PostToolUse", block, output)
                # 子 Agent 输出用灰色 [sub] 前缀标记
                print(f"  \033[90m[sub] {block.name}: {str(output)[:100]}\033[0m")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})
    # 安全兜底：轮数耗尽时向前查找助理文本
    result = extract_text(messages[-1]["content"])
    if not result:
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                result = extract_text(msg["content"])
                if result:
                    break
        if not result:
            result = "Subagent stopped after 30 turns without final answer."
    print(f"\033[35m[Subagent done]\033[0m")
    return result  # 只返回摘要，完整 messages[] 被丢弃


# ═══════════════════════════════════════════════════════════
#  NEW in s08: 四层压缩管道（Four-Layer Compaction Pipeline）
#  核心原则：廉价的先跑（0 API 调用），昂贵的最后跑（1 API 调用）
#  执行顺序与 CC 源码一致：L3 budget → L1 snip → L2 micro → L4 auto
# ═══════════════════════════════════════════════════════════

CONTEXT_LIMIT = 50000       # token 阈值，超过则触发 L4 摘要压缩
KEEP_RECENT = 3             # 保留最近 N 个 tool_result 不压缩
PERSIST_THRESHOLD = 30000   # 输出超过此长度则持久化到磁盘,30000

def estimate_size(msgs):
    """估算 messages 的 token 量（简化：用字符串长度近似）。"""
    return len(str(msgs))


# ── L1: snipCompact —— 消息数超过上限时修剪中间 ─────

def snip_compact(messages, max_messages=50):
    """L1 修剪：当消息数超过 max_messages 时，保留头尾，中间替换为占位消息。"""
    if len(messages) <= max_messages: return messages
    keep_head, keep_tail = 3, max_messages - 3 
    snipped = len(messages) - keep_head - keep_tail
    # 保留头部 3 条（初始上下文）和尾部 47 条（当前工作），中间裁掉的部分用占位符替代
    return messages[:keep_head] + [{"role": "user", "content": f"[snipped {snipped} messages]"}] + messages[-keep_tail:]


# ── L2: microCompact —— 旧 tool_result 替换为占位符 ──

def collect_tool_results(messages):
    """收集 messages 中所有 tool_result 块的位置信息。"""
    blocks = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list): continue
        for bi, block in enumerate(msg["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((mi, bi, block))
    return blocks # 返回 [(message_index, block_index, block), ...] 列表

def micro_compact(messages):
    """L2 微压缩：保留最近 KEEP_RECENT 个 tool_result，其余替换为占位符。"""
    tool_results = collect_tool_results(messages)
    if len(tool_results) <= KEEP_RECENT: return messages
    for _, _, block in tool_results[:-KEEP_RECENT]: # 保留最近 KEEP_RECENT 个，其他替换
        if len(block.get("content", "")) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages


# ── L3: toolResultBudget —— 大结果持久化到磁盘 ───────

def persist_large_output(tool_use_id, output):
    """将超大 tool_result 写入磁盘，只返回预览和路径。"""
    if len(output) <= PERSIST_THRESHOLD: return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not path.exists(): path.write_text(output) 
    return f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n</persisted-output>"

def tool_result_budget(messages, max_bytes=200_000):
    """L3 预算：检查最后一轮 tool_result 总大小，超出则持久化最大的结果到磁盘。"""
    last = messages[-1] if messages else None
    if not last or last.get("role") != "user" or not isinstance(last.get("content"), list): return messages
    # 收集最后一轮所有 tool_result 块，计算总大小
    blocks = [(i, b) for i, b in enumerate(last["content"]) if isinstance(b, dict) and b.get("type") == "tool_result"]
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_bytes: return messages
    # 按大小排序，优先持久化最大的结果
    ranked = sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True)
    for _, block in ranked:
        if total <= max_bytes: break
        content = str(block.get("content", ""))
        if len(content) <= PERSIST_THRESHOLD: continue
        tid = block.get("tool_use_id", "unknown")
        block["content"] = persist_large_output(tid, content)
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    return messages


# ── L4: autoCompact —— LLM 全文摘要 ──────────────────

def write_transcript(messages):
    """将完整 messages 归档到 .transcripts/ 目录。"""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages: f.write(json.dumps(msg, default=str) + "\n")
    return path

def summarize_history(messages):
    """调用 LLM 将整个对话历史总结为一段精简摘要。保留目标、发现、文件、待办、约束 5 项。"""
    conversation = json.dumps(messages, default=str)[:80000]
    prompt = ("Summarize this coding-agent conversation so work can continue.\n"
              "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
              "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n" + conversation)
    response = client.messages.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000)
    return "\n".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", None) == "text").strip() or "(empty summary)"

def compact_history(messages):
    """L4 摘要压缩：归档原始对话 → LLM 摘要 → 返回压缩后的单条消息。"""
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


# ── Emergency: reactiveCompact —— API 报错时紧急压缩 ─

def reactive_compact(messages):
    """紧急压缩：API 返回 prompt_too_long 时的兜底措施。
    归档 + 摘要 + 保留最近 5 条原始消息。"""
    transcript = write_transcript(messages)
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *messages[-5:]]


# ═══════════════════════════════════════════════════════════
#  FROM s07: 工具定义 — 来自 s07 + s08 新增 compact 工具
# ═══════════════════════════════════════════════════════════

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "todo_write", "description": "Create and manage a task list for your current coding session.",
     "input_schema": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"]}}}, "required": ["todos"]}},
    {"name": "task", "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
     "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}},
    {"name": "load_skill", "description": "Load the full content of a skill by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    # s08 新增：compact 工具 —— 触发 compact_history，不是空操作
    {"name": "compact", "description": "Summarize earlier conversation to free context space.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}}},
]

# ── 工具分发映射（s08 compact 不在映射中，因为它在 agent_loop 中有特殊处理）──
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
    "task": spawn_subagent, "load_skill": load_skill,
}

# ═══════════════════════════════════════════════════════════
#  FROM s04 (unchanged): 钩子系统 — 简化版
# ═══════════════════════════════════════════════════════════

HOOKS = {"PreToolUse": [], "PostToolUse": []}
def trigger_hooks(event, *args):
    """触发钩子。简化版：只有 PreToolUse 和 PostToolUse 两个事件。"""
    for cb in HOOKS[event]:
        r = cb(*args)
        if r is not None: return r
    return None

DENY_LIST = ["rm -rf /", "sudo", "shutdown"]
def permission_hook(block):
    """PreToolUse 钩子：权限检查。"""
    if block.name == "bash":
        for p in DENY_LIST:
            if p in block.input.get("command", ""): return "Permission denied"
    return None
def log_hook(block):
    """PreToolUse 钩子：记录每次工具调用（灰色日志）。"""
    print(f"\033[90m[HOOK] {block.name}\033[0m")
    return None

HOOKS["PreToolUse"].append(permission_hook)
HOOKS["PreToolUse"].append(log_hook)


def _tool_input_summary(block) -> str:
    """提取工具调用的关键参数摘要，用于终端显示。"""
    inputs = block.input
    if block.name == "bash":
        return inputs.get("command", "")
    if block.name == "glob":
        return inputs.get("pattern", "")
    if block.name == "todo_write":
        count = len(inputs.get("todos", []))
        return f"{count} tasks"
    if block.name == "task":
        desc = inputs.get("description", "")
        return desc[:60] + ("..." if len(desc) > 60 else "")
    if block.name == "load_skill":
        return inputs.get("name", "")
    if block.name == "compact":
        return inputs.get("focus", "full context")
    return inputs.get("path", "")


# ═══════════════════════════════════════════════════════════
#  agent_loop — s08 核心：LLM 调用前运行压缩管道
#  新增：四层预处理器 + compact 工具特殊处理 + react 重试
# ═══════════════════════════════════════════════════════════

MAX_REACTIVE_RETRIES = 1  # react 紧急压缩最大重试次数

def agent_loop(messages: list):
    """智能体主循环 + 压缩管道。
    流程：
    1. LLM 调用前：L3 budget → L1 snip → L2 micro（3 次 0 API 调用）
    2. 若 token 仍超限 → L4 LLM 摘要压缩（1 API 调用）
    3. 正常调用 LLM；若 prompt_too_long → 紧急压缩 + 重试
    4. 工具执行中若遇到 compact → 立即压缩并开始新一轮
    """
    reactive_retries = 0
    while True:
        # s08: 三层预处理器（0 API 调用，廉价先跑）
        # 执行顺序与 CC 源码一致：budget → snip → micro
        messages[:] = tool_result_budget(messages)    # L3: 大结果持久化
        messages[:] = snip_compact(messages)          # L1: 修剪中间消息
        messages[:] = micro_compact(messages)         # L2: 旧结果占位符

        # s08: token 仍超限 → L4 LLM 摘要压缩（1 API 调用，昂贵）
        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages)

        try:
            response = client.messages.create(model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000)
            reactive_retries = 0  # 成功调用后重置
        except Exception as e:
            # 紧急压缩：API 仍报 prompt_too_long 时触发
            if ("prompt_too_long" in str(e).lower() or "too many tokens" in str(e).lower()) and reactive_retries < MAX_REACTIVE_RETRIES:
                print("[reactive compact]")
                messages[:] = reactive_compact(messages)
                reactive_retries += 1
                continue
            raise

        # 将助手回复追加到消息历史
        messages.append({"role": "assistant", "content": response.content})
        # 若模型未调用工具，循环结束
        if response.stop_reason != "tool_use": return

        results = []
        for block in response.content:
            if block.type != "tool_use": continue

            # 显示工具名和关键参数（黄色）
            print(f"\033[33m$ {block.name}: {_tool_input_summary(block)}\033[0m")

            # s08: compact 工具特殊处理 —— 触发压缩，结束当前轮
            if block.name == "compact":
                messages[:] = compact_history(messages)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "[Compacted. Conversation history has been summarized.]"})
                messages.append({"role": "user", "content": results})
                break  # 结束当前轮，用压缩后的上下文重新开始

            # PreToolUse 钩子（权限检查 + 日志）
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(blocked)})
                continue
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            trigger_hooks("PostToolUse", block, output)
            # 工具结果：粗体品红标签 + 品红内容
            print(f"\033[1;35m[{block.name} -> Tool Calling Result]\033[0m \033[35m{str(output)[:200]}\033[0m")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        else:
            # 正常路径：未调用 compact，追加 results 并继续循环
            messages.append({"role": "user", "content": results})
            continue
        # compact 被调用：results 已在上方追加，跳过正常路径
        continue


# ── 入口：交互式 REPL ──────────────────────────────────
# 流程：读取用户输入 → agent_loop（内含压缩管道）→ 打印最终回复 → 循环
if __name__ == "__main__":
    print("s08: Context Compact — four-layer compaction pipeline")
    print("输入问题，回车发送。输入 q 退出。\n")
    history = []  # 消息历史，贯穿整个交互会话
    while True:
        try: query = input("\033[36ms08 >> \033[0m")
        except (EOFError, KeyboardInterrupt): break
        if query.strip().lower() in ("q", "exit", ""): break
        history.append({"role": "user", "content": query})
        agent_loop(history)  # 进入含压缩管道的工具调用循环
        # 打印模型最终文本回复（蓝色）
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(f"\033[34m{block.text}\033[0m")
        print()
