#!/usr/bin/env python3
"""
s06: Subagent — 派生子智能体，用全新的 messages[] 实现上下文隔离。

AI 编程智能体的任务分解能力：
1. task 工具 —— 父 Agent 将复杂子问题派发给子 Agent，只接收最终摘要
2. 子 Agent 拥有全新 messages[] —— 中间结果不进父上下文，避免污染
3. 安全限制：子 Agent 最多 30 轮 + 无 task 工具，防止递归派生
4. 循环本身不变：新工具通过 TOOL_HANDLERS 自动分发

  Parent Agent / 父智能体                 Subagent / 子智能体
  +------------------+                  +------------------+
  | messages=[...]   |                  | messages=[task]  | <-- fresh
  |                  |   dispatch       |                  |
  | tool: task       | ---------------> | own while loop   |
  |   prompt="..."   |                  |   bash/read/...  |
  |                  |   summary only   |   (max 30 turns) |
  | result = "..."   | <--------------- | return last text |
  +------------------+                  +------------------+
        ^                                      |
        |       intermediate results DISCARDED  |
        +--------------------------------------+

  Subagent tools: bash, read, write, edit, glob (NO task — no recursion)
  子 Agent 工具：bash, read, write, edit, glob（无 task —— 禁止递归）

Changes from s05 / 相对 s05 的变更:
  + task tool + spawn_subagent() with fresh messages[]
  + Safety limit: max 30 turns per subagent
  + extract_text() helper
  Subagent cannot spawn sub-subagents (no task tool in sub_tools).
  Main loop unchanged: task auto-dispatches via TOOL_HANDLERS.

Run / 运行: python s06_subagent/code.py
Needs: pip install anthropic python-dotenv + ANTHROPIC_API_KEY in .env
"""

import os, subprocess
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
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()  # 工作目录，作为路径安全校验的根
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
CURRENT_TODOS: list[dict] = []  # s05: 内存中的任务列表

# 父 Agent 系统提示词：引导将复杂子问题委派给子 Agent
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "For complex sub-problems, use the task tool to spawn a subagent."
)

# s06: 子 Agent 有自己独立的系统提示词 —— 无 task 工具，禁止再委派
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)


# ═══════════════════════════════════════════════════════════
#  FROM s02-s05 (unchanged): 工具实现 — 来自 s02/s05（未改动）
# ═══════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    """路径安全校验。将用户输入的相对路径解析为绝对路径，
    若路径逃逸出工作目录则抛出异常，防止模型读写系统文件。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    """执行 shell 命令并返回输出。权限检查已上移到 PreToolUse 钩子。"""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int | None = None) -> str:
    """读取文件内容。可指定行数上限，超出则截断并显示剩余行数。"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    """写入文件。自动创建父目录，返回写入字节数。"""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    """精确替换文件中的文本。仅替换首次出现，old_text 不存在时报错。"""
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    """按 glob 模式匹配文件。结果限制在工作目录内，防止路径遍历。"""
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

def run_todo_write(todos: list) -> str:
    """todo_write 工具实现。接收带状态的任务列表，保存在内存中并终端显示。"""
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

# s06: 辅助函数，从 message content 中提取纯文本（用于子 Agent 返回摘要）
def extract_text(content) -> str:
    """从消息的 content 块中提取文本。处理 list 和 str 两种类型。"""
    if not isinstance(content, list):
        return str(content)
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text") # list 中可能混有 tool_use 块，过滤后提取文本

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
]

# ── 父 Agent 工具分发映射（s06 新增 task，后面 append）──
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
}


# ═══════════════════════════════════════════════════════════
#  NEW in s06: 子智能体（Subagent）—— 全新 messages[]，仅返回摘要
#  核心设计：上下文隔离 + 安全限制 + 无递归
# ═══════════════════════════════════════════════════════════

# 子 Agent 的工具列表：只有 5 个基础工具，无 task（禁止递归派生）
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
# 无 "task" 工具 —— 禁止子 Agent 再派生孙子 Agent

# 子 Agent 的工具分发映射
SUB_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}

def spawn_subagent(description: str) -> str:
    """派生一个子智能体，拥有全新 messages[]，仅返回最终摘要。
    安全措施：
    1. 上下文隔离：子 Agent 用全新 messages=[task]，中间结果不回传父 Agent
    2. 轮数限制：最多 30 轮，防止无限循环
    3. 无递归：SUB_TOOLS 不含 task 工具，禁止派生孙子 Agent
    4. 钩子继承：子 Agent 同样走 PreToolUse/PostToolUse 钩子
    """
    print(f"\n\033[35m[Subagent spawned]\033[0m") # 派生子 Agent 的提示（品红色）
    messages = [{"role": "user", "content": description}]  # 全新上下文，无历史包袱

    for _ in range(30):  # 安全限制：最多 30 轮
        response = client.messages.create(
            model=MODEL, system=SUB_SYSTEM,
            messages=messages, tools=SUB_TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # 子 Agent 同样走钩子系统（权限检查 + 日志）
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
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
        messages.append({"role": "user", "content": results})

    # 安全兜底：若 30 轮耗尽时最后一条消息是 user 的（意味着没有textblock），向前查找assistant的textblock作为结果摘要；若没有则返回默认提示
    result = extract_text(messages[-1]["content"])
    if not result:
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                result = extract_text(msg["content"])
                if result:
                    break
        if not result:
            result = "Subagent stopped after 30 turns without final answer."
    print(f"\033[35m[Subagent done]\033[0m") # 子 Agent 结束的提示（品红色）
    return result  # 只返回摘要，完整的 messages[] 被丢弃

# 将 task 工具追加到父 Agent 的工具列表和分发映射中
TOOLS.append({
    "name": "task",
    "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
    "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]},
})
TOOL_HANDLERS["task"] = spawn_subagent


# ═══════════════════════════════════════════════════════════
#  FROM s04 (unchanged): 钩子系统 — 来自 s04（未改动）
#  父 Agent 和子 Agent 共享同一套钩子（permission_hook 等）
# ═══════════════════════════════════════════════════════════

# 钩子注册表
HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}

def register_hook(event: str, callback):
    """注册钩子：将回调函数绑定到指定事件。"""
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    """触发钩子：依次调用该事件下所有回调。
    若任一回调返回非 None，立即短路返回该值（用于拦截工具调用）。"""
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:  # 教学简化：非 None 返回值表示拦截
            return result
    return None

DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]

def permission_hook(block):
    """PreToolUse 钩子：权限检查。拦截硬拒绝列表中的危险命令。"""
    if block.name == "bash":
        for p in DENY_LIST:
            if p in block.input.get("command", ""):
                print(f"\n\033[31m⛔ Blocked: '{p}'\033[0m")
                return "Permission denied"
    return None

def log_hook(block):
    """PreToolUse 钩子：记录每次工具调用（灰色日志）。"""
    print(f"\033[90m[HOOK] {block.name}\033[0m")
    return None

def context_inject_hook(query: str):
    """UserPromptSubmit 钩子：用户输入前打印工作目录信息。"""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None

def summary_hook(messages: list):
    """Stop 钩子：循环退出时打印本次会话的工具调用统计。"""
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None

# ── 注册钩子 ─────────────────────────────────────────
register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("Stop", summary_hook)


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
    return inputs.get("path", "")


# ═══════════════════════════════════════════════════════════
#  agent_loop — 与 s05 结构一致 + nag 提醒 + task 自动分发
#  task 工具触发 spawn_subagent()，父循环不感知子 Agent 内部细节
# ═══════════════════════════════════════════════════════════

rounds_since_todo = 0

def agent_loop(messages: list):
    """智能体主循环 + 钩子系统 + Nag 提醒 + 子 Agent 分发。
    流程：
    1. 若连续 3 轮未调用 todo_write → 注入提醒消息
    2. 将完整消息历史发送给 LLM
    3. 若 tool_use → 经 PreToolUse 钩子 → 执行工具（含 task 子 Agent 派生）
       → PostToolUse 钩子 → 重置计数器 → 追加结果 → 循环
    """
    global rounds_since_todo
    while True:
        # s05: Nag 提醒 —— 连续 3 轮未更新 todos 则注入提醒
        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0

        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # 将助手回复追加到消息历史
        messages.append({"role": "assistant", "content": response.content})

        # 若模型未调用工具，触发 Stop 钩子后退出
        if response.stop_reason != "tool_use":
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        rounds_since_todo += 1
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # 显示工具名和关键参数（黄色）
            print(f"\033[33m$ {block.name}: {_tool_input_summary(block)}\033[0m")

            # PreToolUse 钩子（权限检查 + 日志）
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            # 工具结果：粗体品红标签 + 品红内容
            print(f"\033[1;35m[{block.name} -> Tool Calling Result]\033[0m \033[35m{str(output)[:200]}\033[0m")

            # PostToolUse 钩子
            trigger_hooks("PostToolUse", block, output)

            # s05: todo_write 被调用时重置 nag 计数器
            if block.name == "todo_write":
                rounds_since_todo = 0

            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": output})

        # 将工具结果作为用户消息追加，触发下一轮循环
        messages.append({"role": "user", "content": results})


# ── 入口：交互式 REPL ──────────────────────────────────
# 流程：触发 UserPromptSubmit 钩子 → 读取输入 → agent_loop → 打印最终回复 → 循环
if __name__ == "__main__":
    print("s06: Subagent — spawn sub-agents with fresh context, summary only")
    print("Type a question, press Enter. Type q to quit.\n")

    history = []  # 消息历史，贯穿整个交互会话
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)  # s04: 用户输入前触发钩子
        history.append({"role": "user", "content": query})
        agent_loop(history)  # 进入含子 Agent 分发的工具调用循环
        # 打印模型最终文本回复（蓝色）
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(f"\033[34m{block.text}\033[0m")
        print()
