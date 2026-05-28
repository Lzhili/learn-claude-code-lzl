#!/usr/bin/env python3
"""
s07: Skill Loading — 两级按需知识注入。

AI 编程智能体的知识管理：
1. Layer 1（廉价层，始终存在）：SYSTEM prompt 含技能名称 + 一行描述（~100 tokens/技能）
2. Layer 2（昂贵层，按需加载）：Agent 调用 load_skill("code-review") → 完整的 SKILL.md
   内容通过 tool_result 注入（~2000 tokens/技能）
3. 启动时扫描 skills/ 目录，构建 SKILL_REGISTRY，SYSTEM 自动生成目录
4. 循环本身不变：load_skill 通过 TOOL_HANDLERS 自动分发

  Layer 1 (cheap, always present):
    SYSTEM prompt includes skill names + one-line descriptions (~100 tokens/skill)
    "Skills available: agent-builder, code-review, mcp-builder, pdf"

  Layer 2 (expensive, on demand):
    Agent calls load_skill("code-review") → full SKILL.md content
    injected via tool_result (~2000 tokens/skill)

  skills/
    agent-builder/SKILL.md
    code-review/SKILL.md
    mcp-builder/SKILL.md
    pdf/SKILL.md

Changes from s06 / 相对 s06 的变更:
  + build_system() — scan skills/ dir at startup, inject catalog into SYSTEM
  + load_skill(name) — return full SKILL.md content via tool_result
  + SKILLS_DIR config
  Loop unchanged: load_skill auto-dispatches via TOOL_HANDLERS.

Run / 运行: python s07_skill_loading/code.py
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
SKILLS_DIR = WORKDIR / "skills"  # s07: 技能文件存放目录
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
CURRENT_TODOS: list[dict] = []  # s05: 内存中的任务列表

# ═══════════════════════════════════════════════════════════
#  NEW in s07: 两级技能加载系统
#  Layer 1: 启动时扫描 skills/ → 注入 SYSTEM 目录（廉价，始终存在）
#  Layer 2: load_skill() 返回完整 SKILL.md（昂贵，按需触发）
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

# 技能注册表：启动时构建，load_skill 通过查表安全获取（无路径遍历风险）
SKILL_REGISTRY: dict[str, dict] = {}

def _scan_skills():
    """启动时扫描 skills/ 目录，将每个 SKILL.md 的 name/description/content 填入注册表。"""
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

def build_system() -> str:
    """构建 SYSTEM prompt。启动时将技能目录注入（廉价——仅名称+描述）。"""
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )

SYSTEM = build_system()  # s07: SYSTEM prompt 在启动时动态生成

# s06: 子 Agent 独立系统提示词 —— 无 skill loading，无 task
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)


# ═══════════════════════════════════════════════════════════
#  FROM s02-s06 (unchanged): 工具实现 — 来自 s02/s06（未改动）
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
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")


# ═══════════════════════════════════════════════════════════
#  FROM s06 (unchanged): 子智能体 — 来自 s06（未改动）
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

def spawn_subagent(description: str) -> str:
    """派生一个子智能体，拥有全新 messages[]，仅返回最终摘要。
    安全措施：上下文隔离 + 30 轮限制 + 无递归 + 钩子继承。"""
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    messages = [{"role": "user", "content": description}]  # 全新上下文
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
#  NEW in s07: load_skill — 运行时按需加载完整技能内容
#  通过 SKILL_REGISTRY 查表获取，无路径遍历风险
# ═══════════════════════════════════════════════════════════

def load_skill(name: str) -> str:
    """加载指定技能的完整 SKILL.md 内容。
    通过 SKILL_REGISTRY 查表获取，避免直接读取文件系统。"""
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]


# ── 工具注册表 —— s02 到 s07 的所有工具 ──────────────
# s07: 新增 load_skill 工具（目录已在 SYSTEM 中，此工具加载完整内容）

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
    # s07 新增：技能加载工具（目录已在 SYSTEM prompt 中，此工具加载完整内容）
    {"name": "load_skill", "description": "Load the full content of a skill by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
]

# ── 工具分发映射（s07 新增 load_skill）────────────────
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
    "task": spawn_subagent, "load_skill": load_skill,
}


# ═══════════════════════════════════════════════════════════
#  FROM s04 (unchanged): 钩子系统 — 来自 s04（未改动）
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
    if block.name == "load_skill":
        return inputs.get("name", "")
    return inputs.get("path", "")


# ═══════════════════════════════════════════════════════════
#  agent_loop — 与 s05/s06 结构一致 + nag 提醒
#  s07 变更：nag 提醒改为插入到最后一条 user 消息的 content 列表中
# ═══════════════════════════════════════════════════════════

rounds_since_todo = 0

def agent_loop(messages: list):
    """智能体主循环 + 钩子系统 + Nag 提醒 + 技能分发。
    流程：
    1. 若连续 3 轮未调用 todo_write → 在最后一条 user 消息中注入提醒
    2. 将完整消息历史发送给 LLM
    3. 若 tool_use → 经 PreToolUse 钩子 → 执行工具（含 load_skill/task）
       → PostToolUse 钩子 → 重置计数器 → 追加结果 → 循环
    """
    global rounds_since_todo
    while True:
        # s05: Nag 提醒 —— 插入到最后一条 user 消息的内容中（内联注入）
        if rounds_since_todo >= 3 and messages:
            last = messages[-1]
            if last["role"] == "user" and isinstance(last.get("content"), list):
                last["content"].append({
                    "type": "text",
                    "text": "<reminder>Update your todos.</reminder>",
                })

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
    print("s07: Skill Loading — catalog in SYSTEM, content on demand")
    print("Type a question, press Enter. Type q to quit.\n")

    history = []  # 消息历史，贯穿整个交互会话
    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)  # s04: 用户输入前触发钩子
        history.append({"role": "user", "content": query})
        agent_loop(history)  # 进入含技能加载的工具调用循环
        # 打印模型最终文本回复（蓝色）
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(f"\033[34m{block.text}\033[0m")
        print()
