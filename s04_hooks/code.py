#!/usr/bin/env python3
"""
s04: Hooks — 将扩展逻辑从循环中移出，挂到钩子上。

AI 编程智能体的可扩展架构：
1. HOOKS 注册表 —— 按事件类型（UserPromptSubmit / PreToolUse / PostToolUse / Stop）组织回调
2. register_hook() / trigger_hooks() —— 注册和触发钩子的统一接口
3. 权限逻辑从循环中移除，封装为 permission_hook，挂到 PreToolUse 事件上
4. 循环只保留 trigger_hooks() 调用点，扩展新功能只需注册新钩子

  User types query / 用户输入
       │
       ▼
  ┌──────────────────┐
  │ UserPromptSubmit │ ── trigger_hooks() before LLM
  └────────┬─────────┘
           ▼
  ┌────────────┐     ┌─────────────────────────────┐
  │  messages  │────▶│  LLM (stop_reason=tool_use?)│
  └────────────┘     │   No ──▶ Stop hooks ──▶ exit │
                     │   Yes ──▶ tool_use block ──┐ │
                     └────────────────────────────┘ │
                                                    ▼
                                          ┌──────────────────┐
                                          │ trigger_hooks()   │
                                          │  PreToolUse:      │
                                          │   permission_hook │
                                          │   log_hook        │
                                          └───────┬──────────┘
                                                  │ (not blocked)
                                          ┌───────▼──────────┐
                                          │ TOOL_HANDLERS[x]  │
                                          └───────┬──────────┘
                                                  │
                                          ┌───────▼──────────┐
                                          │ trigger_hooks()   │
                                          │  PostToolUse:     │
                                          │   large_output    │
                                          └───────┬──────────┘
                                                  │
                                          results ──▶ back to messages / 回到 messages

Changes from s03 / 相对 s03 的变更:
  + HOOKS registry (event -> list of callbacks)
  + register_hook() / trigger_hooks()
  + context_inject_hook (UserPromptSubmit)
  + permission_hook, log_hook (PreToolUse)
  + large_output_hook (PostToolUse)
  + summary_hook (Stop)
  - check_permission() removed from loop body
    (logic moved into permission_hook, triggered via PreToolUse)

Run / 运行: python s04_hooks/code.py
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

# 系统提示词：告诉模型它是一个编码智能体，使用工具解决问题
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


# ═══════════════════════════════════════════════════════════
#  FROM s02-s03 (unchanged): 工具实现 — 来自 s02/s03（未改动）
# ═══════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    """路径安全校验。将用户输入的相对路径解析为绝对路径，
    若路径逃逸出工作目录则抛出异常，防止模型读写系统文件。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    """执行 shell 命令并返回输出。
    注意：权限检查已上移到 PreToolUse 钩子（permission_hook），
    这里只做超时和输出截断保护。"""
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
]

# ── 工具分发映射 — 来自 s02（未改动）─────────────────
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}


# ═══════════════════════════════════════════════════════════
#  NEW in s04: 钩子系统（Hook System）
#  s03 的权限逻辑封装为 permission_hook，通过 PreToolUse 事件触发
#  ═══════════════════════════════════════════════════════════

# 钩子注册表：每种事件对应一个回调列表
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


# ── s03 的权限检查逻辑，现在封装为钩子 ──────────────

DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]

def permission_hook(block):
    """PreToolUse 钩子：权限检查。将 s03 的三道门逻辑合并为一个钩子。
    返回非 None 字符串表示拦截，None 表示放行。"""
    if block.name == "bash":
        # Gate 1: 硬拒绝列表
        for pattern in DENY_LIST:
            if pattern in block.input.get("command", ""):
                print(f"\n\033[31m⛔ Blocked: '{pattern}'\033[0m")
                return "Permission denied by deny list"
        # Gate 2+3: 破坏性命令 → 询问用户
        for kw in DESTRUCTIVE:
            if kw in block.input.get("command", ""):
                print(f"\n\033[33m⚠  Potentially destructive command\033[0m")
                print(f"   Tool: {block.name}({block.input})")
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
    if block.name in ("write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m⚠  Writing outside workspace\033[0m")
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None

def log_hook(block):
    """PreToolUse 钩子：记录每次工具调用（灰色日志）。"""
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({args_preview})\033[0m")
    return None

def large_output_hook(block, output):
    """PostToolUse 钩子：输出超过 100k 字符时发出警告。"""
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] ⚠ Large output from {block.name}: {len(str(output))} chars\033[0m")
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

# ── 注册钩子：将回调绑定到对应事件 ─────────────────
register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def _tool_input_summary(block) -> str:
    """提取工具调用的关键参数摘要，用于终端显示。"""
    inputs = block.input
    if block.name == "bash":
        return inputs.get("command", "")
    if block.name == "glob":
        return inputs.get("pattern", "")
    return inputs.get("path", "")


# ═══════════════════════════════════════════════════════════
#  agent_loop — 与 s03 结构一致，硬编码的 check_permission 替换为钩子触发
#  s03: if not check_permission(block): ...
#  s04: if trigger_hooks("PreToolUse", block): ...
# ═══════════════════════════════════════════════════════════

def agent_loop(messages: list):
    """智能体主循环 + 钩子系统。
    流程：
    1. 将完整消息历史发送给 LLM
    2. 若 stop_reason == end_turn → 触发 Stop 钩子 → 退出
    3. 若 tool_use → 触发 PreToolUse 钩子（权限/日志）→ 执行工具
       → 触发 PostToolUse 钩子 → 将结果追加到 messages → 循环
    """
    while True:
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

        # 遍历响应中的每个工具调用块
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # 显示工具名和关键参数（黄色）
            print(f"\033[33m$ {block.name}: {_tool_input_summary(block)}\033[0m")

            # s04 变更：PreToolUse 钩子替代硬编码的 check_permission()
            blocked = trigger_hooks("PreToolUse", block) # 包含权限检查和日志记录钩子
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            # 工具结果：粗体品红标签 + 品红内容
            print(f"\033[1;35m[{block.name} -> Tool Calling Result]\033[0m \033[35m{str(output)[:200]}\033[0m")

            # s04 新增：PostToolUse 钩子（大输出告警等）
            trigger_hooks("PostToolUse", block, output)

            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        # 将工具结果作为用户消息追加，触发下一轮循环
        messages.append({"role": "user", "content": results})


# ── 入口：交互式 REPL ──────────────────────────────────
# 流程：触发 UserPromptSubmit 钩子 → 读取输入 → agent_loop → 打印最终回复 → 循环
if __name__ == "__main__":
    print("s04: Hooks — extension logic on hooks, loop stays clean")
    print("Type a question, press Enter. Type q to quit.\n")

    history = []  # 消息历史，贯穿整个交互会话
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)  # s04: 用户输入前触发钩子
        history.append({"role": "user", "content": query})
        agent_loop(history)  # 进入含钩子系统的工具调用循环
        # 打印模型最终文本回复（蓝色）
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(f"\033[34m{block.text}\033[0m")
        print()
