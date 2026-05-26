#!/usr/bin/env python3
"""
s03_permission.py - 权限系统（Permission System）

在工具执行前插入三道安全门：

    Gate 1: 硬拒绝列表 —— 匹配即拦截（rm -rf /、sudo 等永不执行）
    Gate 2: 规则匹配 —— 上下文判断（写入工作目录外？内容含破坏性命令？）
    Gate 3: 用户审批 —— 暂停等待人工确认

    +-------+    +--------+    +--------+    +--------+    +------+
    | Tool  | -> | Gate 1 | -> | Gate 2 | -> | Gate 3 | -> | Exec |
    | call  |    | deny?  |    | match? |    | allow? |    |      |
    +-------+    +--------+    +--------+    +--------+    +------+
         |            |             |             |
         v            v             v             v
      (正常放行)   (直接拦截)    (询问用户)   (用户拒绝?)

agent_loop 只加了一行：

    if not check_permission(block):
        continue

    Gate 1: Hard deny list (rm -rf /, sudo, ...)        — 直接拦截，不询问
    Gate 2: Rule matching (write outside workspace? destructive cmd?)  — 命中则触发 Gate 3
    Gate 3: User approval (pause and wait for confirmation)  — 人工确认

Builds on s02 (multi-tool). Usage / 用法:

    python s03_permission/code.py
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

# 系统提示词：告诉模型破坏性操作需要用户审批
SYSTEM = f"You are a coding agent at {WORKDIR}. All destructive operations require user approval."


# ═══════════════════════════════════════════════════════════
#  FROM s02 (unchanged): 工具实现 — 来自 s02（未改动）
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
    注意：危险命令黑名单已上移到 Gate 1（check_deny_list），
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


# ═══════════════════════════════════════════════════════════
#  FROM s02 (unchanged): 工具定义 & 分发映射 — 来自 s02（未改动）
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
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}


# ═══════════════════════════════════════════════════════════
#  NEW in s03: 三道门权限管道（Three-Gate Permission Pipeline）
# ═══════════════════════════════════════════════════════════

# Gate 1: 硬拒绝列表 —— 匹配即拦截，永不执行
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]

def check_deny_list(command: str) -> str | None:
    """Gate 1：检查命令是否命中硬拒绝列表。命中返回拦截原因，否则 None。"""
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


# Gate 2: 规则匹配 —— 上下文判断，命中后触发 Gate 3 询问用户
PERMISSION_RULES = [
    {"tools": ["write_file", "edit_file"],
     "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
     "message": "Writing outside workspace"}, # 禁止写出工作目录
    {"tools": ["bash"],
     "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
     "message": "Potentially destructive command"}, # 简单的破坏性命令检测
]

def check_rules(tool_name: str, args: dict) -> str | None:
    """Gate 2：遍历规则列表，检查当前工具调用是否命中。命中返回原因，否则 None。"""
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


# Gate 3: 用户审批 —— 暂停等待人工输入 y/n
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    """Gate 3：打印警告信息并等待用户确认。返回 'allow' 或 'deny'。"""
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


# Pipeline: 三道门串联
def check_permission(block) -> bool:
    """权限检查管道。先 Gate 1（硬拒绝）→ Gate 2 命中则 Gate 3（询问用户）。
    返回 True 表示允许执行，False 表示拦截。"""
    # Gate 1: 仅对 bash 检查硬拒绝列表
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n\033[31m⛔ {reason}\033[0m")
            return False
    # Gate 2 → Gate 3: 规则命中则询问用户
    reason = check_rules(block.name, block.input)
    if reason:
        decision = ask_user(block.name, block.input, reason)
        if decision == "deny":
            return False
    return True


def _tool_input_summary(block) -> str:
    """提取工具调用的关键参数摘要，用于终端显示。"""
    inputs = block.input
    if block.name == "bash":
        return inputs.get("command", "")
    if block.name == "glob":
        return inputs.get("pattern", "")
    return inputs.get("path", "")


# ═══════════════════════════════════════════════════════════
#  agent_loop — 与 s02 结构一致，插入 check_permission() 管道
#  s02: handler = TOOL_HANDLERS[...]; output = handler(...)
#  s03: if not check_permission(block): continue;  然后执行
# ═══════════════════════════════════════════════════════════

def agent_loop(messages: list):
    """智能体主循环 + 权限检查。
    流程：
    1. 将完整消息历史发送给 LLM
    2. 若 LLM 返回 tool_use → 经 permission pipeline → 执行/拦截 → 追加结果
    3. 若 LLM 返回 end_turn → 循环结束，由调用方打印最终回复
    """
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # 将助手回复追加到消息历史
        messages.append({"role": "assistant", "content": response.content})

        # 若模型未调用工具（直接回复文本），循环结束
        if response.stop_reason != "tool_use":
            return

        # 遍历响应中的每个工具调用块，经权限管道后执行
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # 显示工具名和关键参数（黄色）
            print(f"\033[33m$ {block.name}: {_tool_input_summary(block)}\033[0m")

            # s03 新增：执行前通过三道门权限管道
            if not check_permission(block):
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "Permission denied."})
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            # 工具结果：粗体品红标签 + 品红内容
            print(f"\033[1;35m[{block.name} -> Tool Calling Result]\033[0m \033[35m{str(output)[:200]}\033[0m")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        # 将工具结果作为用户消息追加，触发下一轮循环
        messages.append({"role": "user", "content": results})


# ── 入口：交互式 REPL ──────────────────────────────────
# 流程：读取用户输入 → agent_loop（内含权限检查）→ 打印模型最终回复 → 循环
if __name__ == "__main__":
    print("s03: Permission")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []  # 消息历史，贯穿整个交互会话
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)  # 进入含权限管道的工具调用循环
        # 打印模型最终文本回复（蓝色）
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(f"\033[34m{block.text}\033[0m")
        print()
