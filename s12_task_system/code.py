#!/usr/bin/env python3
"""
s12: Task System — 文件持久化任务图，带 blockedBy 依赖关系。

AI 编程智能体的任务管理：
1. Task 数据类（id/subject/description/status/owner/blockedBy）—— 结构化任务定义
2. .tasks/ 目录持久化 JSON 存储 —— 跨会话保留
3. can_start 依赖检查 —— 所有 blockedBy 任务完成才可启动
4. claim_task 认领 + complete_task 完成（自动报告下游解封）
5. 5 个新工具：create_task, list_tasks, get_task, claim_task, complete_task

Run / 运行: python s12_task_system/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

Changes from s11 / 相对 s11 的变更:
  - Task dataclass (id, subject, description, status, owner, blockedBy)
  - TASKS_DIR = .tasks/ for persistent JSON storage
  - CRUD: create_task / save_task / load_task / list_tasks / get_task
  - can_start: checks blockedBy all completed (missing deps = blocked)
  - claim_task: set owner + pending -> in_progress
  - complete_task: set completed + report unblocked downstream
  - 5 new tools: create_task, list_tasks, get_task, claim_task, complete_task

Note: Teaching code keeps a basic agent loop to stay focused on the task
system. S11's full error recovery (RecoveryState, backoff, escalation,
reactive compact, fallback model) is omitted — in real CC, tasks.ts and
withRetry are independent layers that compose naturally.
注意：教学代码保持一个基本的代理循环，以专注于任务系统。S11的完整错误恢复（RecoveryState、
回退、升级、反应式压缩、回退模型）被省略了——在真正的CC中，tasks.ts和withRetry是自然组合的独立层。
"""

import os, subprocess, json, time, random
from pathlib import Path
from dataclasses import dataclass, asdict

# ── 终端中文输入兼容性 ──────────────────────────────────
try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
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
MODEL = os.environ["MODEL_ID"]

# ═══════════════════════════════════════════════════════════
#  NEW in s12: 任务系统（Task System）
#  文件持久化任务图，带 blockedBy 依赖关系 + 5 个工具
# ═══════════════════════════════════════════════════════════

TASKS_DIR = WORKDIR / ".tasks"  # 任务 JSON 文件存储目录
TASKS_DIR.mkdir(exist_ok=True)


@dataclass # 核心作用是自动为类生成常见的特殊方法（如 __init__、__repr__、__eq__ 等），从而大幅减少编写样板代码的工作量，让类更专注于数据的存储。
class Task:
    """任务数据类：结构化任务定义。
    - status: pending → in_progress → completed（三态流转）
    - owner: 认领者名称（多 Agent 场景下的标识）
    - blockedBy: 依赖任务 ID 列表（全部完成才可启动）"""
    id: str
    subject: str
    description: str
    status: str          # pending | in_progress | completed
    owner: str | None    # 认领者名称（多 Agent 场景）
    blockedBy: list[str] # 依赖任务 ID 列表


def _task_path(task_id: str) -> Path:
    """获取任务 JSON 文件的路径。"""
    return TASKS_DIR / f"{task_id}.json"


def create_task(subject: str, description: str = "",
                blockedBy: list[str] | None = None) -> Task:
    """创建新任务并持久化到 .tasks/ 目录。ID 使用时间戳+随机数保证唯一。"""
    task = Task(
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
        subject=subject,
        description=description,
        status="pending",
        owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task


def save_task(task: Task):
    """将任务序列化为 JSON 写入磁盘。"""
    _task_path(task.id).write_text(json.dumps(asdict(task), indent=2))


def load_task(task_id: str) -> Task:
    """从磁盘加载单个任务 JSON 并反序列化为 Task 对象。"""
    return Task(**json.loads(_task_path(task_id).read_text()))


def list_tasks() -> list[Task]:
    """按文件名排序列出所有任务。"""
    return [Task(**json.loads(p.read_text()))
            for p in sorted(TASKS_DIR.glob("task_*.json"))]


def get_task(task_id: str) -> str:
    """获取单个任务的完整 JSON 详情。"""
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)


def can_start(task_id: str) -> bool:
    """依赖检查：所有 blockedBy 任务都已完成（含文件存在性检查）。"""
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists():
            return False  # 依赖不存在 → 阻塞
        if load_task(dep_id).status != "completed":
            return False  # 依赖未完成 → 阻塞
    return True


def claim_task(task_id: str, owner: str = "agent") -> str:
    """认领任务：验证 pending 状态 → 检查依赖 → 设 owner + 改为 in_progress。"""
    task = load_task(task_id)
    if task.status != "pending": # 如果任务已被别人认领或完成，就不能认领了
        return f"Task {task_id} is {task.status}, cannot claim"
    if not can_start(task_id): # 如果依赖未满足，也不能认领
        deps = [d for d in task.blockedBy
                if not _task_path(d).exists() or load_task(d).status != "completed"]
        return f"Blocked by: {deps}"
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress (owner: {owner})\033[0m") # 青色标签显示认领的任务和所有者
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str) -> str:
    """完成任务：验证 in_progress 状态 → 改为 completed → 检查并报告下游解封任务。
    这是依赖图的"传播"节点——完成一个任务可能解锁多个下游。"""
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    task.status = "completed"
    save_task(task)
    # 检查哪些 pending 任务的依赖已全部满足
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy and can_start(t.id)]
    print(f"  \033[32m[complete] {task.subject} ✓\033[0m") # 绿色标签显示完成的任务
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  \033[33m[unblocked] {', '.join(unblocked)}\033[0m") # 黄色标签显示被解封的任务
    return msg


# ═══════════════════════════════════════════════════════════
#  FROM s10: Prompt 组装（同步，tools 段包含任务工具）
# ═══════════════════════════════════════════════════════════

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file, "
             "create_task, list_tasks, get_task, claim_task, complete_task.",
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
        return _last_prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    return _last_prompt


# ── 基础工具实现 ─────────────────────────────────────
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
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════
#  s12: 任务工具（5 个工具函数）任务系统特有的
#  [create](蓝)/[claim](青)/[complete](绿)/[unblocked](黄) 
# ═══════════════════════════════════════════════════════════

def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    """创建新任务，可指定依赖关系。"""
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m") # 蓝色标签显示创建的任务和依赖
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    """列出所有任务，带状态图标和依赖信息。"""
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for t in tasks:
        icon = {"pending": "○", "in_progress": "●",
                "completed": "✓"}.get(t.status, "?")
        deps = f" (blockedBy: {', '.join(t.blockedBy)})" if t.blockedBy else ""
        owner = f" [{t.owner}]" if t.owner else ""
        lines.append(f"  {icon} {t.id}: {t.subject} "
                     f"[{t.status}]{owner}{deps}")
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    """获取单个任务的完整 JSON 详情。"""
    try:
        return get_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"


def run_claim_task(task_id: str) -> str:
    """认领任务：验证依赖 → 设置 owner → 状态改为 in_progress。"""
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    """完成任务：状态改为 completed → 报告下游解封任务。"""
    return complete_task(task_id)


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
    {"name": "create_task",
     "description": "Create a new task with optional blockedBy dependencies.",
     "input_schema": {"type": "object",
                      "properties": {
                          "subject": {"type": "string"},
                          "description": {"type": "string"},
                          "blockedBy": {"type": "array",
                                        "items": {"type": "string"}}},
                      "required": ["subject"]}},
    {"name": "list_tasks",
     "description": "List all tasks with status, owner, and dependencies.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
    {"name": "get_task",
     "description": "Get full details of a specific task by ID.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "claim_task",
     "description": "Claim a pending task. Sets owner, changes status to in_progress.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "complete_task",
     "description": "Complete an in-progress task. Reports unblocked downstream tasks.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
]

# ── 工具分发映射（s12 新增 5 个任务工具）─────────────
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "create_task": run_create_task, "list_tasks": run_list_tasks,
    "get_task": run_get_task, "claim_task": run_claim_task,
    "complete_task": run_complete_task,
}


# ═══════════════════════════════════════════════════════════
#  FROM s10: Context 系统
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
    if block.name in ("create_task",):
        return inputs.get("subject", "")
    if block.name in ("get_task", "claim_task", "complete_task"):
        return inputs.get("task_id", "")
    if block.name == "list_tasks":
        return ""
    return inputs.get("path", "")


# ═══════════════════════════════════════════════════════════
#  agent_loop — 精简版（聚焦任务系统，省略 s11 错误恢复）
# ═══════════════════════════════════════════════════════════

def agent_loop(messages: list, context: dict):
    """智能体主循环 + 任务系统。
    流程：从 context 组装 prompt → LLM 调用 → 工具执行（含 5 个任务工具）
    → 每轮后刷新 context + prompt。"""
    system = get_system_prompt(context)
    while True:
        try:
            response = client.messages.create(
                model=MODEL, system=system, messages=messages,
                tools=TOOLS, max_tokens=8000)
        except Exception as e:
            messages.append({"role": "assistant", "content": [
                {"type": "text",
                 "text": f"[Error] {type(e).__name__}: {e}"}]})
            return

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
            print(f"\033[1;35m[{block.name} -> Tool Calling Result]\033[0m \033[35m{str(output)[:300]}\033[0m")
            results.append({"type": "tool_result",
                            "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})
        context = update_context(context, messages)
        system = get_system_prompt(context)


# ── 入口：交互式 REPL ──────────────────────────────────
if __name__ == "__main__":
    print("s12: task system")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []  # 消息历史
    context = update_context({}, [])  # s10: 启动时初始化 context
    while True:
        try:
            query = input("\033[36ms12 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history, context)
        context = update_context(context, history)  # 刷新 context
        # 打印模型最终文本回复（蓝色）
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(f"\033[34m{block.text}\033[0m")
        print()
