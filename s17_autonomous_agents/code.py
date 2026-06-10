#!/usr/bin/env python3
"""
s17: Autonomous Agents — idle 轮询 + 自动认领 + WORK/IDLE 生命周期。

AI 编程智能体的自治 Agent：
1. scan_unclaimed_tasks: 扫描任务板，发现依赖已满足的未认领任务
2. idle_poll: 60 秒轮询循环（收件箱 + 任务板），在 IDLE 状态也能处理 shutdown
3. claim_task: 新增 owner 检查（已被认领则拒绝）+ 返回值验证
4. Teammate 三阶段生命周期: WORK（干活）→ IDLE（等活）→ SHUTDOWN（关机）
5. Teammate 工具从 5 个扩展到 8 个: + list_tasks, claim_task, complete_task
6. Identity re-injection: 上下文压缩后重新注入身份提示

核心洞察：s15-s16 中，Lead 必须给每个队友分配任务。"Alice 做这个，Bob 做那个"。
任务板上有 10 个未认领的任务，Lead 得手动 assign。s17 让队友自己看板、
自己认领——Lead 只需要创建任务，队友自己发现、自己认领、自己完成。
这是从"领导驱动"到"自组织"的跃迁。

三阶段生命周期：
  WORK:  inbox → LLM → tools → (tool_use? → 循环) → (done? → IDLE)
  IDLE:  每 5s 轮询 → inbox有消息? → WORK / 任务板有未认领? → claim → WORK / 60s超时? → SHUTDOWN

Run / 运行: python s17_autonomous_agents/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

Changes from s16 / 相对 s16 的变更:
  - scan_unclaimed_tasks: find pending, unowned tasks with deps completed
  - idle_poll: 60s polling loop (inbox + task board), dispatches shutdown in IDLE
  - claim_task: owner check + return value verification
  - Teammate lifecycle: WORK → IDLE → SHUTDOWN
  - Teammate tools: + list_tasks, claim_task, complete_task (5→8)
  - consume_lead_inbox: unified inbox consumer for protocol + context injection
  - Identity re-injection after context compression
"""

import os, subprocess, json, time, random, threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field

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
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# ═══════════════════════════════════════════════════════════
#  FROM s12: 任务系统（Task System）
#  s17 重要变更：claim_task 新增 owner 检查——防止多人同时认领
# ═══════════════════════════════════════════════════════════

TASKS_DIR = WORKDIR / ".tasks"  # 任务 JSON 文件存储目录
TASKS_DIR.mkdir(exist_ok=True)


@dataclass  # 核心作用是自动为类生成常见的特殊方法（如 __init__、__repr__、__eq__ 等），从而大幅减少编写样板代码的工作量，让类更专注于数据的存储。
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
        subject=subject, description=description,
        status="pending", owner=None,
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
    """认领任务：s17 新增 owner 检查——已被认领则拒绝，防止多 Agent 同时认领。
    验证顺序：pending 状态 → owner 为空 → 依赖满足 → 设 owner + in_progress。"""
    task = load_task(task_id)
    if task.status != "pending":  # 状态不对——可能已完成或进行中
        return f"Task {task_id} is {task.status}, cannot claim"
    if task.owner:  # s17 新增：已被认领就拒绝（防止重复认领）
        return f"Task {task_id} already owned by {task.owner}"
    if not can_start(task_id):  # 依赖未满足
        deps = [d for d in task.blockedBy
                if _task_path(d).exists() and load_task(d).status != "completed"] # 依赖存在但未完成的情况
        missing = [d for d in task.blockedBy if not _task_path(d).exists()] # 依赖文件不存在的情况
        parts = []
        if deps: parts.append(f"blocked by: {deps}")
        if missing: parts.append(f"missing deps: {missing}")
        return "Cannot start — " + ", ".join(parts)
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress\033[0m")  # 青色标签显示认领的任务
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str) -> str:
    """完成任务：验证 in_progress 状态 → 改为 completed → 报告下游解封任务。"""
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    task.status = "completed"
    save_task(task)
    # 检查哪些 pending 任务的依赖已全部满足
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy and can_start(t.id)]
    print(f"  \033[32m[complete] {task.subject} ✓\033[0m")  # 绿色标签显示完成的任务
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
    return msg


# ═══════════════════════════════════════════════════════════
#  FROM s10: Prompt 组装（同步）
# ═══════════════════════════════════════════════════════════

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file, "
             "create_task, list_tasks, get_task, claim_task, complete_task, "
             "spawn_teammate, send_message, check_inbox, "
             "request_shutdown, request_plan, review_plan.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    """从 PROMPT_SECTIONS 和 context 组装 system prompt。"""
    sections = [PROMPT_SECTIONS["identity"],
                PROMPT_SECTIONS["tools"],
                PROMPT_SECTIONS["workspace"]]
    if context.get("memories"):
        sections.append(f"Relevant memories:\n{context['memories']}")
    return "\n\n".join(sections)


_last_context_hash, _last_prompt = None, None


def get_system_prompt(context: dict) -> str:
    """获取 system prompt（带确定性缓存，避免重复组装）。"""
    global _last_context_hash, _last_prompt
    h = json.dumps(context, sort_keys=True)
    if h == _last_context_hash and _last_prompt:
        return _last_prompt
    _last_context_hash, _last_prompt = h, assemble_system_prompt(context)
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
#  FROM s15/s16: MessageBus（消息总线）+ ProtocolState（协议状态机）
# ═══════════════════════════════════════════════════════════

MAILBOX_DIR = WORKDIR / ".mailboxes"  # 邮箱文件存储目录
MAILBOX_DIR.mkdir(exist_ok=True)


class MessageBus:
    """基于文件的消息总线。s17 保留完整协议支持。"""

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message", metadata: dict = None):
        """发送消息：追加 JSON 行到目标 agent 的 .jsonl 收件箱。"""
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time(), "metadata": metadata or {}}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a") as f:
            f.write(json.dumps(msg) + "\n")
        print(f"  \033[33m[bus] {from_agent} → {to_agent}: "
              f"({msg_type}) {content[:50]}\033[0m") # 黄色标签显示消息发送日志，包含发送者、接收者、消息类型和内容摘要

    def read_inbox(self, agent: str) -> list[dict]:
        """读取收件箱全部消息并删除文件（消费语义）。"""
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        msgs = [json.loads(line) for line in inbox.read_text().splitlines()
                if line.strip()]
        inbox.unlink()  # 消费语义：读后即删
        return msgs


BUS = MessageBus()  # 全局消息总线实例
active_teammates: dict[str, bool] = {}  # 队友注册表


@dataclass
class ProtocolState:
    """协议请求状态数据类。s17 保留完整协议支持。"""
    request_id: str
    type: str       # "shutdown" | "plan_approval"
    sender: str     # 请求发起方
    target: str     # 请求接收方
    status: str     # pending | approved | rejected
    payload: str    # 计划文本或关机原因
    created_at: float = field(default_factory=time.time)

# 全局协议状态表：request_id → ProtocolState。s17 保留完整协议支持。
pending_requests: dict[str, ProtocolState] = {}  # request_id → ProtocolState


def new_request_id() -> str:
    """生成唯一的协议请求 ID。"""
    return f"req_{random.randint(0, 999999):06d}"


def match_response(response_type: str, request_id: str, approve: bool):
    """协议响应匹配：三层校验（ID存在→类型匹配→状态检查）。"""
    state = pending_requests.get(request_id)
    if not state:
        print(f"  \033[31m[protocol] unknown request_id: {request_id}\033[0m")
        return
    if state.type == "shutdown" and response_type != "shutdown_response":
        print(f"  \033[31m[protocol] type mismatch: expected shutdown_response, "
              f"got {response_type}\033[0m")
        return
    if state.type == "plan_approval" and response_type != "plan_approval_response":
        print(f"  \033[31m[protocol] type mismatch: expected plan_approval_response, "
              f"got {response_type}\033[0m")
        return
    state.status = "approved" if approve else "rejected"
    icon = "✓" if approve else "✗"
    color = "32" if approve else "31"
    print(f"  \033[{color}m[protocol] {state.type} {icon} " # 绿色✓表示批准，红色✗表示拒绝
          f"({request_id}: {state.status})\033[0m")


# ═══════════════════════════════════════════════════════════
#  NEW in s17: 自治 Agent（Autonomous Agent）
#  scan_unclaimed_tasks: 扫描任务板发现可认领的任务
#  idle_poll: 60s 轮询循环，在 IDLE 状态自驱干活
# ═══════════════════════════════════════════════════════════

IDLE_POLL_INTERVAL = 5   # 空闲轮询间隔（秒）
IDLE_TIMEOUT = 60         # 空闲超时（秒）——超时后退出


def scan_unclaimed_tasks() -> list[dict]:
    """扫描任务板：找到 pending + 无 owner + 依赖全部满足的任务。
    返回的任务列表按文件名排序（即按创建时间排序）。"""
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"          # 待认领
                and not task.get("owner")             # 未被认领
                and can_start(task["id"])):           # 依赖全部满足
            unclaimed.append(task)
    return unclaimed


def idle_poll(agent_name: str, messages: list,
              name: str, role: str) -> str:
    """IDLE 阶段的核心轮询函数。60 秒内每 5 秒轮询，返回状态：
    - 'work': 收到消息或自动认领了任务 → 恢复工作
    - 'shutdown': 收到 shutdown_request → 体面退出
    - 'timeout': 60s 无事件 → 退出"""
    for _ in range(IDLE_TIMEOUT // IDLE_POLL_INTERVAL):  # 60/5 = 12 轮
        time.sleep(IDLE_POLL_INTERVAL) # 等待 5 秒钟

        # 优先级 1: 检查收件箱（协议消息优先）
        inbox = BUS.read_inbox(agent_name)
        if inbox:
            # 检查是否有 shutdown_request（协议消息优先处理）
            for msg in inbox:
                if msg.get("type") == "shutdown_request":
                    req_id = msg.get("metadata", {}).get("request_id", "")
                    BUS.send(name, "lead", "Shutting down gracefully.",
                             "shutdown_response",
                             {"request_id": req_id, "approve": True})
                    print(f"  \033[35m[protocol] {name} approved shutdown " # 紫色标签显示在 IDLE 轮询中批准了关机请求
                          f"in idle ({req_id})\033[0m")
                    return "shutdown"

            # 非协议收件箱消息 → 注入并恢复工作
            messages.append({"role": "user",
                "content": "<inbox>" + json.dumps(inbox) + "</inbox>"})
            print(f"  \033[36m[idle] {name} found inbox messages\033[0m") # 青色标签显示在 IDLE 轮询中发现了收件箱消息
            return "work"

        # 优先级 2: 扫描任务板自动认领
        unclaimed = scan_unclaimed_tasks()
        if unclaimed:
            task = unclaimed[0]  # 按创建时间取第一个
            result = claim_task(task["id"], agent_name) # 认领者为队友自己的名字
            if "Claimed" in result: # 认领成功 → 注入消息并恢复工作
                messages.append({"role": "user",
                    "content": f"<auto-claimed>Task {task['id']}: "
                               f"{task['subject']}</auto-claimed>"})
                print(f"  \033[32m[idle] {name} auto-claimed: " # 绿色标签显示在 IDLE 轮询中自动认领了任务
                      f"{task['subject']}\033[0m")
                return "work"
            print(f"  \033[33m[idle] {name} claim failed: "
                  f"{result}\033[0m")

    print(f"  \033[31m[idle] {name} timeout ({IDLE_TIMEOUT}s)\033[0m") # 红色标签显示在 IDLE 轮询中超时未发现事件
    return "timeout"


# ═══════════════════════════════════════════════════════════
#  NEW in s17: Teammate Thread（自治队友线程）
#  WORK → IDLE → SHUTDOWN 三阶段生命周期
#  队友工具从 5 扩展到 8：+ list_tasks/claim_task/complete_task
#  新增 identity re-injection：上下文压缩后重新注入身份提示
# ═══════════════════════════════════════════════════════════

def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    """生成自治队友 Agent 线程。三阶段生命：WORK→IDLE→SHUTDOWN。
    队友可以自己扫描任务板、自己认领任务——不需要 Lead 手动分配。"""
    if name in active_teammates:
        return f"Teammate '{name}' already exists"

    system = (f"You are '{name}', a {role}. "
              f"Use tools to complete tasks. "
              f"You can list and claim tasks from the board. "
              f"Check inbox for protocol messages.") # 系统提示明确告知队友要检查收件箱以响应协议消息，并且可以自己查看任务板和认领任务

    def handle_inbox_message(name: str, msg: dict, messages: list):
        """协议消息分发器：按消息类型路由。返回 True 表示应关机。"""
        msg_type = msg.get("type", "message")
        meta = msg.get("metadata", {})
        req_id = meta.get("request_id", "")

        if msg_type == "shutdown_request":
            BUS.send(name, "lead", "Shutting down gracefully.",
                     "shutdown_response",
                     {"request_id": req_id, "approve": True})
            print(f"  \033[35m[protocol] {name} approved shutdown "
                  f"({req_id})\033[0m")
            return True  # 停止循环

        if msg_type == "plan_approval_response":
            approve = meta.get("approve", False)
            if approve:
                messages.append({"role": "user",
                    "content": "[Plan approved] Proceed with the task."})
            else:
                messages.append({"role": "user",
                    "content": f"[Plan rejected] Feedback: {msg['content']}"})
        return False  # 继续循环

    def run():
        """队友的主循环：外层 WORK/IDLE 交替，内层 WORK 最多 10 轮。"""
        messages = [{"role": "user", "content": prompt}]
        # s17: 队友工具有 8 个（s16 的 5 个 + list_tasks/claim_task/complete_task）
        sub_tools = [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object",
                              "properties": {"command": {"type": "string"}},
                              "required": ["command"]}},
            {"name": "read_file", "description": "Read file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"}},
                              "required": ["path"]}},
            {"name": "write_file", "description": "Write file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["path", "content"]}},
            {"name": "send_message",
             "description": "Send message to another agent.",
             "input_schema": {"type": "object",
                              "properties": {"to": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["to", "content"]}},
            {"name": "submit_plan",
             "description": "Submit a plan for Lead approval.",
             "input_schema": {"type": "object",
                              "properties": {"plan": {"type": "string"}},
                              "required": ["plan"]}},
            # s17 新增：队友可以列出、认领、完成任务
            {"name": "list_tasks",
             "description": "List all tasks on the board.",
             "input_schema": {"type": "object", "properties": {},
                              "required": []}},
            {"name": "claim_task",
             "description": "Claim a pending task.",
             "input_schema": {"type": "object",
                              "properties": {"task_id": {"type": "string"}},
                              "required": ["task_id"]}},
            {"name": "complete_task",
             "description": "Mark an in-progress task as completed.",
             "input_schema": {"type": "object",
                              "properties": {"task_id": {"type": "string"}},
                              "required": ["task_id"]}},
        ]

        def _run_list_tasks():
            """队友列出任务板。"""
            tasks = list_tasks()
            if not tasks:
                return "No tasks."
            return "\n".join(
                f"  {t.id}: {t.subject} [{t.status}]"
                for t in tasks)

        def _run_claim_task(task_id: str):
            """队友认领任务——认领者为队友自己的名字。"""
            return claim_task(task_id, owner=name)

        def _run_complete_task(task_id: str):
            """队友完成任务。"""
            return complete_task(task_id)

        sub_handlers = {
            "bash": run_bash, "read_file": run_read, "write_file": run_write,
            "send_message": lambda to, content: (BUS.send(name, to, content),
                                                  "Sent")[1],
            "submit_plan": lambda plan: _teammate_submit_plan(name, plan),
            "list_tasks": _run_list_tasks,
            "claim_task": _run_claim_task,
            "complete_task": _run_complete_task,
        }

        # 外层循环：WORK → IDLE 交替
        while True:
            # s17 新增：Identity re-injection —— 上下文压缩后重新注入身份
            if len(messages) <= 3:
                messages.insert(0, {"role": "user",
                    "content": f"<identity>You are '{name}', role: {role}. "
                               f"Continue your work.</identity>"})

            # WORK 阶段：最多 10 轮
            should_shutdown = False
            for _ in range(10):
                # 先检查收件箱中的协议消息
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    stopped = handle_inbox_message(name, msg, messages)
                    if stopped:
                        should_shutdown = True
                        break
                if should_shutdown:
                    break
                if inbox and not should_shutdown: # 如果有非协议消息，也注入给模型（但优先处理协议消息）
                    non_protocol = [m for m in inbox
                                    if m.get("type") == "message"]
                    if non_protocol:
                        messages.append({"role": "user",
                            "content": f"<inbox>{json.dumps(non_protocol)}</inbox>"})

                try:
                    response = client.messages.create(
                        model=MODEL, system=system, messages=messages[-20:],
                        tools=sub_tools, max_tokens=8000)
                except Exception:
                    break
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break  # LLM 主动停下 → 进入 IDLE
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        handler = sub_handlers.get(block.name)
                        output = handler(**block.input) if handler else "Unknown"
                        results.append({"type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": str(output)})
                messages.append({"role": "user", "content": results})

            if should_shutdown:
                break

            # IDLE 阶段（s17 核心新增）
            idle_result = idle_poll(name, messages, name, role)
            if idle_result == "shutdown":
                break # 收到 shutdown 请求，体面退出
            if idle_result == "timeout":
                break  # 超时退出

        # SHUTDOWN 阶段：发送最终摘要给 Lead
        summary = "Done."
        for msg in reversed(messages):
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for b in msg["content"]:
                    if getattr(b, "type", None) == "text":
                        summary = b.text
                        break
                else:
                    continue
                break
        BUS.send(name, "lead", summary, "result")
        active_teammates.pop(name, None)
        print(f"  \033[32m[teammate] {name} finished\033[0m") # 绿色标签显示队友完成了工作

    active_teammates[name] = True
    threading.Thread(target=run, daemon=True).start()
    print(f"  \033[36m[teammate] {name} spawned as {role}\033[0m") # 青色标签显示生成了新的队友线程
    return f"Teammate '{name}' spawned as {role} (autonomous)"


def _teammate_submit_plan(from_name: str, plan: str) -> str:
    """队友提交计划给 Lead 审批。协议级请求（非代码级门禁）。"""
    req_id = new_request_id()
    pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="plan_approval",
        sender=from_name, target="lead",
        status="pending", payload=plan)
    BUS.send(from_name, "lead", plan,
             "plan_approval_request",
             {"request_id": req_id})
    return f"Plan submitted ({req_id}). Waiting for approval..."


# ═══════════════════════════════════════════════════════════
#  FROM s16: Lead 协议工具（3 个）
# ═══════════════════════════════════════════════════════════

def run_request_shutdown(teammate: str) -> str:
    """Lead 发起关机协议：创建 ProtocolState → BUS.send shutdown_request。"""
    req_id = new_request_id()
    pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="shutdown",
        sender="lead", target=teammate,
        status="pending", payload="")
    BUS.send("lead", teammate, "Please shut down gracefully.",
             "shutdown_request",
             {"request_id": req_id})
    print(f"  \033[35m[protocol] shutdown_request → {teammate} "
          f"({req_id})\033[0m")
    return f"Shutdown request sent to {teammate} (req: {req_id})"


def run_request_plan(teammate: str, task: str) -> str:
    """Lead 要求队友提交工作计划。"""
    BUS.send("lead", teammate, f"Please submit a plan for: {task}",
             "message")
    return f"Asked {teammate} to submit a plan"


def run_review_plan(request_id: str, approve: bool,
                    feedback: str = "") -> str:
    """Lead 审批/拒绝队友提交的计划。"""
    state = pending_requests.get(request_id)
    if not state:
        return f"Request {request_id} not found"
    if state.status != "pending":
        return f"Request {request_id} already {state.status}"
    state.status = "approved" if approve else "rejected"
    BUS.send("lead", state.sender,
             feedback or ("Approved" if approve else "Rejected"),
             "plan_approval_response",
             {"request_id": request_id, "approve": approve})
    icon = "✓" if approve else "✗"
    print(f"  \033[32m[protocol] plan {icon} ({request_id})\033[0m")
    return f"Plan {'approved' if approve else 'rejected'} ({request_id})"


# ═══════════════════════════════════════════════════════════
#  基础工具 handlers（任务 + Team + 协议）
# ═══════════════════════════════════════════════════════════

def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    """创建新任务，可指定依赖关系。"""
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")  # 蓝色标签显示创建的任务和依赖
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    """列出所有任务，简化版（无图标）。"""
    tasks = list_tasks()
    if not tasks:
        return "No tasks."
    return "\n".join(
        f"  {t.id}: {t.subject} [{t.status}]"
        for t in tasks)


def run_get_task(task_id: str) -> str:
    """获取单个任务的完整 JSON。"""
    return get_task(task_id)


def run_claim_task(task_id: str) -> str:
    """认领任务（Lead 端，owner='agent'）。"""
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    """完成任务。"""
    return complete_task(task_id)


def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    """生成自治队友 Agent。"""
    return spawn_teammate_thread(name, role, prompt)


def run_send_message(to: str, content: str) -> str:
    """通过 MessageBus 发送消息。"""
    BUS.send("lead", to, content)
    return f"Sent to {to}"


def consume_lead_inbox(route_protocol=True) -> list[dict]:
    """统一读取 Lead 收件箱：先路由协议响应，再返回全部消息。"""
    msgs = BUS.read_inbox("lead")
    if route_protocol:
        for msg in msgs:
            meta = msg.get("metadata", {})
            req_id = meta.get("request_id", "")
            msg_type = msg.get("type", "")
            if req_id and msg_type.endswith("_response"):
                match_response(msg_type, req_id, meta.get("approve", False))
    return msgs


def run_check_inbox() -> str:
    """检查 Lead 收件箱，自动通过 match_response 路由协议响应。"""
    msgs = consume_lead_inbox(route_protocol=True)
    if not msgs:
        return "(inbox empty)"
    lines = []
    for m in msgs:
        meta = m.get("metadata", {})
        req_id = meta.get("request_id", "")
        tag = f" [{m['type']} req:{req_id}]" if req_id else f" [{m['type']}]" # 消息类型标签，包含请求 ID（如果有的话）
        lines.append(f"  [{m['from']}]{tag} {m['content'][:200]}")
    return "\n".join(lines)


# ── 工具定义（s17 共 14 个工具）─────────────────

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
     "description": "Create a task.",
     "input_schema": {"type": "object",
                      "properties": {"subject": {"type": "string"},
                                     "description": {"type": "string"},
                                     "blockedBy": {"type": "array",
                                                   "items": {"type": "string"}}},
                      "required": ["subject"]}},
    {"name": "list_tasks",
     "description": "List all tasks.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_task",
     "description": "Get full details of a specific task.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "claim_task",
     "description": "Claim a pending task.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "complete_task",
     "description": "Complete an in-progress task.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "spawn_teammate",
     "description": "Spawn an autonomous teammate agent.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"},
                                     "role": {"type": "string"},
                                     "prompt": {"type": "string"}},
                      "required": ["name", "role", "prompt"]}},
    {"name": "send_message",
     "description": "Send message to a teammate.",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["to", "content"]}},
    {"name": "check_inbox",
     "description": "Check inbox for messages and protocol responses.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "request_shutdown",
     "description": "Request a teammate to shut down gracefully.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"}},
                      "required": ["teammate"]}},
    {"name": "request_plan",
     "description": "Ask a teammate to submit a plan for review.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"},
                                     "task": {"type": "string"}},
                      "required": ["teammate", "task"]}},
    {"name": "review_plan",
     "description": "Approve or reject a submitted plan.",
     "input_schema": {"type": "object",
                      "properties": {
                          "request_id": {"type": "string"},
                          "approve": {"type": "boolean"},
                          "feedback": {"type": "string"}},
                      "required": ["request_id", "approve"]}},
]

# ── 工具分发映射 ─────────────────────────────────────
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "create_task": run_create_task, "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_claim_task, "complete_task": run_complete_task,
    "spawn_teammate": run_spawn_teammate,
    "send_message": run_send_message, "check_inbox": run_check_inbox,
    "request_shutdown": run_request_shutdown,
    "request_plan": run_request_plan, "review_plan": run_review_plan,
}


# ═══════════════════════════════════════════════════════════
#  FROM s10: Context 系统 — 从真实状态推导
# ═══════════════════════════════════════════════════════════

MEMORY_DIR = WORKDIR / ".memory"  # s09: 记忆目录
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"  # 记忆索引


def update_context(context: dict, messages: list) -> dict:
    """从真实文件系统状态推导 context。"""
    memories = ""
    if MEMORY_INDEX.exists():
        memories = MEMORY_INDEX.read_text()[:2000]  # 截断到 2000 字符
    return {"memories": memories}


# ═══════════════════════════════════════════════════════════
#  agent_loop — s17 核心：精简版循环（无后台任务/无 cron）
#  聚焦自治 Agent 功能，S11 的完整错误恢复被省略
# ═══════════════════════════════════════════════════════════

def _tool_input_summary(block) -> str:
    """提取工具调用的关键参数摘要，用于终端显示。"""
    inputs = block.input
    if block.name == "bash":
        return inputs.get("command", "")
    if block.name in ("create_task",):
        return inputs.get("subject", "")
    if block.name in ("get_task", "claim_task", "complete_task"):
        return inputs.get("task_id", "")
    if block.name in ("list_tasks", "list_crons", "check_inbox"):
        return ""
    if block.name == "spawn_teammate":
        return f"{inputs.get('name', '')} ({inputs.get('role', '')})"
    if block.name == "send_message":
        return f"→ {inputs.get('to', '')}"
    if block.name == "request_shutdown":
        return f"→ {inputs.get('teammate', '')}"
    if block.name == "request_plan":
        return f"→ {inputs.get('teammate', '')}"
    if block.name == "review_plan":
        return f"{inputs.get('request_id', '')} approve={inputs.get('approve', False)}"
    return inputs.get("path", "")


def agent_loop(messages: list, context: dict):
    """智能体主循环。s17 简化版：无后台任务/无 cron，聚焦自治 Agent 功能。
    所有工具同步执行，通过 TOOL_HANDLERS dispatch map 分发。"""
    system = get_system_prompt(context)
    while True:
        try:
            response = client.messages.create(
                model=MODEL, system=system, messages=messages,
                tools=TOOLS, max_tokens=8000)
        except Exception as e:
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": f"[Error] {type(e).__name__}: {e}"}]})
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
            output = handler(**block.input) if handler else "Unknown"
            # 工具结果：粗体品红标签 + 品红内容
            print(f"\033[1;35m[{block.name} -> Tool Calling Result]\033[0m \033[35m{str(output)[:300]}\033[0m")
            results.append({"type": "tool_result",
                            "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})

        # s10: 每轮后刷新 context + prompt
        context = update_context(context, messages)
        system = get_system_prompt(context)


# ── 入口：交互式 REPL ──────────────────────────────────
# 流程：读取输入 → agent_loop → 打印回复 → consume_lead_inbox 协议路由+注入 → 循环
if __name__ == "__main__":
    print("s17: autonomous agents")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []  # 消息历史
    context = {"memories": ""}  # 初始 context
    while True:
        try:
            query = input("\033[36ms17 >> \033[0m")
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

        # 统一收件箱消费：协议自动路由 + 注入到历史
        inbox = consume_lead_inbox(route_protocol=True)
        if inbox:
            inbox_text = "\n".join(
                f"From {m['from']} [{m.get('type', 'message')}]: "
                f"{m['content'][:200]}" for m in inbox)
            history.append({"role": "user",
                            "content": f"[Inbox]\n{inbox_text}"})
        print()
