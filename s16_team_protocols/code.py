#!/usr/bin/env python3
"""
s16: Team Protocols — request-response 协议 + request_id 关联 + dispatch + 状态机。

AI 编程智能体的团队协作协议：
1. ProtocolState 数据类（request_id, type, sender, status, created_at）—— 结构化协议请求
2. pending_requests dict: 跟踪所有进行中的协议请求
3. dispatch_message: 按消息类型路由到对应 handler（shutdown_request / plan_approval_response）
4. request_shutdown: Lead 发起关机协议请求
5. request_plan / review_plan: Lead 要求队友提交计划 + 审批
6. handle_shutdown_request / handle_plan_response: 队友接收并响应协议消息
7. match_response: Lead 通过 request_id 关联响应到原始请求（含类型校验）
8. Teammate idle loop: 等待收件箱消息而非 10 轮后退出（真正的持久队友）
9. Unified consume_lead_inbox: 协议路由 + 历史注入，解决 s15 的重复消费竞态
10. 3 个新 Lead 工具：request_shutdown, request_plan, review_plan
11. 1 个新队友工具：submit_plan

核心洞察：s15 的队友能通信但协议靠约定——Lead 发 "please stop"，队友可能不理。
s16 把约定升级为协议——每条请求带 request_id，响应必须引用 request_id，
Lead 端 match_response 校验类型和状态。不是"随便聊聊"，是"协议握手"。

协议数据流：
  Lead: BUS.send("shutdown_request", {request_id}) ──────→ teammate inbox
  Teammate: dispatch → handler → BUS.send("shutdown_response", {request_id}) ─→ Lead inbox
  Lead: consume_lead_inbox → match_response(request_id) → pending_requests[req_id].status = approved

Run / 运行: python s16_team_protocols/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

Changes from s15 / 相对 s15 的变更:
  - ProtocolState dataclass (request_id, type, sender, status, created_at)
  - pending_requests dict: tracks in-flight protocol requests
  - dispatch_message: routes incoming messages by type to handlers
  - request_shutdown: Lead sends shutdown protocol request
  - request_plan: Lead asks teammate to submit plan
  - handle_shutdown_request / handle_plan_response: teammate receives & responds
  - match_response: Lead correlates response to request via request_id (with type validation)
  - Teammate idle loop: waits for inbox messages instead of exiting after 10 rounds
  - Unified consume_lead_inbox: protocol routing + injection into history
  - 3 new Lead tools: request_shutdown, request_plan, review_plan
  - 1 new teammate tool: submit_plan
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
MEMORY_DIR = WORKDIR / ".memory"  # s09: 记忆目录
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"  # 记忆索引
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# ═══════════════════════════════════════════════════════════
#  FROM s12: 任务系统（Task System）
#  文件持久化任务图，带 blockedBy 依赖关系 + 5 个工具
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
    """认领任务：验证 pending 状态 → 检查依赖 → 设 owner + 改为 in_progress。"""
    task = load_task(task_id)
    if task.status != "pending":  # 如果任务已被别人认领或完成，就不能认领了
        return f"Task {task_id} is {task.status}, cannot claim"
    if not can_start(task_id):  # 如果依赖未满足，也不能认领
        deps = [d for d in task.blockedBy
                if not _task_path(d).exists() or load_task(d).status != "completed"]
        return f"Blocked by: {deps}"
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress (owner: {owner})\033[0m")  # 青色标签显示认领的任务和所有者
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
    print(f"  \033[32m[complete] {task.subject} ✓\033[0m")  # 绿色标签显示完成的任务
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
        print(f"  \033[33m[unblocked] {', '.join(unblocked)}\033[0m")  # 黄色标签显示被解封的任务
    return msg


# ═══════════════════════════════════════════════════════════
#  FROM s10: Prompt 组装（同步，tools 段包含协议工具）
# ═══════════════════════════════════════════════════════════

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file, "
             "get_task, create_task, list_tasks, claim_task, complete_task, "
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
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")
    return "\n\n".join(sections)


_last_context_key, _last_prompt = None, None


def get_system_prompt(context: dict) -> str:
    """获取 system prompt（带确定性缓存，避免重复组装）。"""
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


def run_bash(command: str, run_in_background: bool = False) -> str:
    """执行 shell 命令并返回输出。
    run_in_background 参数由 agent_loop 分发处理，不在此处使用。"""
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
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")  # 蓝色标签显示创建的任务和依赖
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


# ═══════════════════════════════════════════════════════════
#  FROM s13: 后台任务系统（Background Tasks）
#  守护线程异步执行慢操作 + <task_notification> 通知注入
# 注意：start_background_task 调用 execute_tool（定义在文件后半部分），
# Python 在调用时动态解析，所以定义顺序不影响运行
# ═══════════════════════════════════════════════════════════

_bg_counter = 0  # 全局计数器生成唯一的后台任务 ID
background_tasks: dict[str, dict] = {}   # bg_id → {tool_use_id, command, status} 生命周期跟踪
background_results: dict[str, str] = {}   # bg_id → output 线程安全结果存储
background_lock = threading.Lock()        # 互斥锁，保护共享字典的并发读写


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """启发式判断是否为慢操作（预计 > 30s 的命令）。"""
    if tool_name != "bash":
        return False
    cmd = tool_input.get("command", "").lower()
    slow_keywords = ["install", "build", "test", "deploy", "compile",
                     "docker build", "pip install", "npm install",
                     "cargo build", "pytest", "make"]
    return any(kw in cmd for kw in slow_keywords)


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """判断工具是否应在后台执行。
    优先级：模型显式声明 run_in_background=True → 启发式回退。"""
    if tool_input.get("run_in_background"):
        return True
    return is_slow_operation(tool_name, tool_input)


def start_background_task(block) -> str:
    """将工具调用分发到守护线程后台执行，返回后台任务 ID。
    主线程不等结果——先返回占位符让 agent 继续工作。"""
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    cmd = block.input.get("command", block.name)

    def worker():
        """后台工作线程：执行工具 → 结果写入共享字典（加锁保护）。"""
        result = execute_tool(block)  # execute_tool 定义在文件后半部分（Python 运行时动态解析）
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": cmd,
            "status": "running",
        }
    threading.Thread(target=worker, daemon=True).start()  # daemon=True: 主线程退出时自动回收
    print(f"  \033[33m[background] dispatched {bg_id}: {cmd[:40]}\033[0m")
    return bg_id


def collect_background_results() -> list[str]:
    """收集已完成后台任务的结果，转为 <task_notification> 格式的消息列表。
    加锁取出后从字典中清理，避免重复注入。"""
    with background_lock:
        ready_ids = [bid for bid, task in background_tasks.items()
                     if task["status"] == "completed"]
    notifications = []
    for bg_id in ready_ids:
        with background_lock:
            task = background_tasks.pop(bg_id)  # 从生命周期跟踪字典中移除已完成的任务
            output = background_results.pop(bg_id, "")  # 获取结果后从结果字典中移除
        summary = output[:200] if len(output) > 200 else output  # 截断长输出
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>")
        print(f"  \033[32m[background done] {bg_id}: "
              f"{task['command'][:40]} ({len(output)} chars)\033[0m")
    return notifications


# ═══════════════════════════════════════════════════════════
#  FROM s15: MessageBus（消息总线）
#  基于文件的邮箱系统，s16 新增 metadata 字段支持协议关联
# ═══════════════════════════════════════════════════════════

MAILBOX_DIR = WORKDIR / ".mailboxes"  # 邮箱文件存储目录
MAILBOX_DIR.mkdir(exist_ok=True)


class MessageBus:
    """基于文件的消息总线。s16 新增 metadata 字段用于协议路由。
    - send: 追加一行 JSON（含 metadata）到目标收件箱
    - read_inbox: 读取全部消息后 unlink（消费语义）"""

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message", metadata: dict = None):
        """发送消息：追加 JSON 行到目标 agent 的 .jsonl 收件箱。
        metadata 携带 request_id 等协议关联信息。"""
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time(), "metadata": metadata or {}}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a") as f:
            f.write(json.dumps(msg) + "\n")
        print(f"  \033[33m[bus] {from_agent} → {to_agent}: "
              f"({msg_type}) {content[:50]}\033[0m")

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

# ═══════════════════════════════════════════════════════════
#  NEW in s16: 协议状态机（Protocol State Machine）
#  ProtocolState 跟踪每个协议请求的完整生命周期
#  pending_requests dict 是所有进行中请求的"注册表"
# ═══════════════════════════════════════════════════════════

@dataclass
class ProtocolState:
    """协议请求状态数据类。追踪每个 request 的完整生命周期。
    type: "shutdown"（关机协议）| "plan_approval"（计划审批协议）
    status: pending → approved | rejected"""
    request_id: str
    type: str       # "shutdown" | "plan_approval"
    sender: str     # 请求发起方
    target: str     # 请求接收方
    status: str     # pending | approved | rejected
    payload: str    # 计划文本或关机原因
    created_at: float = field(default_factory=time.time)


pending_requests: dict[str, ProtocolState] = {}  # request_id → ProtocolState 全局注册表


def new_request_id() -> str:
    """生成唯一的协议请求 ID。"""
    return f"req_{random.randint(0, 999999):06d}"


def match_response(response_type: str, request_id: str, approve: bool):
    """Lead 端协议响应匹配：通过 request_id 关联响应到原始请求。
    三层验证：(1) request_id 存在 (2) 响应类型匹配请求类型 (3) 状态仍为 pending"""
    state = pending_requests.get(request_id)
    if not state:
        print(f"  \033[31m[protocol] unknown request_id: {request_id}\033[0m")
        return
    # 验证响应类型与请求类型匹配（防止 shutdown_request 被 plan 响应误匹配）
    if state.type == "shutdown" and response_type != "shutdown_response":
        print(f"  \033[31m[protocol] type mismatch: expected shutdown_response, "
              f"got {response_type}\033[0m")
        return
    if state.type == "plan_approval" and response_type != "plan_approval_response":
        print(f"  \033[31m[protocol] type mismatch: expected plan_approval_response, "
              f"got {response_type}\033[0m")
        return
    if state.status != "pending":  # 防止重复处理
        print(f"  \033[33m[protocol] {request_id} already {state.status}, "
              f"ignoring duplicate\033[0m")
        return
    state.status = "approved" if approve else "rejected"
    icon = "✓" if approve else "✗"
    color = "32" if approve else "31"
    print(f"  \033[{color}m[protocol] {state.type} {icon} " # 绿色✓表示批准，红色✗表示拒绝
          f"({request_id}: {state.status})\033[0m")


# ═══════════════════════════════════════════════════════════
#  NEW in s16: 统一收件箱消费者（Unified Lead Inbox Consumer）
#  解决 s15 的竞态问题：check_inbox 工具和主循环都走同一个函数
#  协议响应先经过 match_response 路由，再返回消息列表
# ═══════════════════════════════════════════════════════════

def consume_lead_inbox(route_protocol: bool = True) -> list[dict]:
    """读取 Lead 收件箱，协议响应先路由再返回。
    route_protocol=True（默认）：自动匹配并更新 pending_requests 状态。
    由 run_check_inbox() 和主循环共同调用，避免消息被消费后协议未路由。"""
    msgs = BUS.read_inbox("lead")
    if not msgs:
        return []
    if route_protocol:
        for msg in msgs:
            meta = msg.get("metadata", {})
            req_id = meta.get("request_id", "")
            msg_type = msg.get("type", "")
            if req_id and msg_type.endswith("_response"):
                approve = meta.get("approve", False)
                match_response(msg_type, req_id, approve)
    return msgs


# ═══════════════════════════════════════════════════════════
#  NEW in s16: Teammate Thread（协议感知的队友线程）
#  s15 的 idle loop 升级版：通过 handle_inbox_message 按协议类型分发
#  队友收到 shutdown_request → 自动响应 shutdown_response → 退出
#  队友收到 plan_approval_response → 继续执行或根据反馈修改
# ═══════════════════════════════════════════════════════════

def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    """生成协议感知的队友 Agent 线程。
    使用 idle loop：每轮 LLM 调用后等待收件箱消息（shutdown_request 等），
    不再简单 10 轮退出。"""
    if name in active_teammates:
        return f"Teammate '{name}' already exists"

    system = (f"You are '{name}', a {role}. "
              f"Use tools to complete tasks. "
              f"Check inbox for protocol messages (shutdown_request, etc).") # 系统提示明确告知队友要检查收件箱以响应协议消息

    def handle_inbox_message(name: str, msg: dict, messages: list) -> bool:
        """协议消息分发器：按消息类型路由到对应 handler。
        返回 True 表示队友应停止（shutdown）。
        只有两种协议消息会触发 handler：shutdown_request 和 plan_approval_response。"""
        msg_type = msg.get("type", "message")
        meta = msg.get("metadata", {})
        req_id = meta.get("request_id", "")

        if msg_type == "shutdown_request":
            # 队友收到关机请求 → 自动同意并回复
            BUS.send(name, "lead", "Shutting down gracefully.",
                     "shutdown_response",
                     {"request_id": req_id, "approve": True})
            print(f"  \033[35m[protocol] {name} approved shutdown "
                  f"({req_id})\033[0m")
            return True  # 停止循环

        if msg_type == "plan_approval_response":
            # 队友收到计划审批结果 → 根据结果决定继续或修改
            approve = meta.get("approve", False)
            if approve:
                messages.append({"role": "user",
                    "content": f"[Plan approved] Proceed with the task."})
            else:
                messages.append({"role": "user",
                    "content": f"[Plan rejected] Feedback: {msg['content']}"})

        return False  # 继续循环

    def run():
        """队友的主循环：idle loop 模式——不会 10 轮后退出。
        工作流程：收件箱检查 → 协议分发 → LLM 调用 → 工具执行 → 循环
        LLM 主动停下时进入 idle 子循环等待新消息，收到 shutdown_request 才退出。"""
        messages = [{"role": "user", "content": prompt}]
        # 队友工具有 5 个：bash/read_file/write_file/send_message/submit_plan
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
        ]
        sub_handlers = {
            "bash": run_bash, "read_file": run_read, "write_file": run_write,
            "send_message": lambda to, content: (BUS.send(name, to, content),
                                                  "Sent")[1],
            "submit_plan": lambda plan: _teammate_submit_plan(name, plan),
        }

        shutdown_requested = False
        while not shutdown_requested:
            # 每轮先检查收件箱中的协议消息
            inbox = BUS.read_inbox(name)
            should_stop = False
            non_protocol = []
            for msg in inbox:
                if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                    should_stop = handle_inbox_message(name, msg, messages) # 处理协议消息
                    if should_stop:
                        break
                else:
                    non_protocol.append(msg)
            if should_stop:
                shutdown_requested = True
                break
            if non_protocol:
                inbox_json = json.dumps(non_protocol)
                messages.append({"role": "user",
                    "content": "<inbox>" + inbox_json + "</inbox>"})

            # LLM 调用
            try:
                response = client.messages.create(
                    model=MODEL, system=system, messages=messages[-20:],
                    tools=sub_tools, max_tokens=8000)
            except Exception:
                break

            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                # Idle 子循环：LLM 停下来后不退出，等待新消息
                # 真实 CC 会发 idle_notification 给 Lead
                while not shutdown_requested:
                    time.sleep(1)
                    inbox = BUS.read_inbox(name)
                    if not inbox: # 没有新消息就继续等待，避免无意义的循环和 CPU 占用
                        continue
                    for msg in inbox:
                        if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                            should_stop = handle_inbox_message(name, msg, messages)
                            if should_stop:
                                shutdown_requested = True
                                break
                        else:
                            non_protocol.append(msg)
                    if shutdown_requested:
                        break
                    if non_protocol:
                        inbox_json = json.dumps(non_protocol)
                        messages.append({"role": "user",
                            "content": "<inbox>" + inbox_json + "</inbox>"})
                        break  # 回到 LLM 轮次，继续工作

            # 执行工具调用
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    handler = sub_handlers.get(block.name)
                    output = handler(**block.input) if handler else "Unknown"
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": str(output)})
            messages.append({"role": "user", "content": results})

        # 发送最终摘要给 Lead
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
        print(f"  \033[32m[teammate] {name} finished\033[0m") # 绿色标签显示队友完成

    active_teammates[name] = True # 注册队友
    threading.Thread(target=run, daemon=True).start() # 启动队友线程
    print(f"  \033[36m[teammate] {name} spawned as {role}\033[0m") # 青色标签显示生成的队友和角色
    return f"Teammate '{name}' spawned as {role}"


def _teammate_submit_plan(from_name: str, plan: str) -> str:
    """队友提交计划给 Lead 审批。这是协议级请求，不是代码级门禁。
    注意：提交后队友线程继续运行——它仍然可以调用 bash/write 等工具。
    真正的执行强制依赖模型在收到审批回复前自觉等待。
    代码级工具门禁需要阻塞队友的工具分发直到审批到达。"""
    req_id = new_request_id()
    pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="plan_approval",
        sender=from_name, target="lead",
        status="pending", payload=plan)
    BUS.send(from_name, "lead", plan, # 计划内容直接发在消息里，msg_type为plan_approval_request，metadata 里带 request_id 供审批回复关联
             "plan_approval_request",
             {"request_id": req_id})
    return f"Plan submitted ({req_id}). Waiting for approval..."


# ═══════════════════════════════════════════════════════════
#  NEW in s16: Lead 协议工具（3 个新工具）
#  request_shutdown: 发起关机协议
#  request_plan: 要求队友提交计划
#  review_plan: 审批或拒绝计划
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


def run_review_plan(request_id: str, approve: bool, feedback: str = "") -> str:
    """Lead 审批/拒绝队友提交的计划。更新 pending_requests 状态 → 发送响应回队友。"""
    state = pending_requests.get(request_id)
    if not state:
        return f"Request {request_id} not found"
    if state.status != "pending":
        return f"Request {request_id} already {state.status}"
    state.status = "approved" if approve else "rejected"
    BUS.send("lead", state.sender, feedback or ("Approved" if approve else "Rejected"),
             "plan_approval_response",
             {"request_id": request_id, "approve": approve})
    icon = "✓" if approve else "✗"
    print(f"  \033[32m[protocol] plan {icon} ({request_id})\033[0m")
    return f"Plan {'approved' if approve else 'rejected'} ({request_id})"


# ═══════════════════════════════════════════════════════════
#  s15: Team 工具（3 个工具）
#  [spawn](青)/[send](黄)/[inbox](品红)
# ═══════════════════════════════════════════════════════════

def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    """生成队友：后台线程启动协议感知的独立 Agent。"""
    return spawn_teammate_thread(name, role, prompt)


def run_send_message(to: str, content: str) -> str:
    """通过 MessageBus 发送消息给指定 agent。"""
    BUS.send("lead", to, content)
    return f"Sent to {to}"


def run_check_inbox() -> str:
    """检查 Lead 收件箱，自动通过 match_response 路由协议响应。"""
    msgs = consume_lead_inbox(route_protocol=True)
    if not msgs:
        return "(inbox empty)"
    lines = []
    for m in msgs:
        meta = m.get("metadata", {})
        req_id = meta.get("request_id", "")
        tag = f" [{m['type']} req:{req_id}]" if req_id else f" [{m['type']}]"
        lines.append(f"  [{m['from']}]{tag} {m['content'][:200]}")
    return "\n".join(lines)


# ── 工具分发 ─────────────────────────────────────────
def execute_tool(block) -> str:
    """执行工具调用块，返回输出结果。包含 s16 新增的 3 个协议工具。"""
    handler = {
        "bash": run_bash, "read_file": run_read, "write_file": run_write,
        "create_task": run_create_task, "list_tasks": run_list_tasks,
        "get_task": run_get_task, "claim_task": run_claim_task,
        "complete_task": run_complete_task,
        "spawn_teammate": run_spawn_teammate,
        "send_message": run_send_message, "check_inbox": run_check_inbox,
        "request_shutdown": run_request_shutdown,
        "request_plan": run_request_plan, "review_plan": run_review_plan,
    }.get(block.name)
    if handler:
        return handler(**block.input)
    return f"Unknown tool: {block.name}"


# ── 工具定义（s16 新增 3 个协议工具，共 14 个）─────────────────

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {
                          "command": {"type": "string"},
                          "run_in_background": {"type": "boolean"}},
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
    {"name": "spawn_teammate",
     "description": "Spawn a teammate agent in a background thread.",
     "input_schema": {"type": "object",
                      "properties": {
                          "name": {"type": "string"},
                          "role": {"type": "string"},
                          "prompt": {"type": "string"}},
                      "required": ["name", "role", "prompt"]}},
    {"name": "send_message",
     "description": "Send message to a teammate via MessageBus.",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["to", "content"]}},
    {"name": "check_inbox",
     "description": "Check Lead's inbox. Routes protocol responses automatically.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
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
     "description": "Approve or reject a submitted plan by request_id.",
     "input_schema": {"type": "object",
                      "properties": {
                          "request_id": {"type": "string"},
                          "approve": {"type": "boolean"},
                          "feedback": {"type": "string"}},
                      "required": ["request_id", "approve"]}},
]


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
        "enabled_tools": [t["name"] for t in TOOLS],
        "workspace": str(WORKDIR),
        "memories": memories,
    }


# ═══════════════════════════════════════════════════════════
#  agent_loop — s16 核心：协议感知的工具执行循环
#  教学版保持基本循环，S11 的完整错误恢复被省略。
#  s16 移除了 cron 调度器（聚焦团队协议），简化了循环。
# ═══════════════════════════════════════════════════════════

def _tool_input_summary(block) -> str:
    """提取工具调用的关键参数摘要，用于终端显示。"""
    inputs = block.input
    if block.name == "bash":
        return inputs.get("command", "")
    if block.name in ("create_task",):
        return inputs.get("subject", "")
    if block.name in ("get_task", "claim_task", "complete_task", "cancel_cron"):
        return inputs.get("task_id", inputs.get("job_id", ""))
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
    """智能体主循环 + 协议工具 + 后台任务。
    s16 简化版：移除了 cron 调度器，聚焦团队协议功能。"""
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

            if should_run_background(block.name, block.input):
                # 慢操作 → 后台守护线程执行，主线程立即返回占位符
                bg_id = start_background_task(block)
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"[Background task {bg_id} started] "
                                           f"Result will be available when complete."})
            else:
                # 快操作 → 同步执行
                output = execute_tool(block)
                # 工具结果：粗体品红标签 + 品红内容
                print(f"\033[1;35m[{block.name} -> Tool Calling Result]\033[0m \033[35m{str(output)[:300]}\033[0m")
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output})

        # 注入工具结果 + 后台通知到一条 user 消息中
        user_content = list(results)
        bg_notifications = collect_background_results()
        if bg_notifications:
            for notif in bg_notifications:
                user_content.append({"type": "text", "text": notif})

        messages.append({"role": "user", "content": user_content})

        # s10: 每轮后刷新 context + prompt
        context = update_context(context, messages)
        system = get_system_prompt(context)


# ── 入口：交互式 REPL ──────────────────────────────────
# 流程：读取输入 → agent_loop → 打印回复 → consume_lead_inbox 协议路由 + 注入 → 循环
if __name__ == "__main__":
    print("s16: team protocols")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []  # 消息历史
    context = update_context({}, [])  # s10: 启动时初始化 context
    while True:
        try:
            query = input("\033[36ms16 >> \033[0m")
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

        # 统一收件箱消费：协议自动路由 + 注入到历史（解决 s15 的竞态）
        inbox_msgs = consume_lead_inbox(route_protocol=True)
        if inbox_msgs:
            inbox_text = "\n".join(
                f"From {m['from']}: {m['content'][:200]}" for m in inbox_msgs)
            history.append({"role": "user",
                            "content": f"[Inbox]\n{inbox_text}"})
            print(f"\n\033[33m[Inbox: {len(inbox_msgs)} messages injected]\033[0m")
        print()
