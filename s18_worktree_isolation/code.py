#!/usr/bin/env python3
"""
s18: Worktree Isolation — git worktree + 任务-目录绑定 + 事件日志。

AI 编程智能体的工作目录隔离：
1. Task 数据类新增 worktree 字段（str | None）—— 任务可绑定独立工作目录
2. validate_worktree_name: 拒绝路径穿越和非法字符（正则白名单）
3. create_worktree: 验证名称 → git worktree add → 可选绑定任务
4. bind_task_to_worktree: 仅写入 worktree 字段，保持任务 pending（等待自动认领）
5. remove_worktree: 安全检查和未提交变更提醒 → 强制删除
6. run_git 返回 (ok, output) 元组，仅在成功时记录事件
7. Teammate 工具：complete_task，在绑定 worktree 时以 worktree 为 cwd 运行
8. 3 个新 Lead 工具：create_worktree, remove_worktree, keep_worktree

核心洞察：s17 的 alice 和 bob 都在同一个目录下工作——alice 改 config.py，
bob 也改 config.py，互相覆盖。s18 给每个任务一个独立的 git worktree——
.worktrees/auth/ 和 .worktrees/ui/，互不干扰。任务完成后 Lead 审查
worktree 中的变更后决定 keep（保留分支）或 remove（清理）。

目录拓扑：
  Main repo (/)
    ├── .worktrees/auth/  (branch: wt/auth)  ← Task #1
    ├── .worktrees/ui/    (branch: wt/ui)     ← Task #2
    ├── .tasks/task_xxx.json (worktree: "auth")
    └── .worktrees/events.jsonl

Run / 运行: python s18_worktree_isolation/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

Changes from s17 / 相对 s17 的变更:
  - Task dataclass gains worktree field (str | None)
  - validate_worktree_name: reject path traversal and illegal chars
  - create_worktree: validate name, git worktree add, optional task binding
  - bind_task_to_worktree: write worktree field only, keep task pending
  - remove_worktree: safety check before force, no auto-complete
  - run_git returns (ok, output), events only on success
  - Teammate tools: + complete_task, run in worktree cwd when bound
  - scan_unclaimed_tasks: uses can_start() for dependency checking
  - idle_poll: checks claim result, dispatches shutdown in IDLE
  - consume_lead_inbox: unified inbox consumer
  - 3 new Lead tools: create_worktree, remove_worktree, keep_worktree
"""

import os, subprocess, json, time, random, threading, re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field

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
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# ═══════════════════════════════════════════════════════════
#  FROM s12 + s18: 任务系统（含 worktree 字段）
# ═══════════════════════════════════════════════════════════

TASKS_DIR = WORKDIR / ".tasks"  # 任务 JSON 文件存储目录
TASKS_DIR.mkdir(exist_ok=True)


@dataclass
class Task:
    """任务数据类。s18 新增 worktree 字段——任务可绑定独立工作目录。"""
    id: str
    subject: str
    description: str
    status: str          # pending | in_progress | completed
    owner: str | None    # 认领者名称
    blockedBy: list[str] # 依赖任务 ID 列表
    worktree: str | None = None  # s18: 绑定的 worktree 名称


def _task_path(task_id: str) -> Path:
    """获取任务 JSON 文件的路径。"""
    return TASKS_DIR / f"{task_id}.json"


def create_task(subject: str, description: str = "",
                blockedBy: list[str] | None = None) -> Task:
    """创建新任务并持久化。"""
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
    """从磁盘加载单个任务 JSON。"""
    return Task(**json.loads(_task_path(task_id).read_text()))


def list_tasks() -> list[Task]:
    """按文件名排序列出所有任务。"""
    return [Task(**json.loads(p.read_text()))
            for p in sorted(TASKS_DIR.glob("task_*.json"))]


def get_task_json(task_id: str) -> str:
    """获取单个任务的完整 JSON 详情。"""
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)


def can_start(task_id: str) -> bool:
    """依赖检查：所有 blockedBy 任务都已完成。"""
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists():
            return False
        if load_task(dep_id).status != "completed":
            return False
    return True


def claim_task(task_id: str, owner: str = "agent") -> str:
    """认领任务：验证 pending → 检查 owner → 依赖满足 → 设 owner。"""
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    if task.owner:
        return f"Task {task_id} already owned by {task.owner}"
    if not can_start(task_id):
        deps = [d for d in task.blockedBy
                if _task_path(d).exists() and load_task(d).status != "completed"]
        missing = [d for d in task.blockedBy if not _task_path(d).exists()]
        parts = []
        if deps: parts.append(f"blocked by: {deps}")
        if missing: parts.append(f"missing deps: {missing}")
        return "Cannot start — " + ", ".join(parts)
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[claim] {task.subject} → in_progress\033[0m")
    return f"Claimed {task.id} ({task.subject})"


def complete_task(task_id: str) -> str:
    """完成任务：in_progress → completed，报告下游解封。"""
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    task.status = "completed"
    save_task(task)
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy and can_start(t.id)]
    print(f"  \033[32m[complete] {task.subject} ✓\033[0m")
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
    return msg


# ═══════════════════════════════════════════════════════════
#  NEW in s18: Worktree System（工作目录隔离系统）
#  git worktree 创建独立分支 + 任务绑定 + 安全检查 + 事件日志
# ═══════════════════════════════════════════════════════════

WORKTREES_DIR = WORKDIR / ".worktrees"  # worktree 存储目录
WORKTREES_DIR.mkdir(exist_ok=True)

VALID_WT_NAME = re.compile(r'^[A-Za-z0-9._-]{1,64}$')  # 名称白名单：只允许字母数字和 ._-


def validate_worktree_name(name: str) -> str | None:
    """验证 worktree 名称：拒绝空字符串、路径穿越（./..）和非法字符。
    返回错误消息或 None（通过）。"""
    if not name:
        return "Worktree name cannot be empty"
    if name == "." or name == "..":
        return f"'{name}' is not a valid worktree name"
    if not VALID_WT_NAME.match(name):
        return (f"Invalid worktree name '{name}': "
                "only letters, digits, dots, underscores, dashes (1-64 chars)")
    return None


def run_git(args: list[str]) -> tuple[bool, str]:
    """执行 git 命令。返回 (ok, output) 元组——调用者自行判断成功与否。"""
    try:
        r = subprocess.run(["git"] + args, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=30)
        out = (r.stdout + r.stderr).strip()
        out = out[:5000] if out else "(no output)"
        return r.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "Error: git timeout"


def log_event(event_type: str, worktree_name: str, task_id: str = ""):
    """向 .worktrees/events.jsonl 追加生命周期事件（create/remove/keep）。"""
    event = {"type": event_type, "worktree": worktree_name,
             "task_id": task_id, "ts": time.time()}
    events_file = WORKTREES_DIR / "events.jsonl"
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")


def create_worktree(name: str, task_id: str = "") -> str:
    """创建 git worktree（独立分支 wt/{name}）。可选绑定到任务。
    流程：验证名称 → git worktree add path -b wt/{name} → 可选 bind → 记录事件。"""
    err = validate_worktree_name(name)
    if err:
        return f"Error: {err}"
    path = WORKTREES_DIR / name # worktree 路径
    if path.exists():
        return f"Worktree '{name}' already exists at {path}"
    ok, result = run_git(["worktree", "add", str(path), "-b", f"wt/{name}", "HEAD"]) # git worktree add path -b wt/name HEAD
    if not ok:
        return f"Git error: {result}"
    if task_id:
        bind_task_to_worktree(task_id, name)
    log_event("create", name, task_id)
    print(f"  \033[33m[worktree] created: {name} at {path}\033[0m") # 日志输出：黄色显示 worktree 创建事件
    return f"Worktree '{name}' created at {path}"


def bind_task_to_worktree(task_id: str, worktree_name: str):
    """将任务绑定到 worktree：仅写入 worktree 字段，保持 pending 状态。
    任务仍然等待队友自动认领——bind 不改变 owner 或 status。"""
    task = load_task(task_id)
    task.worktree = worktree_name
    save_task(task)
    print(f"  \033[33m[bind] {task.subject} → worktree:{worktree_name}\033[0m")


def _count_worktree_changes(path: Path) -> tuple[int, int]:
    """统计 worktree 中未提交文件数和未推送提交数。返回 (files, commits)。"""
    try:
        r1 = subprocess.run(["git", "status", "--porcelain"],
                            cwd=path, capture_output=True, text=True, timeout=10) # git status --porcelain 输出未提交变更的简洁列表
        files = len([l for l in r1.stdout.strip().splitlines() if l.strip()])
        r2 = subprocess.run(["git", "log", "@{push}..HEAD", "--oneline"],
                            cwd=path, capture_output=True, text=True, timeout=10) # git log @{push}..HEAD --oneline 输出未推送提交的简洁列表
        commits = len([l for l in r2.stdout.strip().splitlines() if l.strip()])
        return files, commits
    except Exception:
        return -1, -1  # 无法统计时返回 -1


def remove_worktree(name: str, discard_changes: bool = False) -> str:
    """删除 worktree：安全检查（未提交变更）→ git worktree remove --force → 删除分支。
    默认拒绝有未提交变更的删除——需要 discard_changes=true 或先用 keep_worktree。"""
    err = validate_worktree_name(name)
    if err:
        return err
    path = WORKTREES_DIR / name
    if not path.exists():
        return f"Worktree '{name}' not found"
    if not discard_changes: # 默认不允许删除有未提交变更的 worktree，除非 discard_changes=true
        files, commits = _count_worktree_changes(path)
        if files < 0: # 无法统计变更时的安全提示
            return (f"Cannot verify worktree '{name}' status. "
                    "Use discard_changes=true to force removal.")
        if files > 0 or commits > 0: #  有未提交或未推送变更时的安全提示
            return (f"Worktree '{name}' has {files} uncommitted file(s) "
                    f"and {commits} unpushed commit(s). "
                    "Use discard_changes=true to force removal, "
                    "or keep_worktree to preserve for review.")
    ok1, _ = run_git(["worktree", "remove", str(path), "--force"])
    if not ok1:
        return f"Failed to remove worktree directory for '{name}'"
    run_git(["branch", "-D", f"wt/{name}"])  # 清理分支（即使失败也不影响结果）
    log_event("remove", name)
    print(f"  \033[33m[worktree] removed: {name}\033[0m")
    return f"Worktree '{name}' removed"


def keep_worktree(name: str) -> str:
    """保留 worktree 供人工审查。分支 wt/{name} 保留在 git 中。"""
    err = validate_worktree_name(name)
    if err:
        return err
    log_event("keep", name)
    print(f"  \033[36m[worktree] kept: {name}\033[0m")
    return f"Worktree '{name}' kept for review (branch: wt/{name})"


# ═══════════════════════════════════════════════════════════
#  FROM s10: Prompt 组装（含 worktree 工具）
# ═══════════════════════════════════════════════════════════

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file, "
             "create_task, list_tasks, get_task, claim_task, complete_task, "
             "spawn_teammate, send_message, check_inbox, "
             "request_shutdown, request_plan, review_plan, "
             "create_worktree, remove_worktree, keep_worktree.",
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
    """获取 system prompt（带确定性缓存）。"""
    global _last_context_hash, _last_prompt
    h = json.dumps(context, sort_keys=True)
    if h == _last_context_hash and _last_prompt:
        return _last_prompt
    _last_context_hash, _last_prompt = h, assemble_system_prompt(context)
    return _last_prompt


# ── 基础工具实现（s18: 新增 cwd 参数支持 worktree 目录）──
def safe_path(p: str, cwd: Path = None) -> Path:
    """路径安全校验。cwd 参数支持 worktree 目录。"""
    base = cwd or WORKDIR
    path = (base / p).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str, cwd: Path = None) -> str:
    """执行 shell 命令。cwd 参数可在 worktree 目录中执行。"""
    try:
        r = subprocess.run(command, shell=True, cwd=cwd or WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None, cwd: Path = None) -> str:
    """读取文件内容。cwd 参数支持 worktree 目录。"""
    try:
        lines = safe_path(path, cwd).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str, cwd: Path = None) -> str:
    """写入文件。cwd 参数支持 worktree 目录。"""
    try:
        fp = safe_path(path, cwd)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════
#  FROM s15/s16: MessageBus + ProtocolState
# ═══════════════════════════════════════════════════════════

MAILBOX_DIR = WORKDIR / ".mailboxes"
MAILBOX_DIR.mkdir(exist_ok=True)


class MessageBus:
    """基于文件的消息总线。"""

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message", metadata: dict = None):
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time(), "metadata": metadata or {}}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a") as f:
            f.write(json.dumps(msg) + "\n")
        print(f"  \033[33m[bus] {from_agent} → {to_agent}: "
              f"({msg_type}) {content[:50]}\033[0m")

    def read_inbox(self, agent: str) -> list[dict]:
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        msgs = [json.loads(line) for line in inbox.read_text().splitlines()
                if line.strip()]
        inbox.unlink()
        return msgs


BUS = MessageBus()
active_teammates: dict[str, bool] = {}


@dataclass
class ProtocolState:
    """协议状态数据类。"""
    request_id: str
    type: str
    sender: str
    target: str
    status: str
    payload: str
    created_at: float = field(default_factory=time.time)


pending_requests: dict[str, ProtocolState] = {}


def new_request_id() -> str:
    return f"req_{random.randint(0, 999999):06d}"


def match_response(response_type: str, request_id: str, approve: bool):
    """协议响应匹配：三层校验。"""
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
    print(f"  \033[{color}m[protocol] {state.type} {icon} "
          f"({request_id}: {state.status})\033[0m")


def consume_lead_inbox(route_protocol=True) -> list[dict]:
    """统一消费 Lead 收件箱。"""
    msgs = BUS.read_inbox("lead")
    if route_protocol:
        for msg in msgs:
            meta = msg.get("metadata", {})
            req_id = meta.get("request_id", "")
            msg_type = msg.get("type", "")
            if req_id and msg_type.endswith("_response"):
                match_response(msg_type, req_id, meta.get("approve", False))
    return msgs


# ═══════════════════════════════════════════════════════════
#  FROM s17 + s18: Autonomous Agent（worktree 感知）
#  idle_poll 在 auto-claim 时读取 worktree 信息注入到消息
# ═══════════════════════════════════════════════════════════

IDLE_POLL_INTERVAL = 5   # 空闲轮询间隔
IDLE_TIMEOUT = 60         # 空闲超时


def scan_unclaimed_tasks() -> list[dict]:
    """扫描任务板：pending + 无 owner + 依赖全部满足。"""
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and can_start(task["id"])):
            unclaimed.append(task)
    return unclaimed


def idle_poll(agent_name: str, messages: list,
              name: str, role: str) -> str:
    """IDLE 阶段轮询。s18 新增：auto-claim 时注入 worktree 路径信息。"""
    for _ in range(IDLE_TIMEOUT // IDLE_POLL_INTERVAL):
        time.sleep(IDLE_POLL_INTERVAL)

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
                    print(f"  \033[35m[protocol] {name} approved shutdown "
                          f"in idle ({req_id})\033[0m")
                    return "shutdown"

            # 非协议收件箱消息 → 注入并恢复工作
            messages.append({"role": "user",
                "content": "<inbox>" + json.dumps(inbox) + "</inbox>"})
            print(f"  \033[36m[idle] {name} found inbox messages\033[0m")
            return "work"

        # 优先级 2: 自动认领任务（带 worktree 信息注入）
        unclaimed = scan_unclaimed_tasks()
        if unclaimed:
            task_data = unclaimed[0]
            result = claim_task(task_data["id"], agent_name)
            if "Claimed" in result:
                wt_info = ""
                if task_data.get("worktree"):  # s18：如果任务有 worktree，告诉队友路径
                    wt_path = WORKTREES_DIR / task_data["worktree"]
                    wt_info = f"\nWork directory: {wt_path}"
                messages.append({"role": "user",
                    "content": f"<auto-claimed>Task {task_data['id']}: "
                               f"{task_data['subject']}{wt_info}</auto-claimed>"})
                print(f"  \033[32m[idle] {name} auto-claimed: "
                      f"{task_data['subject']}\033[0m")
                return "work"
            print(f"  \033[33m[idle] {name} claim failed: "
                  f"{result}\033[0m")

    print(f"  \033[31m[idle] {name} timeout ({IDLE_TIMEOUT}s)\033[0m")
    return "timeout"


# ═══════════════════════════════════════════════════════════
#  FROM s17 + s18: Teammate Thread（worktree 感知）
#  s18 核心变更：wt_ctx 追踪当前 worktree 作为 cwd
#  队友 claim 有 worktree 的任务后，bash/read/write 自动在该目录执行
# ═══════════════════════════════════════════════════════════

def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    """生成 worktree 感知的自治队友。s18 新增 wt_ctx：追踪当前 worktree
    作为 bash/read/write 的工作目录。claim 有 worktree 的任务时自动切换 cwd。"""
    if name in active_teammates:
        return f"Teammate '{name}' already exists"

    system = (f"You are '{name}', a {role}. "
              f"Use tools to complete tasks. "
              f"You can list and claim tasks from the board. "
              f"If a task has a worktree, work in that directory.") # 系统提示：告知队友他们的身份、角色、工具使用和 worktree 相关信息

    def handle_inbox_message(name: str, msg: dict, messages: list):
        """协议消息分发器。按消息类型路由。返回 True 表示应关机，否则继续工作。"""
        msg_type = msg.get("type", "message")
        meta = msg.get("metadata", {})
        req_id = meta.get("request_id", "")

        if msg_type == "shutdown_request":
            BUS.send(name, "lead", "Shutting down gracefully.",
                     "shutdown_response",
                     {"request_id": req_id, "approve": True})
            print(f"  \033[35m[protocol] {name} approved shutdown "
                  f"({req_id})\033[0m")
            return True

        if msg_type == "plan_approval_response":
            approve = meta.get("approve", False)
            if approve:
                messages.append({"role": "user",
                    "content": "[Plan approved] Proceed with the task."})
            else:
                messages.append({"role": "user",
                    "content": f"[Plan rejected] Feedback: {msg['content']}"})
        return False

    def run():
        # s18 核心：wt_ctx 追踪当前 worktree 路径，作为 bash/read/write 的 cwd
        wt_ctx = {"path": None}

        def _wt_cwd() -> Path | None:
            """获取当前 worktree 的工作目录路径。"""
            p = wt_ctx["path"]
            return Path(p) if p else None

        def _run_bash(command: str) -> str:
            """在 worktree 目录中执行 bash（如果已绑定）。"""
            return run_bash(command, cwd=_wt_cwd())

        def _run_read(path: str) -> str:
            """从 worktree 目录中读取文件（如果已绑定）。"""
            return run_read(path, cwd=_wt_cwd())

        def _run_write(path: str, content: str) -> str:
            """向 worktree 目录中写入文件（如果已绑定）。"""
            return run_write(path, content, cwd=_wt_cwd())

        def _run_list_tasks():
            """列出任务板，包含 worktree 信息。"""
            tasks = list_tasks()
            if not tasks:
                return "No tasks."
            return "\n".join(
                f"  {t.id}: {t.subject} [{t.status}]"
                + (f" (wt:{t.worktree})" if t.worktree else "")
                for t in tasks)

        def _run_claim_task(task_id: str):
            """认领任务。s18：如果任务有 worktree，自动设置 wt_ctx。"""
            result = claim_task(task_id, owner=name)
            if "Claimed" in result:
                task = load_task(task_id)
                if task.worktree:
                    wt_ctx["path"] = str(WORKTREES_DIR / task.worktree)  # 切换到 worktree
                else:
                    wt_ctx["path"] = None  # 无 worktree → 使用默认目录
            return result

        def _run_complete_task(task_id: str):
            """完成任务。s18：完成后清除 wt_ctx（下一个任务可能有不同 worktree，认领时会重新设置）。"""
            result = complete_task(task_id)
            wt_ctx["path"] = None  # 完成当前任务后清除 worktree cwd
            return result

        messages = [{"role": "user", "content": prompt}]
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

        sub_handlers = {
            "bash": _run_bash, "read_file": _run_read,
            "write_file": _run_write,
            "send_message": lambda to, content: (BUS.send(name, to, content),
                                                  "Sent")[1],
            "submit_plan": lambda plan: _teammate_submit_plan(name, plan),
            "list_tasks": _run_list_tasks,
            "claim_task": _run_claim_task,
            "complete_task": _run_complete_task,
        }

        # 外层循环：WORK → IDLE 交替
        while True:
            # Identity re-injection
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
                if inbox and not should_shutdown:
                    non_protocol = [m for m in inbox
                                    if m.get("type") == "message"]
                    if non_protocol:
                        messages.append({"role": "user",
                            "content": "<inbox>" + json.dumps(non_protocol) + "</inbox>"})

                try:
                    response = client.messages.create(
                        model=MODEL, system=system, messages=messages[-20:],
                        tools=sub_tools, max_tokens=8000)
                except Exception:
                    break
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break
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

            # IDLE 阶段
            idle_result = idle_poll(name, messages, name, role)
            if idle_result == "shutdown":
                break
            if idle_result == "timeout":
                break

        # SHUTDOWN 阶段：最终摘要
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
        print(f"  \033[32m[teammate] {name} finished\033[0m") # 日志输出：绿色显示队友完成

    active_teammates[name] = True
    threading.Thread(target=run, daemon=True).start()
    print(f"  \033[36m[teammate] {name} spawned as {role}\033[0m") # 日志输出：青色显示队友 spawned
    return f"Teammate '{name}' spawned as {role} (autonomous)"


def _teammate_submit_plan(from_name: str, plan: str) -> str:
    """队友提交计划给 Lead 审批。"""
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
#  FROM s16 + s18: Lead 工具（协议 + Worktree）
# ═══════════════════════════════════════════════════════════

def run_request_shutdown(teammate: str) -> str:
    """发起关机协议。"""
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
    """要求队友提交计划。"""
    BUS.send("lead", teammate, f"Please submit a plan for: {task}",
             "message")
    return f"Asked {teammate} to submit a plan"


def run_review_plan(request_id: str, approve: bool,
                    feedback: str = "") -> str:
    """审批/拒绝计划。"""
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


# s18 新增：Worktree 工具
def run_create_worktree(name: str, task_id: str = "") -> str:
    """创建独立 git worktree。"""
    return create_worktree(name, task_id)


def run_remove_worktree(name: str, discard_changes: bool = False) -> str:
    """删除 worktree（安全检查 + 可选强制）。"""
    return remove_worktree(name, discard_changes)


def run_keep_worktree(name: str) -> str:
    """保留 worktree 供审查。"""
    return keep_worktree(name)


# ═══════════════════════════════════════════════════════════
#  基础工具 handlers
# ═══════════════════════════════════════════════════════════

def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    """创建新任务。"""
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    """列出所有任务（含 worktree 信息）。"""
    tasks = list_tasks()
    if not tasks:
        return "No tasks."
    return "\n".join(
        f"  {t.id}: {t.subject} [{t.status}]"
        + (f" (wt:{t.worktree})" if t.worktree else "")
        for t in tasks)


def run_get_task(task_id: str) -> str:
    return get_task_json(task_id)


def run_claim_task(task_id: str) -> str:
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    return complete_task(task_id)


def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    return spawn_teammate_thread(name, role, prompt)


def run_send_message(to: str, content: str) -> str:
    BUS.send("lead", to, content)
    return f"Sent to {to}"


def run_check_inbox() -> str:
    """检查收件箱（协议响应自动路由）。"""
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


# ── 工具定义（s18 新增 3 个 worktree 工具，共 17 个）──
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
    # s18 new: worktree tools
    {"name": "create_worktree",
     "description": "Create an isolated git worktree with its own branch.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"},
                                     "task_id": {"type": "string"}},
                      "required": ["name"]}},
    {"name": "remove_worktree",
     "description": "Remove a worktree. Refuses if uncommitted changes unless discard_changes=true.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"},
                                     "discard_changes": {"type": "boolean"}},
                      "required": ["name"]}},
    {"name": "keep_worktree",
     "description": "Keep a worktree for manual review.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"}},
                      "required": ["name"]}},
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
    "create_worktree": run_create_worktree,
    "remove_worktree": run_remove_worktree,
    "keep_worktree": run_keep_worktree,
}


# ═══════════════════════════════════════════════════════════
#  FROM s10: Context 系统
# ═══════════════════════════════════════════════════════════

MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"


def update_context(context: dict, messages: list) -> dict:
    """从真实文件系统状态推导 context。"""
    memories = ""
    if MEMORY_INDEX.exists():
        memories = MEMORY_INDEX.read_text()[:2000]
    return {"memories": memories}


# ═══════════════════════════════════════════════════════════
#  agent_loop — s18 核心：精简版循环（无后台/无 cron）
# ═══════════════════════════════════════════════════════════

def _tool_input_summary(block) -> str:
    """提取工具调用的关键参数摘要。"""
    inputs = block.input
    if block.name == "bash":
        return inputs.get("command", "")
    if block.name in ("create_task",):
        return inputs.get("subject", "")
    if block.name in ("get_task", "claim_task", "complete_task"):
        return inputs.get("task_id", "")
    if block.name in ("list_tasks", "check_inbox", "list_crons"):
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
    if block.name == "create_worktree":
        return f"{inputs.get('name', '')}"
    if block.name in ("remove_worktree", "keep_worktree"):
        return f"{inputs.get('name', '')}"
    return inputs.get("path", "")


def agent_loop(messages: list, context: dict):
    """智能体主循环。精简版：同步工具执行，TOOL_HANDLERS dispatch。"""
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
        context = update_context(context, messages)
        system = get_system_prompt(context)


# ── 入口：交互式 REPL ──────────────────────────────────
if __name__ == "__main__":
    print("s18: worktree isolation")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []
    context = {"memories": ""}
    while True:
        try:
            query = input("\033[36ms18 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history, context)
        context = update_context(context, history)
        # 打印模型最终文本回复（蓝色）
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(f"\033[34m{block.text}\033[0m")

        # 统一收件箱消费：协议路由 + 注入
        inbox = consume_lead_inbox(route_protocol=True)
        if inbox:
            inbox_text = "\n".join(
                f"From {m['from']} [{m.get('type', 'message')}]: "
                f"{m['content'][:200]}" for m in inbox)
            history.append({"role": "user",
                            "content": f"[Inbox]\n{inbox_text}"})
        print()
