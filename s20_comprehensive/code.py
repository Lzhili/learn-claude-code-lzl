#!/usr/bin/env python3
"""
s20: Comprehensive Agent — 全部教学组件合一，一个循环挂所有机制。

AI 编程智能体的终极形态：将 s01-s19 的机制合回一个完整 harness。
  机制很多，循环一个。

集结清单：
  s01 agent_loop · s02 dispatch · s03 permission · s04 hooks · s05 todo_write
  s06 subagent · s07 skills · s08 compaction · s09 memory · s10 prompt assembly
  s11 error recovery · s12 task graph · s13 background tasks · s14 cron scheduler
  s15 teams · s16 protocols · s17 autonomous · s18 worktrees · s19 MCP

核心洞察：前 19 章通过每次只加一个机制来教学——拆开是为了看清每部分做什么。
但真实 Agent 不会这样拆开运行。s20 把机制合回同一个 agent_loop 上：
prepare_context (L4压缩) → cron 注入 → background 通知 → LLM 调用 (with_retry)
→ PreToolUse hooks (permission+log) → 工具分发 (dispatch+bg+cron+MCP)
→ PostToolUse hooks → build_user_content → 循环。

Run / 运行: python s20_comprehensive/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY
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
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

from anthropic import Anthropic
from dotenv import load_dotenv

# ── 初始化 + 全局常量 ────────────────────────────────
load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()  # 工作目录
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
PRIMARY_MODEL = MODEL
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL_ID")

SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"

DEFAULT_MAX_TOKENS = 8000
ESCALATED_MAX_TOKENS = 16000  # max_tokens 升级目标
MAX_RETRIES = 3               # 429/529 最大重试次数
MAX_CONSECUTIVE_529 = 2       # 触发 Fallback Model 切换的连续 529 阈值
MAX_RECOVERY_RETRIES = 2      # 续写提示最大次数
BASE_DELAY_MS = 500           # 指数退避基础延迟
CONTEXT_LIMIT = 50000          # 触发 auto compact 的上下文大小阈值
KEEP_RECENT_TOOL_RESULTS = 3  # micro_compact 保留的最近 tool result 数
PERSIST_THRESHOLD = 30000     # 触发输出持久化到文件的大小阈值
CONTINUATION_PROMPT = "Continue from the previous response. Do not repeat completed work."
PROMPT = "\033[36ms20 >> \033[0m"  # 用户输入提示符（青色）
CLI_ACTIVE = False # 指示是否在 CLI 模式下运行，影响 terminal_print 行为


def terminal_print(text: str):
    """线程安全的终端输出。后台线程输出时保留用户当前的输入行。"""
    if threading.current_thread() is threading.main_thread() or not CLI_ACTIVE: # 后台线程输出时不干扰用户输入行
        print(text)
        return
    line = ""
    if READLINE_AVAILABLE: # 尝试获取用户当前输入行，避免 readline 不可用时抛异常
        try:
            line = readline.get_line_buffer()
        except Exception:
            line = ""
    print(f"\r\033[K{text}")  # 清除当前行，打印文本
    print(PROMPT + line, end="", flush=True)  # 恢复提示符和用户已输入内容

# ═══════════════════════════════════════════════════════════
#  s01-s19 组件集成
#  以下按 s01 到 s19 顺序排列：agent_loop(s01) → dispatch(s02) →
#  permission(s03) → hooks(s04) → todo_write(s05) → subagent(s06) →
#  skills(s07) → compaction(s08) → memory(s09) → prompt(s10) →
#  error_recovery(s11) → task_graph(s12) → background(s13) →
#  cron(s14) → teams(s15) → protocols(s16) → autonomous(s17) →
#  worktrees(s18) → mcp(s19)
# ═══════════════════════════════════════════════════════════

# ── s12: Task System（任务图）──
TASKS_DIR = WORKDIR / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)
CURRENT_TODOS: list[dict] = []  # s05: 当前会话的内存 todo 列表


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str]
    worktree: str | None = None


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


def get_task_json(task_id: str) -> str:
    """获取单个任务的完整 JSON 详情。"""
    return json.dumps(asdict(load_task(task_id)), indent=2)


def can_start(task_id: str) -> bool:
    """依赖检查：所有 blockedBy 任务文件存在且已完成。"""
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
    """完成任务：in_progress → completed，报告下游解封任务。"""
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


# ── s18: Worktree System（目录隔离）──
WORKTREES_DIR = WORKDIR / ".worktrees"
WORKTREES_DIR.mkdir(exist_ok=True)

VALID_WT_NAME = re.compile(r'^[A-Za-z0-9._-]{1,64}$')


def validate_worktree_name(name: str) -> str | None:
    """验证 worktree 名称：拒绝空、路径穿越、非法字符。返回错误或 None。"""
    if not name:
        return "Worktree name cannot be empty"
    if name in (".", ".."):
        return f"'{name}' is not a valid worktree name"
    if not VALID_WT_NAME.match(name):
        return (f"Invalid worktree name '{name}': "
                "only letters, digits, dots, underscores, dashes (1-64 chars)")
    return None


def run_git(args: list[str]) -> tuple[bool, str]:
    """执行 git 命令。返回 (ok, output) 元组。"""
    try:
        r = subprocess.run(["git"] + args, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=30)
        out = (r.stdout + r.stderr).strip()
        return r.returncode == 0, out[:5000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return False, "Error: git timeout"


def log_event(event_type: str, worktree_name: str, task_id: str = ""):
    """向 .worktrees/events.jsonl 追加生命周期事件。"""
    event = {"type": event_type, "worktree": worktree_name,
             "task_id": task_id, "ts": time.time()}
    events_file = WORKTREES_DIR / "events.jsonl"
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")


def create_worktree(name: str, task_id: str = "") -> str:
    """创建 git worktree（独立分支 wt/{name}），可选绑定到任务。"""
    err = validate_worktree_name(name)
    if err:
        return f"Error: {err}"
    if task_id:
        try:
            load_task(task_id)
        except FileNotFoundError:
            return f"Error: task {task_id} not found"
    path = WORKTREES_DIR / name
    if path.exists():
        return f"Worktree '{name}' already exists at {path}"
    ok, result = run_git(["worktree", "add", str(path), "-b", f"wt/{name}", "HEAD"])
    if not ok:
        return f"Git error: {result}"
    if task_id:
        bind_task_to_worktree(task_id, name)
    log_event("create", name, task_id)
    print(f"  \033[33m[worktree] created: {name} at {path}\033[0m")
    return f"Worktree '{name}' created at {path}"


def bind_task_to_worktree(task_id: str, worktree_name: str):
    """将任务绑定到 worktree——仅写入 worktree 字段，不改状态。"""
    task = load_task(task_id)
    task.worktree = worktree_name
    save_task(task)


def _count_worktree_changes(path: Path) -> tuple[int, int]:
    """统计 worktree 中未提交文件数和未推送提交数。"""
    try:
        r1 = subprocess.run(["git", "status", "--porcelain"],
                            cwd=path, capture_output=True, text=True, timeout=10)
        files = len([l for l in r1.stdout.strip().splitlines() if l.strip()])
        r2 = subprocess.run(["git", "log", "@{push}..HEAD", "--oneline"],
                            cwd=path, capture_output=True, text=True, timeout=10)
        commits = len([l for l in r2.stdout.strip().splitlines() if l.strip()])
        return files, commits
    except Exception:
        return -1, -1


def remove_worktree(name: str, discard_changes: bool = False) -> str:
    """删除 worktree：安全检查 → git worktree remove → 清理分支。"""
    err = validate_worktree_name(name)
    if err:
        return err
    path = WORKTREES_DIR / name
    if not path.exists():
        return f"Worktree '{name}' not found"
    if not discard_changes:
        files, commits = _count_worktree_changes(path)
        if files < 0:
            return "Cannot verify status. Use discard_changes=true to force."
        if files > 0 or commits > 0:
            return (f"Worktree '{name}' has {files} file(s), {commits} commit(s). "
                    "Use discard_changes=true or keep_worktree.")
    ok1, _ = run_git(["worktree", "remove", str(path), "--force"])
    if not ok1:
        return f"Failed to remove worktree '{name}'"
    run_git(["branch", "-D", f"wt/{name}"])
    log_event("remove", name)
    print(f"  \033[33m[worktree] removed: {name}\033[0m")
    return f"Worktree '{name}' removed"


def keep_worktree(name: str) -> str:
    """保留 worktree 供人工审查，分支 wt/{name} 保留。"""
    err = validate_worktree_name(name)
    if err:
        return err
    log_event("keep", name)
    return f"Worktree '{name}' kept for review (branch: wt/{name})"


# ── s07: Skill Loading（技能加载）──
SKILL_REGISTRY: dict[str, dict] = {} # 技能注册表：技能名称 → {name, description, content}


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 YAML frontmatter。返回 (元数据字典, 正文)。"""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, parts[2].strip()


def scan_skills():
    """扫描 skills/ 目录，将发现的技能注册到 SKILL_REGISTRY。"""
    SKILL_REGISTRY.clear()
    if not SKILLS_DIR.exists():
        return
    for directory in sorted(SKILLS_DIR.iterdir()):
        if not directory.is_dir():
            continue
        manifest = directory / "SKILL.md"
        if not manifest.exists():
            continue
        raw = manifest.read_text()
        meta, _ = _parse_frontmatter(raw)
        name = meta.get("name", directory.name)
        desc = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
        SKILL_REGISTRY[name] = {
            "name": name,
            "description": desc,
            "content": raw,
        }

# 启动时扫描技能目录，构建技能注册表，供后续查询和加载。
scan_skills()


def list_skills() -> str:
    """列出所有已注册的技能名称和描述。"""
    if not SKILL_REGISTRY:
        return "(no skills found)"
    return "\n".join(
        f"- {skill['name']}: {skill['description']}"
        for skill in SKILL_REGISTRY.values())


def load_skill(name: str) -> str:
    """加载指定技能的完整内容。"""
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        available = ", ".join(SKILL_REGISTRY.keys()) or "(none)"
        return f"Skill not found: {name}. Available: {available}"
    return skill["content"]


# ── Prompt Assembly ──

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file, edit_file, glob, "
             "todo_write, task, load_skill, compact, "
             "create_task, list_tasks, get_task, claim_task, complete_task, "
             "schedule_cron, list_crons, cancel_cron, "
             "spawn_teammate, send_message, check_inbox, "
             "request_shutdown, request_plan, review_plan, "
             "create_worktree, remove_worktree, keep_worktree, "
             "connect_mcp. MCP tools are prefixed mcp__{server}__{tool}.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    # The system prompt is rebuilt each turn from live context. This is where
    # memory, skill catalog, MCP state, and active teammates become visible.
    """根据当前上下文动态组装系统提示。包含固定部分（身份、工具、工作目录）和动态部分（时间、技能目录、相关记忆、MCP 状态）。"""
    sections = [PROMPT_SECTIONS["identity"],
                PROMPT_SECTIONS["tools"],
                PROMPT_SECTIONS["workspace"]]
    sections.append(f"Current time: {datetime.now().isoformat(timespec='seconds')}") # 注入当前时间，供计划和调度使用
    sections.append("Skills catalog:\n" + list_skills() +
                    "\nUse load_skill(name) when a skill is relevant.") # 注入技能目录，供模型查询和加载技能使用
    if context.get("memories"): # 注入相关记忆，供模型参考。实际应用中可根据上下文相关性动态选择注入哪些记忆。
        sections.append(f"Relevant memories:\n{context['memories']}")
    mcp_names = list(mcp_clients.keys())
    if mcp_names: # 注入 MCP 连接状态，供模型了解当前可用的 MCP 服务器和工具。
        sections.append(f"Connected MCP servers: {', '.join(mcp_names)}")
    return "\n\n".join(sections)


# ── Basic Tools ──

def safe_path(p: str, cwd: Path = None) -> Path:
    """路径安全校验。文件工具只能访问工作目录内的路径。"""
    base = cwd or WORKDIR
    path = (base / p).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str, cwd: Path = None,
             run_in_background: bool = False) -> str:
    """执行 shell 命令。run_in_background 由 dispatcher 消费，直接调用忽略。"""
    try:
        r = subprocess.run(command, shell=True, cwd=cwd or WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int | None = None,
             offset: int = 0, cwd: Path = None) -> str:
    """读取文件内容。支持 limit/offset 分页，cwd 支持 worktree。"""
    try:
        lines = safe_path(path, cwd).read_text().splitlines()
        offset = max(int(offset or 0), 0)
        limit = int(limit) if limit is not None else None
        lines = lines[offset:]
        if limit is not None and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str, cwd: Path = None) -> str:
    """写入文件内容，自动创建父目录。cwd 支持 worktree。"""
    try:
        fp = safe_path(path, cwd)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str,
             cwd: Path = None) -> str:
    """精确替换文件中的文本一次。"""
    try:
        fp = safe_path(path, cwd)
        text = fp.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        fp.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str, cwd: Path = None) -> str:
    """按 glob 模式查找文件。"""
    import glob as g
    try:
        base = cwd or WORKDIR
        results = []
        for match in g.glob(pattern, root_dir=base):
            if (base / match).resolve().is_relative_to(base):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


def call_tool_handler(handler, args: dict, name: str) -> str:
    """统一工具 handler 调用入口：查找 → 执行 → 错误处理。"""
    if not handler:
        return f"Unknown: {name}"
    try:
        return handler(**(args or {}))
    except TypeError as e:
        return f"Error: {e}"


def run_todo_write(todos: list) -> str:
    """更新当前会话的内存 todo 列表（s05）。验证每个 todo 的结构。"""
    global CURRENT_TODOS
    for i, todo in enumerate(todos):
        if "content" not in todo or "status" not in todo:
            return f"Error: todos[{i}] missing 'content' or 'status'"
        if todo["status"] not in ("pending", "in_progress", "completed"):
            return f"Error: todos[{i}] has invalid status '{todo['status']}'"
    CURRENT_TODOS = todos
    print(f"  \033[33m[todo] updated {len(CURRENT_TODOS)} item(s)\033[0m")
    return f"Updated {len(CURRENT_TODOS)} todos"


# ── s15: MessageBus（消息总线）──
MAILBOX_DIR = WORKDIR / ".mailboxes"
MAILBOX_DIR.mkdir(exist_ok=True)


class MessageBus:
    """基于文件的消息总线：send=append JSONL, read_inbox=read+unlink。"""

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message", metadata: dict = None):
        """发送消息：追加 JSON 行到目标收件箱。"""
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time(), "metadata": metadata or {}}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a") as f:
            f.write(json.dumps(msg) + "\n")
        terminal_print(f"  \033[33m[bus] {from_agent} → {to_agent}: "
                       f"({msg_type}) {content[:50]}\033[0m")

    def read_inbox(self, agent: str) -> list[dict]:
        """读取收件箱全部消息并删除文件（消费语义）。"""
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        msgs = [json.loads(line) for line in inbox.read_text().splitlines()
                if line.strip()]
        inbox.unlink()
        return msgs


BUS = MessageBus() # 全局消息总线实例，供所有组件使用
active_teammates: dict[str, bool] = {} # 活跃队友列表，键是队友名称


@dataclass
class ProtocolState:
    """协议请求状态。追踪 request_id 的完整生命周期。"""
    request_id: str # 唯一协议请求 ID
    type: str # 协议类型，如 "shutdown"、"plan_approval"
    sender: str # 请求发起者
    target: str # 请求目标者
    status: str # 当前状态，如 "pending"、"approved"、"rejected"
    payload: str # 协议相关的额外信息，如计划内容
    created_at: float = field(default_factory=time.time)


pending_requests: dict[str, ProtocolState] = {} # 全局协议请求状态字典，键是 request_id，值是 ProtocolState 对象


def new_request_id() -> str:
    """生成唯一协议请求 ID。"""
    return f"req_{random.randint(0, 999999):06d}"


def match_response(response_type: str, request_id: str, approve: bool):
    """协议响应匹配：通过 request_id 关联响应到原始请求。"""
    state = pending_requests.get(request_id)
    if not state:
        return
    if state.type == "shutdown" and response_type != "shutdown_response":
        return
    if state.type == "plan_approval" and response_type != "plan_approval_response":
        return
    state.status = "approved" if approve else "rejected"


def consume_lead_inbox(route_protocol=True) -> list[dict]:
    """统一消费 Lead 收件箱：先路由协议响应，再返回全部消息。"""
    msgs = BUS.read_inbox("lead")
    if route_protocol: 
        for msg in msgs:
            meta = msg.get("metadata", {})
            req_id = meta.get("request_id", "")
            msg_type = msg.get("type", "")
            if req_id and msg_type.endswith("_response"): # 仅路由协议响应消息，其他消息正常返回给 Lead 处理。
                match_response(msg_type, req_id, meta.get("approve", False))
    return msgs


# ── Autonomous Agent ──

IDLE_POLL_INTERVAL = 5 # 轮询间隔秒数，Agent 在 IDLE 阶段每隔这个时间检查一次收件箱和任务板
IDLE_TIMEOUT = 60 # IDLE 超时时间，单位秒。Agent 在 IDLE 阶段等待这个时间，如果没有收到消息或找到可认领的任务，就返回 timeout 状态


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
              name: str, role: str,
              worktree_context: dict | None = None) -> str:
    """IDLE 阶段轮询：inbox 优先 → 任务板扫描 → 60s 超时。"""
    for _ in range(IDLE_TIMEOUT // IDLE_POLL_INTERVAL):
        time.sleep(IDLE_POLL_INTERVAL)
        # 优先级 1：优先检查收件箱，处理协议请求（如 shutdown_request）并将其他消息注入提示。协议请求通过 BUS.send 直接回复，其他消息则包装成 <inbox> 注入提示供模型参考。
        inbox = BUS.read_inbox(agent_name)
        if inbox:
            for msg in inbox:
                if msg.get("type") == "shutdown_request":
                    req_id = msg.get("metadata", {}).get("request_id", "")
                    BUS.send(name, "lead", "Shutting down.",
                             "shutdown_response",
                             {"request_id": req_id, "approve": True})
                    return "shutdown"
            messages.append({"role": "user",
                "content": "<inbox>" + json.dumps(inbox) + "</inbox>"})
            return "work"
        # 优先级 2：扫描任务板，寻找 pending + 无 owner + 依赖满足的任务，自动认领并注入提示。认领成功后，如果任务绑定了 worktree，则将 worktree 路径注入上下文供工具使用。
        unclaimed = scan_unclaimed_tasks()
        if unclaimed:
            task_data = unclaimed[0]
            result = claim_task(task_data["id"], agent_name)
            if "Claimed" in result:
                wt_info = ""
                if task_data.get("worktree"):
                    wt_path = WORKTREES_DIR / task_data["worktree"]
                    wt_info = f"\nWork directory: {wt_path}"
                    if worktree_context is not None:
                        worktree_context["path"] = str(wt_path)
                messages.append({"role": "user",
                    "content": f"<auto-claimed>Task {task_data['id']}: "
                               f"{task_data['subject']}{wt_info}</auto-claimed>"})
                return "work"
    return "timeout" # 超时返回，Lead 可选择继续等待或让 Agent 休眠。


# ── Teammate Thread ──

def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    """生成自治队友 Agent 线程。s20 新增 plan approval 门禁——submit_plan 后对模型停止工具执行。"""
    if name in active_teammates:
        return f"Teammate '{name}' already exists"

    # 计划审批是一道真正的关卡：提交计划后，团队成员便停止行动
    # 直到领导发送计划审批回复，持续执行模型/工具步骤。
    protocol_ctx = {"waiting_plan": None}
    system = (f"You are '{name}', a {role}. "
              f"Use tools to complete tasks. "
              f"If a task has a worktree, work in that directory.")

    def handle_inbox_message(name: str, msg: dict, messages: list):
        """协议消息分发器。按消息类型路由。返回 True 表示应关机，否则继续工作。"""
        msg_type = msg.get("type", "message")
        meta = msg.get("metadata", {})
        req_id = meta.get("request_id", "")
        if msg_type == "shutdown_request":
            BUS.send(name, "lead", "Shutting down.",
                     "shutdown_response",
                     {"request_id": req_id, "approve": True})
            return True
        if msg_type == "plan_approval_response":
            approve = meta.get("approve", False)
            if req_id == protocol_ctx["waiting_plan"]:
                protocol_ctx["waiting_plan"] = None
            messages.append({"role": "user",
                "content": "[Plan approved]" if approve
                           else f"[Plan rejected] {msg['content']}"})
        return False

    def run():
        # 核心：wt_ctx 追踪当前 worktree 路径，作为 bash/read/write 的 cwd。claim_task 时如果任务绑定了 worktree 就设置路径，complete_task 时清除路径。工具调用时使用这个路径确保在正确的目录上下文中操作。
        wt_ctx = {"path": None}

        def _wt_cwd():
            """获取当前 worktree 的工作目录路径。"""
            # 一旦拥有工作台的任务被认领，所有队友都会归档工具
            # 透明地运行在隔离的目录中，避免相互干扰。工具函数通过这个接口获取当前工作目录路径，确保在正确的上下文中执行。
            p = wt_ctx["path"]
            return Path(p) if p else None

        def _run_bash(command: str) -> str:
            return run_bash(command, cwd=_wt_cwd())

        def _run_read(path: str) -> str:
            return run_read(path, cwd=_wt_cwd())

        def _run_write(path: str, content: str) -> str:
            return run_write(path, content, cwd=_wt_cwd())

        def _run_list_tasks():
            tasks = list_tasks()
            if not tasks:
                return "No tasks."
            return "\n".join(
                f"  {t.id}: {t.subject} [{t.status}]"
                + (f" (wt:{t.worktree})" if t.worktree else "")
                for t in tasks)

        def _run_claim_task(task_id: str):
            result = claim_task(task_id, owner=name)
            if "Claimed" in result:
                task = load_task(task_id)
                wt_ctx["path"] = (str(WORKTREES_DIR / task.worktree)
                                  if task.worktree else None)
            return result

        def _run_complete_task(task_id: str):
            result = complete_task(task_id)
            wt_ctx["path"] = None
            return result

        messages = [{"role": "user", "content": prompt}]
        sub_tools = [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object",
                              "properties": {"command": {"type": "string"}},
                              "required": ["command"]}},
            {"name": "read_file", "description": "Read file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"},
                                             "limit": {"type": "integer"},
                                             "offset": {"type": "integer"}},
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
             "description": "List all tasks.",
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
            "list_tasks": _run_list_tasks,
            "claim_task": _run_claim_task,
            "complete_task": _run_complete_task,
        }

        # 外层循环：WORK → IDLE 交替
        while True:
            if len(messages) <= 3:
                messages.insert(0, {"role": "user",
                    "content": f"<identity>You are '{name}', role: {role}. "
                               f"Continue your work.</identity>"})
            should_shutdown = False
            # WORK 阶段：最多 10 轮
            for _ in range(10):
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    stopped = handle_inbox_message(name, msg, messages)
                    if stopped:
                        should_shutdown = True
                        break
                if should_shutdown:
                    break
                if protocol_ctx["waiting_plan"]:
                    # 在审批门关闭时，仅对协议回复进行轮询；不要让模型继续执行任务
                    time.sleep(IDLE_POLL_INTERVAL)
                    continue
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
                if not has_tool_use(response.content): # 检查是否包含工具调用
                    break
                results = []
                for block in response.content:
                    if block.type == "tool_use":
                        if block.name == "submit_plan": # 特殊工具：提交计划后进入等待审批状态，直到收到审批回复。
                            output = _teammate_submit_plan(
                                name, block.input.get("plan", ""))
                            match = re.search(r"\((req_\d+)\)", output)
                            protocol_ctx["waiting_plan"] = (
                                match.group(1) if match else output)
                        else:
                            handler = sub_handlers.get(block.name)
                            output = call_tool_handler(handler, block.input,
                                                       block.name)
                        results.append({"type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": str(output)})
                        if protocol_ctx["waiting_plan"]:
                            # 忽略同一模型响应中后续的tool_use块；它们属于批准之后，而不是之前。
                            break
                messages.append({"role": "user", "content": results})
                if protocol_ctx["waiting_plan"]:
                    break
            if should_shutdown:
                break
            if protocol_ctx["waiting_plan"]:
                continue
            # IDLE 阶段：轮询收件箱和任务板，等待消息或可认领任务，60s 超时返回。
            idle_result = idle_poll(name, messages, name, role, wt_ctx)
            if idle_result in ("shutdown", "timeout"):
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
        BUS.send(name, "lead", summary, "result") # 发送最终结果给 Lead
        active_teammates.pop(name, None)  # 清理活跃队友列表

    active_teammates[name] = True # 标记队友为活跃状态
    threading.Thread(target=run, daemon=True).start()
    return f"Teammate '{name}' spawned as {role}"


def _teammate_submit_plan(from_name: str, plan: str) -> str:
    """队友提交计划给 Lead 审批。返回协议请求 ID。"""
    req_id = new_request_id()
    pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="plan_approval",
        sender=from_name, target="lead",
        status="pending", payload=plan)
    BUS.send(from_name, "lead", plan,
             "plan_approval_request",
             {"request_id": req_id})
    return f"Plan submitted ({req_id})"


# ── Lead Protocol Tools ──

def run_request_shutdown(teammate: str) -> str:
    """Lead 发起关机协议请求。"""
    req_id = new_request_id()
    pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="shutdown",
        sender="lead", target=teammate,
        status="pending", payload="")
    BUS.send("lead", teammate, "Shut down.", "shutdown_request",
             {"request_id": req_id})
    return f"Shutdown request sent to {teammate}"


def run_request_plan(teammate: str, task: str) -> str:
    """Lead 请求队友提交计划。"""
    BUS.send("lead", teammate, f"Submit plan for: {task}", "message")
    return f"Asked {teammate} to submit a plan"


def run_review_plan(request_id: str, approve: bool,
                    feedback: str = "") -> str:
    """Lead 审批计划请求。通过 request_id 定位请求状态，发送审批回复。"""
    state = pending_requests.get(request_id)
    if not state:
        return f"Request {request_id} not found"
    state.status = "approved" if approve else "rejected"
    BUS.send("lead", state.sender,
             feedback or ("Approved" if approve else "Rejected"),
             "plan_approval_response",
             {"request_id": request_id, "approve": approve})
    return f"Plan {'approved' if approve else 'rejected'}"


# ── s03+s04: Hooks + Permission Pipeline（权限+钩子）──
HOOKS = {"UserPromptSubmit": [], "PreToolUse": [],
         "PostToolUse": [], "Stop": []}


def register_hook(event: str, callback):
    """注册钩子回调到指定事件。"""
    HOOKS[event].append(callback)


def trigger_hooks(event: str, *args):
    """触发指定事件的所有钩子。返回第一个非 None 的结果（用于打断链）。"""
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="] # 禁止命令列表
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"] # 可能具有破坏性的命令关键词，触发用户确认


def permission_hook(block):
    # 权限层在分派之前看到原始tool_use。它可以拒绝、询问用户或允许执行继续。
    """ 权限钩子：在工具调用前检查命令或路径是否合法。对于 bash 命令，检查是否包含禁止的关键词；
    对于文件操作，检查路径是否安全；对于 MCP 工具，提示用户确认。返回字符串表示拒绝理由，返回 None 表示允许执行。"""
    if block.name == "bash":
        command = block.input.get("command", "")
        for pattern in DENY_LIST:
            if pattern in command:
                return f"Permission denied: '{pattern}' is on the deny list"
        if any(token in command for token in DESTRUCTIVE):
            print(f"\n\033[33m[permission] destructive command\033[0m")
            print(f"  {command}")
            choice = input("  Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    if block.name in ("write_file", "edit_file"):
        path = block.input.get("path", "")
        try:
            safe_path(path)
        except Exception:
            return f"Permission denied: path escapes workspace: {path}"
    if block.name.startswith("mcp__") and "deploy" in block.name:
        print(f"\n\033[33m[permission] MCP destructive-looking tool: {block.name}\033[0m")
        choice = input("  Allow? [y/N] ").strip().lower()
        if choice not in ("y", "yes"):
            return "Permission denied by user"
    return None


def log_hook(block):
    """日志钩子：记录工具调用信息。"""
    print(f"\033[90m[HOOK] {block.name}\033[0m")
    return None


def large_output_hook(block, output):
    """大输出钩子：如果工具输出超过 100k 字符，打印警告。"""
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] large output from {block.name}: "
              f"{len(str(output))} chars\033[0m")
    return None


def user_prompt_hook(query: str):
    """用户提示钩子：在用户提示提交时触发，记录提示内容和当前工作目录。"""
    print(f"\033[90m[HOOK] UserPromptSubmit: {WORKDIR}\033[0m")
    return None


def stop_hook(messages: list):
    """停止钩子：在 Agent 停止时触发，统计本次会话中工具调用的总数。"""
    tool_count = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            tool_count += sum(1 for item in content
                              if isinstance(item, dict)
                              and item.get("type") == "tool_result")
    print(f"\033[90m[HOOK] Stop: {tool_count} tool result(s)\033[0m")
    return None

# ── 注册钩子：将回调绑定到对应事件 ─────────────────
register_hook("UserPromptSubmit", user_prompt_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", stop_hook)


# ── s06: Subagent Tool（子代理）──
# 子代理工具允许模型在独立上下文中执行一系列步骤，适合复杂任务的分解和执行。
# 子代理有自己的系统提示、工具集和最多 30 轮的对话限制。
# 子代理的输出最终会被提取为一个简洁的文本摘要返回给主代理。
SUB_SYSTEM = (
    f"You are a coding subagent at {WORKDIR}. "
    "Complete the task, then return a concise final summary. "
    "Do not spawn more agents."
)


SUB_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"},
                                     "offset": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
]

# 子代理工具的处理函数
SUB_HANDLERS = {
    "bash": run_bash, "read_file": run_read,
    "write_file": run_write, "edit_file": run_edit,
    "glob": run_glob,
}


def extract_text(content) -> str:
    """从助手回复内容中提取纯文本（type=text 的 block）。"""
    if not isinstance(content, list):
        return str(content)
    return "\n".join(
        getattr(block, "text", "")
        for block in content
        if getattr(block, "type", None) == "text").strip()


def has_tool_use(content) -> bool:
    """检查响应中是否包含 tool_use block。"""
    return any(getattr(block, "type", None) == "tool_use"
               for block in content)


def spawn_subagent(description: str) -> str:
    """启动子 Agent（s06）：独立上下文，最多 30 轮，返回最终文本摘要。"""
    messages = [{"role": "user", "content": description}]
    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUB_SYSTEM, messages=messages,
            tools=SUB_TOOLS, max_tokens=8000)
        messages.append({"role": "assistant", "content": response.content})
        if not has_tool_use(response.content):
            break
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                output = str(blocked)
            else:
                handler = SUB_HANDLERS.get(block.name)
                output = call_tool_handler(handler, block.input, block.name)
                trigger_hooks("PostToolUse", block, output)
            results.append({"type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(output)})
        messages.append({"role": "user", "content": results})
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            text = extract_text(msg["content"])
            if text:
                return text
    return "Subagent finished without a text summary."


# ── s08: Context Compaction（四层压缩管道）──
def estimate_size(messages: list) -> int:
    """估算消息列表的 JSON 序列化大小（用于压缩判断）。"""
    return len(json.dumps(messages, default=str))


def collect_tool_results(messages: list):
    """收集消息列表中所有 tool_result block 的位置。"""
    found = []
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if msg.get("role") != "user" or not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                found.append((mi, bi, block))
    return found


def persist_large_output(tool_use_id: str, output: str) -> str:
    """超大工具输出持久化到文件，返回引用路径。L3 层。"""
    if len(output) <= PERSIST_THRESHOLD:
        return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not path.exists():
        path.write_text(output)
    return (f"<persisted-output>\nFull output: {path}\n"
            f"Preview:\n{output[:2000]}\n</persisted-output>")


def tool_result_budget(messages: list, max_bytes: int = 200_000) -> list:
    """L3 层：限制最近一条 user 消息中 tool_result 的总字节数。"""
    if not messages:
        return messages
    last = messages[-1]
    content = last.get("content")
    if last.get("role") != "user" or not isinstance(content, list):
        return messages
    blocks = [(i, b) for i, b in enumerate(content)
              if isinstance(b, dict) and b.get("type") == "tool_result"]
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_bytes:
        return messages
    for _, block in sorted(blocks,
                           key=lambda pair: len(str(pair[1].get("content", ""))),
                           reverse=True):
        if total <= max_bytes:
            break
        text = str(block.get("content", ""))
        block["content"] = persist_large_output(
            block.get("tool_use_id", "unknown"), text)
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    return messages


def snip_compact(messages: list, max_messages: int = 50) -> list:
    """L1 层：消息数超限时保留头尾，中间替换为 snipped 标记。"""
    if len(messages) <= max_messages:
        return messages
    keep_head, keep_tail = 3, max_messages - 3
    snipped = len(messages) - keep_head - keep_tail
    return (messages[:keep_head]
            + [{"role": "user", "content": f"[snipped {snipped} messages]"}]
            + messages[-keep_tail:])


def micro_compact(messages: list) -> list:
    """L2 层：压缩较早的 tool_result（只保留最近 N 个完整内容）。"""
    tool_results = collect_tool_results(messages)
    if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS:
        return messages
    for _, _, block in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
        if len(str(block.get("content", ""))) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages


def write_transcript(messages: list) -> Path:
    """将完整对话写入 .transcripts/ 目录的 JSONL 文件。"""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    return path


def summarize_history(messages: list) -> str:
    """调用 LLM 生成对话摘要（L4 auto compact）。"""
    conversation = json.dumps(messages, default=str)[:80000]
    # 提示词：让模型总结对话，保留当前目标、关键发现、已修改文件、剩余工作和用户约束等重要信息，以便继续工作。
    prompt = ("Summarize this coding-agent conversation so work can continue. "
              "Preserve current goal, key findings, changed files, remaining work, "
              "and user constraints.\n\n" + conversation)
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000)
    return extract_text(response.content) or "(empty summary)"


def compact_history(messages: list) -> list:
    """L4 主动压缩：写入 transcript → LLM 摘要 → 返回压缩后的消息列表。"""
    transcript = write_transcript(messages)
    print(f"  \033[36m[compact] transcript saved: {transcript}\033[0m")
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


def reactive_compact(messages: list) -> list:
    """被动压缩：仅在检测到 prompt-too-long 错误时触发。写入 transcript → LLM 摘要 → 返回压缩后的消息列表。"""
    transcript = write_transcript(messages)
    print(f"  \033[31m[reactive compact] transcript saved: {transcript}\033[0m")
    try:
        summary = summarize_history(messages)
    except Exception:
        summary = "Earlier conversation was trimmed after a prompt-too-long error."
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"},
            *messages[-5:]]


# ── s11: Error Recovery（错误恢复）──

class RecoveryState:
    """追踪循环中的恢复尝试状态。
    字段说明：
    - has_escalated: 是否已从 8K 升级到 64K max_tokens
    - recovery_count: 续写提示已发次数（最多 MAX_RECOVERY_RETRIES）
    - consecutive_529: 连续过载错误计数（达阈值时切换模型）
    - has_attempted_reactive_compact: 是否已尝试过紧急压缩
    - current_model: 当前使用的模型 ID"""
    def __init__(self):
        self.has_escalated = False 
        self.recovery_count = 0
        self.consecutive_529 = 0
        self.has_attempted_reactive_compact = False
        self.current_model = PRIMARY_MODEL


def retry_delay(attempt: int) -> float:
    """指数退避 + 随机抖动。API 的 Retry-After 头优先。"""
    base = min(BASE_DELAY_MS * (2 ** attempt), 32000) / 1000
    return base + random.uniform(0, base * 0.25)


def with_retry(fn, state: RecoveryState):
    """Path 3: 瞬态错误（429/529）的指数退避包装器。
        429（速率限制）→ 纯退避；
        529（过载）→ 退避 + 连续 3 次后切换备选模型。
        非瞬态异常不做退避，直接向上抛出给外层 handler。"""
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_529 = 0
            return result
        except Exception as e:
            name = type(e).__name__.lower()
            msg = str(e).lower()
            if "ratelimit" in name or "429" in msg:
                delay = retry_delay(attempt)
                print(f"  \033[33m[429] retry {attempt + 1}/{MAX_RETRIES} "
                      f"after {delay:.1f}s\033[0m")
                time.sleep(delay)
                continue
            if "overloaded" in name or "529" in msg or "overloaded" in msg:
                state.consecutive_529 += 1
                if state.consecutive_529 >= MAX_CONSECUTIVE_529 and FALLBACK_MODEL:
                    state.current_model = FALLBACK_MODEL
                    state.consecutive_529 = 0
                    print(f"  \033[31m[529] switching to {FALLBACK_MODEL}\033[0m")
                delay = retry_delay(attempt)
                print(f"  \033[33m[529] retry {attempt + 1}/{MAX_RETRIES} "
                      f"after {delay:.1f}s\033[0m")
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded")


def is_prompt_too_long_error(e: Exception) -> bool:
    """判断 API 异常是否为 prompt/上下文过长错误。"""
    msg = str(e).lower()
    return (("prompt" in msg and "long" in msg)
            or "context_length_exceeded" in msg
            or "max_context_window" in msg)


# ── s13: Background Tasks（后台任务）──
_bg_counter = 0 # 全局计数器生成唯一的后台任务 ID
background_tasks: dict[str, dict] = {} # bg_id → {tool_use_id, command, status} 生命周期跟踪
background_results: dict[str, str] = {} # bg_id → output 线程安全结果存储
background_lock = threading.Lock() # 互斥锁保护 background_tasks 和 background_results


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """启发式判断 bash 命令是否为慢操作（含 install/build/test 等关键词）。"""
    if tool_name != "bash":
        return False
    command = tool_input.get("command", "").lower()
    slow_keywords = ["install", "build", "test", "deploy", "compile",
                     "docker build", "pip install", "npm install",
                     "cargo build", "pytest", "make"]
    return any(keyword in command for keyword in slow_keywords)


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """判断是否应后台执行：run_in_background 显式声明 或 慢操作启发式。"""
    if tool_name != "bash":
        return False
    return bool(tool_input.get("run_in_background")) or is_slow_operation(tool_name, tool_input)


def start_background_task(block, handlers: dict) -> str:
    """将慢操作分发到守护线程后台执行，返回 bg_id。"""
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    command = block.input.get("command", block.name)

    def worker():
        handler = handlers.get(block.name)
        result = call_tool_handler(handler, block.input, block.name)
        trigger_hooks("PostToolUse", block, result)
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = str(result)

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": command,
            "status": "running",
        }
    threading.Thread(target=worker, daemon=True).start()
    print(f"  \033[33m[background] {bg_id}: {str(command)[:60]}\033[0m")
    return bg_id


def collect_background_results() -> list[str]:
    """收集已完成后台任务的结果，格式化为 <task_notification> 消息列表。"""
    with background_lock:
        ready = [bg_id for bg_id, task in background_tasks.items()
                 if task["status"] == "completed"]
    notifications = []
    for bg_id in ready:
        with background_lock:
            task = background_tasks.pop(bg_id)
            output = background_results.pop(bg_id, "")
        summary = output[:200] if len(output) > 200 else output
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>")
    return notifications


# ── s14: Cron Scheduler（定时调度）──
DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"


@dataclass
class CronJob:
    """定时任务数据类：结构化定时任务定义。
    - cron: 5 字段 cron 表达式（"分 时 日 月 周"）
    - prompt: 触发时注入到对话中的提示词
    - recurring: True=循环执行, False=一次性
    - durable: True=持久化到磁盘，重启后恢复"""
    id: str
    cron: str
    prompt: str
    recurring: bool
    durable: bool


scheduled_jobs: dict[str, CronJob] = {} # 注册的定时任务
cron_queue: list[CronJob] = [] # Layer 2: 已触发的任务队列（调度器写入，消费者取出）
cron_lock = threading.Lock()  # 互斥锁保护 scheduled_jobs 和 cron_queue
_last_fired: dict[str, str] = {} # 记录每个任务上次触发的时间标记，job_id → "YYYY-MM-DD HH:MM" 防止同一分钟重复触发，避免同一分钟内重复触发


def _cron_field_matches(field: str, value: int) -> bool:
    """匹配单个 cron 字段与给定值。
    支持通配符 *、步进 */N、列表 a,b,c、范围 a-b。"""
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        return any(_cron_field_matches(part.strip(), value)
                   for part in field.split(","))
    if "-" in field:
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)
    return value == int(field)


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """检查 5 字段 cron 表达式是否匹配给定时间。
    标准 cron 语义：DOM（月内第几天）和 DOW（周几）同时约束时使用 OR（任一满足即可）。
    例如 "0 0 1 * 1" = 每月 1 号 OR 每个周一 00:00 触发。
    分钟  小时  日  月  星期"""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val = (dt.weekday() + 1) % 7

    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)

    # 分钟、小时、月份必须全部匹配；日和周满足任一即可
    if not (m and h and month_ok):
        return False
    if dom == "*" and dow == "*":
        return True
    if dom == "*":
        return dow_ok
    if dow == "*":
        return dom_ok
    return dom_ok or dow_ok


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    """验证单个 cron 字段值是否在 [lo, hi] 范围内。返回错误信息或 None。"""
    if field == "*":
        return None
    if field.startswith("*/"):
        step = field[2:]
        if not step.isdigit() or int(step) <= 0:
            return f"Invalid step: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err:
                return err
        return None
    if "-" in field:
        left, right = field.split("-", 1)
        if not left.isdigit() or not right.isdigit():
            return f"Invalid range: {field}"
        a, b = int(left), int(right)
        if a < lo or a > hi or b < lo or b > hi:
            return f"Range {field} out of bounds [{lo}-{hi}]"
        if a > b:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    value = int(field)
    if value < lo or value > hi:
        return f"Value {value} out of bounds [{lo}-{hi}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    """验证 cron 表达式合法性。返回错误信息或 None（表示通过）。"""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    for field, (lo, hi), name in zip(fields, bounds, names):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


def save_durable_jobs():
    """将 durable=True 的定时任务持久化到 .scheduled_tasks.json。"""
    durable = [asdict(job) for job in scheduled_jobs.values() if job.durable]
    DURABLE_PATH.write_text(json.dumps(durable, indent=2))


def load_durable_jobs():
    """启动时从磁盘加载持久化的定时任务。跳过 cron 表达式不合法的任务。"""
    if not DURABLE_PATH.exists():
        return
    try:
        for item in json.loads(DURABLE_PATH.read_text()):
            job = CronJob(**item)
            if not validate_cron(job.cron):
                scheduled_jobs[job.id] = job
    except Exception:
        pass


def schedule_job(cron: str, prompt: str,
                 recurring: bool = True, durable: bool = True) -> CronJob | str:
    """注册新的定时任务。先验证 cron → 创建 CronJob → 加锁写入 scheduled_jobs。
    返回 CronJob 或错误字符串。"""
    err = validate_cron(cron)
    if err:
        return err
    job = CronJob(
        id=f"cron_{random.randint(0, 999999):06d}",
        cron=cron, prompt=prompt,
        recurring=recurring, durable=durable)
    with cron_lock:
        scheduled_jobs[job.id] = job
    if durable:
        save_durable_jobs()
    return job


def cancel_job(job_id: str) -> str:
    """取消定时任务。从 scheduled_jobs 中移除，durable 任务同步更新磁盘。"""
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    if not job:
        return f"Job {job_id} not found"
    if job.durable:
        save_durable_jobs()
    return f"Cancelled {job_id}"


def cron_scheduler_loop():
    """Layer 1: 独立守护线程——每秒轮询，触发匹配的定时任务。
    - minute_marker 防止同一分钟内重复触发（每天的第 2+ 天也不会漏掉）
    - 单个任务的异常被捕获，防止一个坏任务拖垮整个调度器线程
    - 一次性任务（recurring=False）触发后自动移除"""
    while True:
        time.sleep(1)
        now = datetime.now()
        marker = now.strftime("%Y-%m-%d %H:%M")
        with cron_lock:
            for job in list(scheduled_jobs.values()):
                try:
                    # 只有当 cron 表达式匹配当前时间且上次触发时间标记不同（防止同一分钟内重复触发）时才将任务加入 cron_queue
                    if cron_matches(job.cron, now) and _last_fired.get(job.id) != marker:
                        cron_queue.append(job)
                        _last_fired[job.id] = marker
                        if not job.recurring:
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                save_durable_jobs()
                except Exception as e:
                    print(f"  \033[31m[cron error] {job.id}: {e}\033[0m")


def consume_cron_queue() -> list[CronJob]:
    """Layer 4: 消费 cron_queue 中已触发的任务（由 agent_loop 调用）。
    加锁取出全部任务后清空队列，避免长时间持锁。"""
    with cron_lock:
        fired = list(cron_queue)
        cron_queue.clear()
    return fired


def run_schedule_cron(cron: str, prompt: str,
                      recurring: bool = True, durable: bool = True) -> str:
    """注册定时任务：验证 cron → 创建 CronJob → 加入调度。"""
    result = schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: '{cron}' -> {prompt}"


def run_list_crons() -> str:
    """列出所有已注册的定时任务，含 cron 表达式和标签（循环/一次性, 持久/会话）。"""
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs."
    return "\n".join(
        f"  {job.id}: '{job.cron}' -> {job.prompt[:40]} "
        f"[{'recurring' if job.recurring else 'one-shot'}, "
        f"{'durable' if job.durable else 'session'}]"
        for job in jobs)


def run_cancel_cron(job_id: str) -> str:
    """取消定时任务：从 scheduled_jobs 移除（durable 任务同步更新磁盘）。"""
    return cancel_job(job_id)

# 启动时加载持久化任务，然后启动调度器守护线程
load_durable_jobs()
threading.Thread(target=cron_scheduler_loop, daemon=True).start() # 启动 cron 调度器线程


# ── s19: MCP System（外部插件）──
class MCPClient:
    """MCP 客户端：发现并调用外部 MCP 服务器的工具（教学版 mock）。
    真实 MCP 协议通过 JSON-RPC 通信，教学版用内存 handler 字典模拟。"""

    def __init__(self, name: str):
        self.name = name
        self.tools: list[dict] = [] # 工具定义列表（schema）
        self._handlers: dict[str, callable] = {} # 工具名 → handler 函数

    def register(self, tool_defs: list[dict],
                 handlers: dict[str, callable]):
        """注册工具定义和对应的 handler 函数。"""
        self.tools = tool_defs
        self._handlers = handlers

    def call_tool(self, tool_name: str, args: dict) -> str:
        """调用 MCP 工具：handler 查找 → 执行 → 错误处理。"""
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"MCP error: unknown tool '{tool_name}'"
        try:
            return handler(**args)
        except Exception as e:
            return f"MCP error: {e}"


mcp_clients: dict[str, MCPClient] = {} # 已连接的 MCP 服务器注册表

_DISALLOWED_CHARS = re.compile(r'[^a-zA-Z0-9_-]') # 非法字符正则


def normalize_mcp_name(name: str) -> str:
    """标准化 MCP 名称：将非法字符（非字母数字下划线横线）替换为下划线。
    目的：保证 mcp__{server}__{tool} 前缀格式不含特殊字符。"""
    return _DISALLOWED_CHARS.sub('_', name)


def _mock_server_docs():
    """Mock MCP 服务器 "docs"：提供 search 和 get_version 两个工具。"""
    client = MCPClient("docs")
    client.register(
        tool_defs=[
            {"name": "search", "description": "Search documentation. (readOnly)",
             "inputSchema": {"type": "object",
                             "properties": {"query": {"type": "string"}},
                             "required": ["query"]}},
            {"name": "get_version", "description": "Get API version. (readOnly)",
             "inputSchema": {"type": "object", "properties": {},
                             "required": []}},
        ],
        handlers={
            "search": lambda query: f"[docs] Found 3 results for '{query}'",
            "get_version": lambda: "[docs] API v2.1.0",
        })
    return client


def _mock_server_deploy():
    """Mock MCP 服务器 "deploy"：提供 trigger（破坏性）和 status（只读）工具。"""
    client = MCPClient("deploy")
    client.register(
        tool_defs=[
            {"name": "trigger",
             "description": "Trigger a deployment. (destructive — requires approval in real CC)",
             "inputSchema": {"type": "object",
                             "properties": {"service": {"type": "string"}},
                             "required": ["service"]}},
            {"name": "status", "description": "Check deployment status. (readOnly)",
             "inputSchema": {"type": "object",
                             "properties": {"service": {"type": "string"}},
                             "required": ["service"]}},
        ],
        handlers={
            "trigger": lambda service: f"[deploy] Triggered: {service}",
            "status": lambda service: f"[deploy] {service}: running (v1.4.2)",
        })
    return client

# 可用 MCP 服务器注册表（教学版）
MOCK_SERVERS = {
    "docs": _mock_server_docs,
    "deploy": _mock_server_deploy,
}


def connect_mcp(name: str) -> str:
    """连接 MCP 服务器：查 MOCK_SERVERS → 创建客户端 → 发现工具。"""
    if name in mcp_clients:
        return f"MCP server '{name}' already connected"
    factory = MOCK_SERVERS.get(name)
    if not factory:
        available = ", ".join(MOCK_SERVERS.keys())
        return f"Unknown server '{name}'. Available: {available}"
    mcp_client = factory() # 创建 MCPClient 实例并注册工具
    mcp_clients[name] = mcp_client
    tool_names = [t["name"] for t in mcp_client.tools]
    print(f"  \033[31m[mcp] connected: {name} → {tool_names}\033[0m")
    return (f"Connected to MCP server '{name}'. "
            f"Discovered {len(mcp_client.tools)} tools: {', '.join(tool_names)}")


def assemble_tool_pool() -> tuple[list[dict], dict]:
    """动态组装工具池：内置工具 + 所有已连接 MCP 服务器的工具。
    命名规则：mcp__{safe_server}__{safe_tool}，利用 __ 作为命名空间分隔符。"""
    tools = list(BUILTIN_TOOLS) # 内置工具定义列表（schema）
    handlers = dict(BUILTIN_HANDLERS)
    for server_name, mcp_client in mcp_clients.items(): # 外层循环：遍历已连接的 MCP 服务器
        safe_server = normalize_mcp_name(server_name)
        for tool_def in mcp_client.tools: # 内层循环：遍历每个服务器的工具定义
            safe_tool = normalize_mcp_name(tool_def["name"])
            prefixed = f"mcp__{safe_server}__{safe_tool}"
            tools.append({
                "name": prefixed,
                "description": tool_def.get("description", ""),
                "input_schema": tool_def.get("inputSchema", {}),
            })
            handlers[prefixed] = (
                lambda *, c=mcp_client, t=tool_def["name"], **kw: c.call_tool(t, kw))
    return tools, handlers


# ── Lead Worktree Tools ──

def run_create_worktree(name: str, task_id: str = "") -> str:
    """创建新的工作树。可选地关联到一个任务 ID（仅记录，不强制）。"""
    return create_worktree(name, task_id)

def run_remove_worktree(name: str, discard_changes: bool = False) -> str:
    """移除工作树。可选择是否丢弃未提交的更改（慎用）。"""
    return remove_worktree(name, discard_changes)

def run_keep_worktree(name: str) -> str:
    """保留工作树"""
    return keep_worktree(name)


# ── Basic tool handlers ──

def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    """创建新任务，可指定依赖关系。"""
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
    return f"Created {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    """列出所有任务。"""
    tasks = list_tasks()
    if not tasks:
        return "No tasks."
    return "\n".join(
        f"  {t.id}: {t.subject} [{t.status}]"
        + (f" (wt:{t.worktree})" if t.worktree else "")
        for t in tasks)


def run_get_task(task_id: str) -> str:
    """获取任务详情。"""
    try:
        return get_task_json(task_id)
    except FileNotFoundError:
        return f"Error: task {task_id} not found"

def run_claim_task(task_id: str) -> str:
    """认领任务。"""
    try:
        return claim_task(task_id, owner="agent")
    except FileNotFoundError:
        return f"Error: task {task_id} not found"

def run_complete_task(task_id: str) -> str:
    """完成任务。"""
    try:
        return complete_task(task_id)
    except FileNotFoundError:
        return f"Error: task {task_id} not found"

def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    """启动队友协作线程"""
    return spawn_teammate_thread(name, role, prompt)

def run_send_message(to: str, content: str) -> str:
    """lead发送消息到队友的inbox"""
    BUS.send("lead", to, content)
    return f"Sent to {to}"

def run_check_inbox() -> str:
    """查看队友发来的消息"""
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

def run_connect_mcp(name: str) -> str:
    """连接 MCP 服务器，发现并注册其工具。"""
    return connect_mcp(name)


# ── Tool Definitions ──

# The model sees tool schemas; Python executes handlers. S20 keeps both tables
# explicit so every added capability is visible in one place.
BUILTIN_TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"},
                                     "run_in_background": {"type": "boolean"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"},
                                     "offset": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "old_text": {"type": "string"},
                                     "new_text": {"type": "string"}},
                      "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object",
                      "properties": {"pattern": {"type": "string"}},
                      "required": ["pattern"]}},
    {"name": "todo_write",
     "description": "Create and manage a task list for the current session.",
     "input_schema": {"type": "object",
                      "properties": {"todos": {"type": "array",
                          "items": {"type": "object",
                                    "properties": {
                                        "content": {"type": "string"},
                                        "status": {"type": "string",
                                                   "enum": ["pending", "in_progress", "completed"]}},
                                    "required": ["content", "status"]}}},
                      "required": ["todos"]}},
    {"name": "task",
     "description": "Launch a focused subagent. Returns only its final summary.",
     "input_schema": {"type": "object",
                      "properties": {"description": {"type": "string"}},
                      "required": ["description"]}},
    {"name": "load_skill",
     "description": "Load the full content of a skill by name.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"}},
                      "required": ["name"]}},
    {"name": "compact",
     "description": "Summarize earlier conversation and continue with compacted context.",
     "input_schema": {"type": "object",
                      "properties": {"focus": {"type": "string"}},
                      "required": []}},
    {"name": "create_task", "description": "Create a task.",
     "input_schema": {"type": "object",
                      "properties": {"subject": {"type": "string"},
                                     "description": {"type": "string"},
                                     "blockedBy": {"type": "array",
                                                   "items": {"type": "string"}}},
                      "required": ["subject"]}},
    {"name": "list_tasks", "description": "List all tasks.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_task", "description": "Get full task details.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "claim_task", "description": "Claim a pending task.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "complete_task", "description": "Complete an in-progress task.",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "schedule_cron",
     "description": ("Schedule a cron job. cron is 5-field: min hour dom "
                     "month dow. For one-shot reminders, compute the target "
                     "minute and set recurring=false."),
     "input_schema": {"type": "object",
                      "properties": {"cron": {"type": "string"},
                                     "prompt": {"type": "string"},
                                     "recurring": {"type": "boolean"},
                                     "durable": {"type": "boolean"}},
                      "required": ["cron", "prompt"]}},
    {"name": "list_crons", "description": "List registered cron jobs.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "cancel_cron", "description": "Cancel a cron job by ID.",
     "input_schema": {"type": "object",
                      "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]}},
    {"name": "spawn_teammate", "description": "Spawn an autonomous teammate.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"},
                                     "role": {"type": "string"},
                                     "prompt": {"type": "string"}},
                      "required": ["name", "role", "prompt"]}},
    {"name": "send_message", "description": "Send message to a teammate.",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["to", "content"]}},
    {"name": "check_inbox",
     "description": "Check inbox for messages and protocol responses.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "request_shutdown",
     "description": "Request a teammate to shut down.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"}},
                      "required": ["teammate"]}},
    {"name": "request_plan",
     "description": "Ask a teammate to submit a plan.",
     "input_schema": {"type": "object",
                      "properties": {"teammate": {"type": "string"},
                                     "task": {"type": "string"}},
                      "required": ["teammate", "task"]}},
    {"name": "review_plan",
     "description": "Approve or reject a submitted plan.",
     "input_schema": {"type": "object",
                      "properties": {"request_id": {"type": "string"},
                                     "approve": {"type": "boolean"},
                                     "feedback": {"type": "string"}},
                      "required": ["request_id", "approve"]}},
    {"name": "create_worktree",
     "description": "Create an isolated git worktree.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"},
                                     "task_id": {"type": "string"}},
                      "required": ["name"]}},
    {"name": "remove_worktree",
     "description": "Remove a worktree. Refuses if changes exist.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"},
                                     "discard_changes": {"type": "boolean"}},
                      "required": ["name"]}},
    {"name": "keep_worktree",
     "description": "Keep a worktree for manual review.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"}},
                      "required": ["name"]}},
    {"name": "connect_mcp",
     "description": "Connect to an MCP server (docs, deploy) and discover tools.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"}},
                      "required": ["name"]}},
]

BUILTIN_HANDLERS = { # 内置工具的 handler 映射表
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
    "todo_write": run_todo_write, "task": spawn_subagent,
    "load_skill": load_skill,
    "create_task": run_create_task, "list_tasks": run_list_tasks,
    "get_task": run_get_task,
    "claim_task": run_claim_task, "complete_task": run_complete_task,
    "schedule_cron": run_schedule_cron,
    "list_crons": run_list_crons,
    "cancel_cron": run_cancel_cron,
    "spawn_teammate": run_spawn_teammate,
    "send_message": run_send_message, "check_inbox": run_check_inbox,
    "request_shutdown": run_request_shutdown,
    "request_plan": run_request_plan, "review_plan": run_review_plan,
    "create_worktree": run_create_worktree,
    "remove_worktree": run_remove_worktree,
    "keep_worktree": run_keep_worktree,
    "connect_mcp": run_connect_mcp,
}


# ── Context ──

MEMORY_DIR = WORKDIR / ".memory" # s09: 记忆存放目录
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md" # s09: 记忆索引文件（用于快速读取最近记忆片段）


def update_context(context: dict, messages: list) -> dict:
    """从真实文件系统和内存状态推导 context（记忆+MCP+队友）。"""
    memories = ""
    if MEMORY_INDEX.exists():
        memories = MEMORY_INDEX.read_text()[:2000]
    return {
        "memories": memories,
        "connected_mcp": list(mcp_clients.keys()),
        "active_teammates": list(active_teammates.keys()),
    }


# ── Agent Loop ──

rounds_since_todo = 0 # 记录自上次 todo 更新以来的循环轮数，用于触发 todo 提醒
agent_lock = threading.Lock() # 保护 agent 内部状态，防止多线程访问冲突


def prepare_context(messages: list) -> list:
    """每个LLM回合都通过相同的上下文压缩管道进入，包含4层机制"""
    messages[:] = tool_result_budget(messages) # L3: 大结果持久化
    messages[:] = snip_compact(messages) # L1: 修剪中间消息
    messages[:] = micro_compact(messages) # L2: 旧结果占位符
    if estimate_size(messages) > CONTEXT_LIMIT:
        messages[:] = compact_history(messages) # token 仍超限 → L4 LLM 摘要压缩（1 API 调用，昂贵）
    return messages


def build_user_content(results: list[dict]) -> list[dict]:
    """工具结果和完成的后台通知都作为用户端内容返回给模型，与Tool_result反馈循环相匹配。"""
    content = list(results)
    for note in collect_background_results():
        content.append({"type": "text", "text": note})
    return content


def inject_background_notifications(messages: list):
    """从后台结果收集器获取通知并注入到消息流中，确保模型及时收到重要事件更新。"""
    notes = collect_background_results()
    if notes:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": note} for note in notes]})


def call_llm(messages: list, context: dict, tools: list,
             state: RecoveryState, max_tokens: int):
    """调用 LLM API，传入当前消息列表、上下文和工具定义。使用 with_retry 处理可能的速率限制或超时错误。"""
    system = assemble_system_prompt(context)
    return with_retry(
        lambda: client.messages.create(
            model=state.current_model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens),
        state)


def _tool_input_summary(block) -> str:
    """提取工具调用的关键参数摘要，用于终端显示。"""
    inputs = block.input
    if block.name == "bash":
        return inputs.get("command", "")
    if block.name in ("create_task", "schedule_cron"):
        return inputs.get("subject", inputs.get("cron", ""))
    if block.name in ("get_task", "claim_task", "complete_task", "cancel_cron"):
        return inputs.get("task_id", inputs.get("job_id", ""))
    if block.name in ("list_tasks", "list_crons", "check_inbox", "load_skill",
                      "compact", "list_skills"):
        return ""
    if block.name == "spawn_teammate":
        return f"{inputs.get('name', '')} ({inputs.get('role', '')})"
    if block.name == "send_message":
        return f"→ {inputs.get('to', '')}"
    if block.name in ("request_shutdown", "request_plan"):
        return f"→ {inputs.get('teammate', '')}"
    if block.name == "review_plan":
        return f"{inputs.get('request_id', '')} approve={inputs.get('approve', False)}"
    if block.name in ("create_worktree", "remove_worktree", "keep_worktree"):
        return f"{inputs.get('name', '')}"
    if block.name == "connect_mcp":
        return f"{inputs.get('name', '')}"
    if block.name == "task":
        return inputs.get("description", "")[:60]
    if block.name == "todo_write":
        return f"{len(inputs.get('todos', []))} tasks"
    if block.name.startswith("mcp__"):
        return str(inputs)
    return inputs.get("path", "")


def agent_loop(messages: list, context: dict):
    """智能体主循环。s20 是唯一一个包含全部机制的循环——
    每个 loop iteration 都经过：cron注入 → background通知 → todo提醒(s05) →
    上下文预算准备(s08 L4) → LLM调用(s11 with_retry) → PreToolUse hooks(s03/s04)
    → 工具分发(s02+s13+s19) → PostToolUse hooks → 结果组装 → 循环。"""
    global rounds_since_todo
    tools, handlers = assemble_tool_pool()
    state = RecoveryState()
    max_tokens = DEFAULT_MAX_TOKENS

    while True:
        # ── cron 定时任务注入 (s14) ──
        fired = consume_cron_queue()
        for job in fired:
            messages.append({"role": "user",
                             "content": f"[Scheduled] {job.prompt}"})
            print(f"  \033[35m[cron inject] {job.prompt[:60]}\033[0m")

        # ── 后台任务通知注入 (s13) ──
        inject_background_notifications(messages)

        # ── todo 提醒 (s05): 3 轮未更新 todo 则注入提醒 ──
        if rounds_since_todo >= 3:
            messages.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0

        # ── 上下文压缩准备 (s08): L4 压缩管道 ──
        prepare_context(messages)
        context = update_context(context, messages) # 上下文更新（记忆+MCP+队友状态）
        tools, handlers = assemble_tool_pool() # MCP 连接可能改变工具池，必须在上下文更新后重新组装

        # ── LLM 调用 (s11): with_retry 处理 429/529 ──
        try:
            response = call_llm(messages, context, tools, state, max_tokens)
        except Exception as e:
            # s11 Path 2: prompt_too_long → reactive compact (1次)
            if is_prompt_too_long_error(e) and not state.has_attempted_reactive_compact:
                messages[:] = reactive_compact(messages)
                state.has_attempted_reactive_compact = True
                continue
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": f"[Error] {type(e).__name__}: {e}"}]})
            return

        # ── s11 Path 1: max_tokens 截断 → 升级 或 续写 ──
        if response.stop_reason == "max_tokens":
            if not state.has_escalated:
                max_tokens = ESCALATED_MAX_TOKENS
                state.has_escalated = True
                print(f"  \033[33m[max_tokens] retry with {max_tokens}\033[0m")
                continue
            messages.append({"role": "assistant", "content": response.content})
            if state.recovery_count < MAX_RECOVERY_RETRIES:
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                state.recovery_count += 1
                continue
            return

        max_tokens = DEFAULT_MAX_TOKENS
        state.has_escalated = False
        messages.append({"role": "assistant", "content": response.content})
        if not has_tool_use(response.content):
            trigger_hooks("Stop", messages)  # s04: Stop hook
            return

        # ── 工具执行 (s02 dispatch + s03 permission + s04 hooks + s13 bg) ──
        results = []
        compacted_now = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            # 显示工具名和关键参数（黄色）
            print(f"\033[33m$ {block.name}: {_tool_input_summary(block)}\033[0m")

            # s08: compact 工具——模型主动请求压缩上下文
            if block.name == "compact":
                messages[:] = compact_history(messages)
                messages.append({"role": "user",
                                 "content": "[Compacted. Continue with summarized context.]"})
                compacted_now = True
                break

            # s03+s04: PreToolUse hooks — permission check + logging
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(blocked)})
                continue

            # s13: 后台任务——慢操作异步执行
            if should_run_background(block.name, block.input):
                bg_id = start_background_task(block, handlers)
                output = (f"[Background task {bg_id} started] "
                          "Result will arrive as a task_notification.")
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output})
                continue

            # s02: 工具分发——通过 handlers dict 查找执行
            handler = handlers.get(block.name)
            output = call_tool_handler(handler, block.input, block.name)
            trigger_hooks("PostToolUse", block, output)  # s04: PostToolUse hook
            # 工具结果：粗体品红标签 + 品红内容
            print(f"\033[1;35m[{block.name} -> Tool Calling Result]\033[0m \033[35m{str(output)[:300]}\033[0m")

            # s05: todo_write 重置计数，其他工具递增
            if block.name == "todo_write":
                rounds_since_todo = 0
            else:
                rounds_since_todo += 1

            results.append({"type": "tool_result",
                            "tool_use_id": block.id, "content": output})

        if compacted_now: # 如果本轮执行了 compact 工具，跳过结果组装直接进入下一轮，让模型在新的上下文基础上继续生成，而不是被旧结果干扰。
            continue

        # ── 结果组装: tool_results + background notifications (s13) ──
        messages.append({"role": "user", "content": build_user_content(results)})


def print_turn_assistants(messages: list, turn_start: int):
    """打印本轮新增的助手文本回复（蓝色）。"""
    for msg in messages[turn_start:]:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if getattr(block, "type", None) == "text":
                terminal_print(f"\033[34m{block.text}\033[0m") # 终端打印助手文本（蓝色）


def cron_autorun_loop(history: list, context: dict):
    """s14 Layer 3: cron 自动执行线程——当用户空闲时，定时触发的任务自动启动 agent_loop。
    与用户输入互斥（agent_lock），防止并发执行导致状态冲突。"""
    while True:
        time.sleep(1)
        fired = consume_cron_queue()
        if not fired:
            continue
        with agent_lock:
            turn_start = len(history)
            for job in fired:
                history.append({"role": "user",
                                "content": f"[Scheduled] {job.prompt}"})
                terminal_print(
                    f"  \033[35m[cron auto] {job.prompt[:60]}\033[0m")
            agent_loop(history, context) # 主循环agent_loop
            context.update(update_context(context, history))
            print_turn_assistants(history, turn_start)


# ── 入口：交互式 REPL ──────────────────────────────────
if __name__ == "__main__":
    CLI_ACTIVE = True
    print("s20: comprehensive agent")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []  # 消息历史
    context = update_context({}, [])  # 初始 context
    # s14: layer 3 启动 cron 自动执行线程，与用户输入互斥，防止状态冲突，空闲时自动触发定时任务执行 agent_loop
    threading.Thread(target=cron_autorun_loop,
                     args=(history, context), daemon=True).start()
    while True:
        try:
            query = input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)  # s04: UserPromptSubmit hook
        turn_start = len(history)
        history.append({"role": "user", "content": query})
        with agent_lock:  # 与 cron_autorun_loop 互斥
            agent_loop(history, context) # 主循环
            context = update_context(context, history)
            print_turn_assistants(history, turn_start)

        # s15: 队友收件箱消费 + 注入 history
        inbox = consume_lead_inbox(route_protocol=True)
        if inbox:
            def inbox_label(msg):
                req_id = msg.get("metadata", {}).get("request_id", "")
                suffix = f" req:{req_id}" if req_id else ""
                return f"{msg.get('type', 'message')}{suffix}"

            inbox_text = "\n".join(
                f"From {m['from']} [{inbox_label(m)}]: "
                f"{m['content'][:200]}" for m in inbox)
            history.append({"role": "user",
                            "content": f"[Inbox]\n{inbox_text}"})
        print()
