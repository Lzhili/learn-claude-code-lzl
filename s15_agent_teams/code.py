#!/usr/bin/env python3
"""
s15: Agent Teams — MessageBus 文件邮箱 + spawn_teammate_thread + 收件箱注入。

AI 编程智能体的多 Agent 协作：
1. MessageBus 类：基于文件的邮箱系统（.mailboxes/*.jsonl），read 即消费
2. spawn_teammate_thread: 后台线程创建队友 Agent，独立 agent_loop
3. Teammate 运行简化版 agent_loop（bash, read, write, send_message 4 个工具）
4. Lead 3 个新工具：spawn_teammate, send_message, check_inbox
5. Lead inbox: 队友消息注入到对话历史（不是只打印）
6. 教学版：队友限制 10 轮（真实 CC 用 idle loop 持续等待）

核心洞察：一个 Agent 的注意力是有限的，"重构整个后端"不是一个 Agent 能搞定的。
s15 把 Agent 变成"Lead + N 个 Teammate"：Lead 通过 spawn_teammate 派活，
Teammate 在后台线程独立工作，通过 MessageBus（文件邮箱）和 Lead 通信。
这是从"单兵作战"到"团队协作"的跃迁。

数据流：
  Lead: cron_queue → messages → prompt → LLM → TOOLS ────→ loop
                ↑                     ↓                        |
                └── inbox ← MessageBus ← teammate.send_message ←┘
  Teammate: inbox → LLM → bash/read/write/send → loop (max 10 turns)

Run / 运行: python s15_agent_teams/code.py
Need: pip install anthropic python-dotenv + .env with ANTHROPIC_API_KEY

Changes from s14 / 相对 s14 的变更:
  - MessageBus class: file-based mailboxes (.mailboxes/*.jsonl)
  - spawn_teammate_thread: creates teammate in background thread
  - Teammate runs own simplified agent_loop (bash, read, write, send_message)
  - Lead tools: spawn_teammate, send_message, check_inbox (3 new)
  - Lead inbox: teammate messages injected into history (not just printed)
  - Teaching version: teammates limited to 10 rounds (real CC uses idle loop)
"""

import os, subprocess, json, time, random, threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

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
#  FROM s10: Prompt 组装（同步，tools 段包含队友工具）
# ═══════════════════════════════════════════════════════════

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file, "
             "get_task, create_task, list_tasks, claim_task, complete_task, "
             "schedule_cron, list_crons, cancel_cron, "
             "spawn_teammate, send_message, check_inbox.",
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


def execute_tool(block) -> str:
    """执行工具调用块，返回输出结果。"""
    handler = {
        "bash": run_bash, "read_file": run_read, "write_file": run_write,
        "create_task": run_create_task, "list_tasks": run_list_tasks,
        "get_task": run_get_task, "claim_task": run_claim_task,
        "complete_task": run_complete_task,
        "schedule_cron": run_schedule_cron, "list_crons": run_list_crons,
        "cancel_cron": run_cancel_cron,
        "spawn_teammate": run_spawn_teammate,
        "send_message": run_send_message, "check_inbox": run_check_inbox,
    }.get(block.name)
    if handler:
        return handler(**block.input)
    return f"Unknown tool: {block.name}"


def start_background_task(block) -> str:
    """将工具调用分发到守护线程后台执行，返回后台任务 ID。
    主线程不等结果——先返回占位符让 agent 继续工作。"""
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    cmd = block.input.get("command", block.name)

    def worker():
        """后台工作线程：执行工具 → 结果写入共享字典（加锁保护）。"""
        result = execute_tool(block)
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
#  FROM s14: 定时调度系统（Cron Scheduler）
#  独立守护线程轮询时间 + cron_queue 解耦
# ═══════════════════════════════════════════════════════════

DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"  # 持久化文件路径


@dataclass
class CronJob:
    """定时任务数据类：结构化定时任务定义。
    - cron: 5 字段 cron 表达式（"分 时 日 月 周"）
    - prompt: 触发时注入到对话中的提示词
    - recurring: True=循环执行, False=一次性
    - durable: True=持久化到磁盘，重启后恢复"""
    id: str
    cron: str        # "0 9 * * *"
    prompt: str      # 触发时注入的消息
    recurring: bool  # 是否循环执行
    durable: bool    # 是否持久化到磁盘


scheduled_jobs: dict[str, CronJob] = {}  # 注册的定时任务
cron_queue: list[CronJob] = []           # 已触发的任务队列（调度器写入，消费者取出）
cron_lock = threading.Lock()             # 保护 scheduled_jobs + cron_queue 的互斥锁
_last_fired: dict[str, str] = {}         # job_id → "YYYY-MM-DD HH:MM" 防止同一分钟重复触发


def _cron_field_matches(field: str, value: int) -> bool:
    """匹配单个 cron 字段与给定值。
    支持通配符 *、步进 */N、列表 a,b,c、范围 a-b。"""
    if field == "*":
        return True
    if field.startswith("*/"):  # 步进：*/5 表示每 5 个单位
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:  # 列表：1,3,5 表示匹配 1、3、5
        return any(_cron_field_matches(f.strip(), value)
                   for f in field.split(","))
    if "-" in field:  # 范围：1-5 表示 1 到 5
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)
    return value == int(field)


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """检查 5 字段 cron 表达式是否匹配给定时间。
    标准 cron 语义：DOM（月内第几天）和 DOW（周几）同时约束时使用 OR。"""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val = (dt.weekday() + 1) % 7  # Python Monday=0 → cron Sunday=0

    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)

    # 分钟、小时、月份必须全部匹配
    if not (m and h and month_ok):
        return False
    # DOM 和 DOW：同时约束时 OR 语义（任一满足即可）
    dom_unconstrained = dom == "*"
    dow_unconstrained = dow == "*"
    if dom_unconstrained and dow_unconstrained:
        return True
    if dom_unconstrained:
        return dow_ok
    if dow_unconstrained:
        return dom_ok
    return dom_ok or dow_ok


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    """验证单个 cron 字段值是否在 [lo, hi] 范围内。返回错误信息或 None。"""
    if field == "*":
        return None
    if field.startswith("*/"):
        step_str = field[2:]
        if not step_str.isdigit():
            return f"Invalid step: {field}"
        step = int(step_str)
        if step <= 0:
            return f"Step must be > 0: {field}"
        return None
    if "," in field:  # 递归验证列表中的每个值
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err: return err
        return None
    if "-" in field:  # 验证范围：两端必须数字且不越界，start ≤ end
        parts = field.split("-", 1)
        if not parts[0].isdigit() or not parts[1].isdigit():
            return f"Invalid range: {field}"
        a, b = int(parts[0]), int(parts[1])
        if a < lo or a > hi or b < lo or b > hi:
            return f"Range {field} out of bounds [{lo}-{hi}]"
        if a > b:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    val = int(field)
    if val < lo or val > hi:
        return f"Value {val} out of bounds [{lo}-{hi}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    """验证 cron 表达式合法性。返回错误信息或 None（表示通过）。"""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    for i, (field, (lo, hi), name) in enumerate(zip(fields, bounds, names)):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


def save_durable_jobs():
    """将 durable=True 的定时任务持久化到 .scheduled_tasks.json。"""
    durable = [asdict(j) for j in scheduled_jobs.values() if j.durable]
    DURABLE_PATH.write_text(json.dumps(durable, indent=2))


def load_durable_jobs():
    """启动时从磁盘加载持久化的定时任务。跳过 cron 表达式不合法的任务。"""
    if not DURABLE_PATH.exists():
        return
    try:
        jobs = json.loads(DURABLE_PATH.read_text())
        for j in jobs:
            job = CronJob(**j)
            err = validate_cron(job.cron)  # 重新验证（防止人为篡改 JSON）
            if err:
                print(f"  \033[31m[cron] skipping invalid job {job.id}: {err}\033[0m")
                continue
            scheduled_jobs[job.id] = job
        valid = [j for j in jobs if j["id"] in scheduled_jobs]
        if valid:
            print(f"  \033[35m[cron] loaded {len(valid)} durable job(s)\033[0m")
    except Exception:
        pass


def schedule_job(cron: str, prompt: str, recurring: bool = True,
                 durable: bool = True) -> CronJob | str:
    """注册新的定时任务。先验证 cron → 创建 CronJob → 加锁写入 scheduled_jobs。"""
    err = validate_cron(cron)
    if err:
        return err
    job = CronJob(
        id=f"cron_{random.randint(0, 999999):06d}",
        cron=cron, prompt=prompt,
        recurring=recurring, durable=durable,
    )
    with cron_lock:
        scheduled_jobs[job.id] = job
    if durable:
        save_durable_jobs()  # 持久化到磁盘
    print(f"  \033[35m[cron register] {job.id} '{cron}' → {prompt[:40]}\033[0m")
    return job


def cancel_job(job_id: str) -> str:
    """取消定时任务。从 scheduled_jobs 中移除，durable 任务同步更新磁盘。"""
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    if not job:
        return f"Job {job_id} not found"
    if job.durable:
        save_durable_jobs()
    print(f"  \033[31m[cron cancel] {job_id}\033[0m")
    return f"Cancelled {job_id}"


def cron_scheduler_loop():
    """Layer 1: 独立守护线程——每秒轮询，触发匹配的定时任务。
    - minute_marker 防止同一分钟内重复触发（跨天不断档）
    - 单个任务的异常被捕获，防止一个坏任务拖垮整个调度器线程
    - 一次性任务（recurring=False）触发后自动移除"""
    while True:
        time.sleep(1)
        now = datetime.now()
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
        with cron_lock:
            for job in list(scheduled_jobs.values()):
                try:
                    if cron_matches(job.cron, now):
                        if _last_fired.get(job.id) != minute_marker:
                            cron_queue.append(job)  # 写入队列
                            _last_fired[job.id] = minute_marker
                            print(f"  \033[35m[cron fire] {job.id} → "
                                  f"{job.prompt[:40]}\033[0m")
                        if not job.recurring:  # 一次性任务：触发后移除
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                save_durable_jobs()
                except Exception as e:  # 单个任务异常不杀死调度器线程
                    print(f"  \033[31m[cron error] {job.id}: {e}\033[0m")


def consume_cron_queue() -> list[CronJob]:
    """消费 cron_queue 中已触发的任务（由 agent_loop 调用）。"""
    with cron_lock:
        fired = list(cron_queue)
        cron_queue.clear()
    return fired


# 启动时加载持久化任务，然后启动调度器守护线程
load_durable_jobs()
threading.Thread(target=cron_scheduler_loop, daemon=True).start()
print("  \033[35m[cron] scheduler thread started\033[0m")


# ═══════════════════════════════════════════════════════════
#  s14: Cron 工具（3 个工具）
#  [schedule](紫)/[list](紫)/[cancel](红)
# ═══════════════════════════════════════════════════════════

def run_schedule_cron(cron: str, prompt: str,
                      recurring: bool = True, durable: bool = True) -> str:
    """注册定时任务：验证 cron → 创建 CronJob → 加入调度。"""
    result = schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: '{cron}' → {prompt}"


def run_list_crons() -> str:
    """列出所有已注册的定时任务，含 cron 表达式和标签（循环/一次性, 持久/会话）。"""
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs. Use schedule_cron to add one."
    lines = []
    for j in jobs:
        tag = "recurring" if j.recurring else "one-shot"
        dur = "durable" if j.durable else "session"
        lines.append(f"  {j.id}: '{j.cron}' → {j.prompt[:40]} "
                     f"[{tag}, {dur}]")
    return "\n".join(lines)


def run_cancel_cron(job_id: str) -> str:
    """取消定时任务：从 scheduled_jobs 移除（durable 任务同步更新磁盘）。"""
    return cancel_job(job_id)


# ═══════════════════════════════════════════════════════════
#  NEW in s15: MessageBus（消息总线）
#  基于文件的邮箱系统，每个 agent 一个 .jsonl inbox
#  read 即消费（read_text + unlink），教学版无文件锁
# ═══════════════════════════════════════════════════════════

MAILBOX_DIR = WORKDIR / ".mailboxes"  # 邮箱文件存储目录
MAILBOX_DIR.mkdir(exist_ok=True)


class MessageBus:
    """基于文件的消息总线。每个 agent 有一个 .jsonl 收件箱。
    - send: 追加一行 JSON 到目标 agent 的 .jsonl 文件
    - read_inbox: 读取全部消息后 unlink（消费语义——读即删除）
    教学版：无文件锁；真实 CC 用 proper-lockfile 保证并发写安全。"""

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message"):
        """发送消息：追加 JSON 行到目标 agent 的 .jsonl 收件箱。"""
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time()}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a") as f:
            f.write(json.dumps(msg) + "\n")
        print(f"  \033[33m[bus] {from_agent} → {to_agent}: "
              f"{content[:50]}\033[0m")

    def read_inbox(self, agent: str) -> list[dict]:
        """读取收件箱全部消息并删除文件（消费语义）。"""
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        msgs = [json.loads(line) for line in inbox.read_text().splitlines()
                if line.strip()]
        inbox.unlink()  # 读后即删，实现消费语义
        return msgs


BUS = MessageBus()  # 全局消息总线实例

# 跟踪已生成的队友
active_teammates: dict[str, bool] = {}


# ═══════════════════════════════════════════════════════════
#  NEW in s15: Teammate Thread（队友线程）
#  后台线程创建独立 Agent，运行简化版 agent_loop
#  教学版：最多 10 轮；真实 CC：idle loop 持续监听收件箱
# ═══════════════════════════════════════════════════════════

def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    """在后台线程中生成队友 Agent。
    教学版：每个队友最多 10 轮对话。
    真实 CC：队友使用 idle loop（等待收件箱 → 工作 → 重复）直到 shutdown。"""
    if name in active_teammates: # 简单检查名字冲突，真实 CC 可改为更健壮的 UUID 生成
        return f"Teammate '{name}' already exists"

    system = (f"You are '{name}', a {role}. "
              f"Use tools to complete tasks. "
              f"Send results via send_message to 'lead'.")

    def run():
        """队友的主循环：收件箱检查 → LLM 调用 → 工具执行 → 循环（最多 10 次）。"""
        messages = [{"role": "user", "content": prompt}]
        # 队友只有 4 个工具：bash, read_file, write_file, send_message（不含任务/定时/生成队友）
        sub_tools = [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object",
                              "properties": {"command": {"type": "string"}},
                              "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"}},
                              "required": ["path"]}},
            {"name": "write_file", "description": "Write content to a file.",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["path", "content"]}},
            {"name": "send_message",
             "description": "Send a message to another agent.",
             "input_schema": {"type": "object",
                              "properties": {"to": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["to", "content"]}},
        ]
        sub_handlers = {
            "bash": run_bash, "read_file": run_read, "write_file": run_write,
            "send_message": lambda to, content: (BUS.send(name, to, content), "Sent")[1], # 发送消息后返回 "Sent" 作为工具结果
        }

        for _ in range(10):  # 最多 10 轮
            # 每轮检查自己的收件箱
            inbox = BUS.read_inbox(name)
            if inbox:
                messages.append({"role": "user",
                                 "content": f"<inbox>{json.dumps(inbox)}</inbox>"})
            try:
                response = client.messages.create(
                    model=MODEL, system=system, messages=messages[-20:],  # 只保留最近 20 条
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

        # 提取最后的文本回复作为摘要发送给 Lead
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

    active_teammates[name] = True # 标记队友为活跃状态
    threading.Thread(target=run, daemon=True).start() # daemon=True: 主线程退出时自动回收
    print(f"  \033[36m[teammate] {name} spawned as {role}\033[0m") # 青色标签显示生成的队友和角色
    return f"Teammate '{name}' spawned as {role}"


# ═══════════════════════════════════════════════════════════
#  s15: Team 工具（3 个新工具）
#  [spawn](青)/[send](黄)/[inbox](品红)
# ═══════════════════════════════════════════════════════════

def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    """生成队友：后台线程启动独立 Agent，4 工具（bash/read/write/send_message）。"""
    return spawn_teammate_thread(name, role, prompt)


def run_send_message(to: str, content: str) -> str:
    """通过 MessageBus 发送消息给指定 agent。"""
    BUS.send("lead", to, content)
    return f"Sent to {to}"


def run_check_inbox() -> str:
    """检查 Lead 收件箱中的队友消息（消费语义——读取后删除）。"""
    msgs = BUS.read_inbox("lead")
    if not msgs:
        return "(inbox empty)"
    lines = []
    for m in msgs:
        lines.append(f"  [{m['from']}] {m['content'][:200]}")
    return "\n".join(lines)


# ── 工具定义（s15 新增 3 个 Team 工具，共 14 个）─────────────────

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
    {"name": "schedule_cron",
     "description": "Schedule a cron job. cron is 5-field: min hour dom month dow.",
     "input_schema": {"type": "object",
                      "properties": {
                          "cron": {"type": "string",
                                   "description": "5-field cron expression"},
                          "prompt": {"type": "string",
                                     "description": "Message to inject when fired"},
                          "recurring": {"type": "boolean",
                                        "description": "True=recurring, False=one-shot"},
                          "durable": {"type": "boolean",
                                      "description": "True=persist to disk"}},
                      "required": ["cron", "prompt"]}},
    {"name": "list_crons",
     "description": "List all registered cron jobs.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
    {"name": "cancel_cron",
     "description": "Cancel a cron job by ID.",
     "input_schema": {"type": "object",
                      "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]}},
    {"name": "spawn_teammate",
     "description": "Spawn a teammate agent in a background thread.",
     "input_schema": {"type": "object",
                      "properties": {
                          "name": {"type": "string"},
                          "role": {"type": "string"},
                          "prompt": {"type": "string"}},
                      "required": ["name", "role", "prompt"]}},
    {"name": "send_message",
     "description": "Send a message to a teammate via MessageBus.",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["to", "content"]}},
    {"name": "check_inbox",
     "description": "Check Lead's inbox for teammate messages.",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
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
#  agent_loop — s15 核心：定时任务 + 后台工具 + 队友收件箱注入
#  教学版保持基本循环，S11 的完整错误恢复被省略。
# ═══════════════════════════════════════════════════════════

def _tool_input_summary(block) -> str:
    """提取工具调用的关键参数摘要，用于终端显示。"""
    inputs = block.input
    if block.name == "bash":
        return inputs.get("command", "")
    if block.name in ("create_task", "schedule_cron"):
        return inputs.get("subject", inputs.get("cron", ""))
    if block.name in ("get_task", "claim_task", "complete_task", "cancel_cron"):
        return inputs.get("task_id", inputs.get("job_id", ""))
    if block.name in ("list_tasks", "list_crons", "check_inbox"):
        return ""
    if block.name == "spawn_teammate":
        return f"{inputs.get('name', '')} ({inputs.get('role', '')})"
    if block.name == "send_message":
        return f"→ {inputs.get('to', '')}"
    return inputs.get("path", "")


def agent_loop(messages: list, context: dict):
    """智能体主循环 + 定时任务 + 后台工具 + 队友收件箱。
    流程：
    1. 消费 cron_queue 中已触发的定时任务 → 注入 user 消息
    2. LLM 调用 → 工具分发（慢→后台 / 快→同步）
    3. 收集后台通知 + 工具结果 → 合入一条 user 消息
    4. 刷新 context + prompt → 循环
    """
    system = get_system_prompt(context)
    while True:
        # 消费已触发的定时任务 → 注入对话
        fired = consume_cron_queue()
        for job in fired:
            messages.append({"role": "user",
                             "content": f"[Scheduled] {job.prompt}"})
            print(f"  \033[35m[inject cron] {job.prompt[:50]}\033[0m")

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

        # 注入 工具结果 + 后台通知 到一条 user 消息中
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
# 流程：读取输入 → agent_loop → 打印回复 → 检查队友收件箱注入 → 循环
if __name__ == "__main__":
    print("s15: agent teams")
    print("Enter a question, press Enter to send. Type q to quit.\n")
    history = []  # 消息历史
    context = update_context({}, [])  # s10: 启动时初始化 context
    while True:
        try:
            query = input("\033[36ms15 >> \033[0m")
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

        # 检查队友收件箱 → 注入到历史（让 LLM 后续可见队友消息）
        inbox = BUS.read_inbox("lead")
        if inbox:
            inbox_text = "\n".join(
                f"From {m['from']}: {m['content'][:200]}" for m in inbox)
            history.append({"role": "user",
                            "content": f"[Inbox]\n{inbox_text}"})
            print(f"\n\033[33m[Inbox: {len(inbox)} messages injected]\033[0m")
        print()
