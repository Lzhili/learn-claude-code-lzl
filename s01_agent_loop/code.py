#!/usr/bin/env python3
"""
s01_agent_loop.py - 智能体主循环（Agent Loop）

AI 编程智能体的全部秘密，浓缩为一个模式：
1. 将用户消息发送给 LLM
2. LLM 返回工具调用请求 → 执行工具 → 将结果追加到消息历史
3. 循环上述过程，直到 LLM 不再请求工具，返回最终文本回复

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)
                          （循环继续）

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
这就是核心循环：将工具执行结果反复喂回给模型，直到模型决定停止。
生产级智能体在此基础上叠加策略、钩子和生命周期控制。

Usage / 用法:
    pip install anthropic python-dotenv
    ANTHROPIC_API_KEY=... python s01_agent_loop/code.py
"""

import os
import subprocess

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

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# 系统提示词：告诉模型它是一个编码智能体，使用 bash 解决问题
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ── 工具定义：仅提供 bash ───────────────────────────────
# 定义单个 bash 工具，模型可以请求执行任意 shell 命令
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


# ── 工具执行 ────────────────────────────────────────────
def run_bash(command: str) -> str:
    """执行 shell 命令并返回输出。

    安全措施：
    1. 黑名单检测：拦截危险命令（sudo/rm -rf / 等）
    2. 超时限制：120 秒，防止命令挂死
    3. 输出截断：最多返回 50,000 字符，避免 token 爆炸
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# ── 核心模式：while 循环调用工具，直到模型停止 ──────
def agent_loop(messages: list):
    """智能体主循环。
    流程：
    1. 将完整消息历史发送给 LLM
    2. 若 LLM 返回 tool_use → 执行工具 → 将结果追加到 messages
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

        # 遍历响应中的每个工具调用块，逐一执行
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                print(f"\033[1;35m[{block.name} -> Tool Calling Result]\033[0m \033[35m{output[:200]}\033[0m")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        # 将工具结果作为用户消息追加，触发下一轮循环
        messages.append({"role": "user", "content": results})


# ── 入口：交互式 REPL ──────────────────────────────────
# 流程：读取用户输入 → 调用 agent_loop → 打印模型最终回复 → 循环
if __name__ == "__main__":
    print("s01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []  # 消息历史，贯穿整个交互会话
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)  # 进入工具调用循环，直到模型给出最终回复
        # 打印模型最终文本回复
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(f"\033[34m{block.text}\033[0m")
        print()
