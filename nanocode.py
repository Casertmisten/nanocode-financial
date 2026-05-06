#!/usr/bin/env python3
"""nanocode-financial - 个人金融智能分析助手"""

import json
import os
import re
import urllib.request

import config
from tools import make_schema, run_tool


def call_api(messages, system_prompt):
    """调用 LLM API (OpenAI Chat Completions compatible)."""
    all_messages = [{"role": "system", "content": system_prompt}] + messages
    request = urllib.request.Request(
        config.API_URL,
        data=json.dumps(
            {
                "model": config.MODEL,
                "max_tokens": 8192,
                "messages": all_messages,
                "tools": make_schema(),
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.API_KEY}",
        },
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read())


def separator():
    """终端分隔线"""
    return f"{config.DIM}{'─' * min(os.get_terminal_size().columns, 80)}{config.RESET}"


def render_markdown(text):
    """渲染 Markdown（仅支持 **加粗**）"""
    return re.sub(r"\*\*(.+?)\*\*", f"{config.BOLD}\\1{config.RESET}", text)


def load_system_prompt() -> str:
    """加载系统提示词，优先从 prompts/rag_system.txt，否则用默认。"""
    prompt_path = os.path.join(config.BASE_DIR, "prompts", "rag_system.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            return f.read().replace("{cwd}", os.getcwd())
    return f"你是一个金融分析助手。当前工作目录: {os.getcwd()}"


def _run_agentic_loop(messages, system_prompt):
    """执行 agentic loop，直到模型不再发起工具调用。"""
    while True:
        response = call_api(messages, system_prompt)
        msg = response["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        tool_results = []

        if msg.get("content"):
            print(f"\n{config.CYAN}⏺{config.RESET} {render_markdown(msg['content'])}")

        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_args = json.loads(tc["function"]["arguments"])
            arg_preview = str(list(tool_args.values())[0])[:50]
            print(
                f"\n{config.GREEN}⏺ {tool_name.capitalize()}{config.RESET}"
                f"({config.DIM}{arg_preview}{config.RESET})"
            )

            result = run_tool(tool_name, tool_args)
            result_lines = result.split("\n")
            preview = result_lines[0][:60]
            if len(result_lines) > 1:
                preview += f" ... +{len(result_lines) - 1} lines"
            elif len(result_lines[0]) > 60:
                preview += "..."
            print(f"  {config.DIM}⎿  {preview}{config.RESET}")

            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )

        messages.append(msg)

        if not tool_results:
            break
        messages.extend(tool_results)


def _deep_research_loop(query: str):
    """深度研究循环。"""
    import deep_research

    print(f"\n{config.YELLOW}⏺ 进入深度研究模式{config.RESET}")
    print(f"{config.DIM}  研究课题: {query}{config.RESET}\n")

    report = deep_research.run(query)

    print(f"\n{separator()}")
    print(render_markdown(report))
    print(separator())


# 斜杠命令注册表
_SLASH_COMMANDS = {
    "/deep_research": _deep_research_loop,
    "/clear": None,  # 内置命令，在 main 中特殊处理
}


def _handle_slash(user_input: str, messages: list) -> bool:
    """处理斜杠命令。返回 True 表示已处理，False 表示非斜杠命令。"""
    if user_input in ("/q", "exit"):
        raise SystemExit

    if user_input == "/clear":
        messages.clear()
        print(f"{config.GREEN}⏺ Cleared conversation{config.RESET}")
        return True

    cmd = user_input.split()[0]
    if cmd in _SLASH_COMMANDS:
        # 命令后面的文本作为参数
        args = user_input[len(cmd):].strip()
        handler = _SLASH_COMMANDS[cmd]
        if handler:
            handler(args)
        return True

    # 以 / 开头但未注册的命令
    if user_input.startswith("/"):
        print(f"{config.RED}⏺ 未知命令: {cmd}{config.RESET}")
        print(f"{config.DIM}  可用命令: {', '.join(_SLASH_COMMANDS.keys())}{config.RESET}")
        return True

    return False


def main():
    """主循环：读取用户输入，驱动 agentic loop 与模型交互。"""
    print(
        f"{config.BOLD}nanocode-financial{config.RESET} | "
        f"{config.DIM}{config.MODEL} | {os.getcwd()}{config.RESET}\n"
    )
    messages = []
    system_prompt = load_system_prompt()

    while True:
        try:
            print(separator())
            user_input = input(f"{config.BOLD}{config.BLUE}❯{config.RESET} ").strip()
            print(separator())

            if not user_input:
                continue

            # 斜杠命令检测
            if _handle_slash(user_input, messages):
                continue

            messages.append({"role": "user", "content": user_input})

            # RAG QA
            print(f"{config.YELLOW}⏺ 知识问答{config.RESET}")
            _run_agentic_loop(messages, system_prompt)
            print()

        except SystemExit:
            break
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as err:
            print(f"{config.RED}⏺ Error: {err}{config.RESET}")


if __name__ == "__main__":
    main()
