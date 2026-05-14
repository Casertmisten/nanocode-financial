#!/usr/bin/env python3
"""nanocode-financial - 个人金融智能分析助手-CLI 版本"""

import datetime
import json
import os
import re
import config
from prompts.main_prompts import system_prompt as _SYSTEM_PROMPT_TEMPLATE
from tools import make_schema, run_tool


def call_api(messages, system_prompt):
    """调用 LLM API (OpenAI Chat Completions compatible)."""
    import httpx
    body = {
        "model": config.MODEL,
        "max_tokens": 8192,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "tools": make_schema(),
    }
    resp = httpx.post(config.API_URL, json=body, headers={
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json",
    }, timeout=120.0)
    return resp.json()


def separator():
    """终端分隔线"""
    return f"{config.DIM}{'─' * min(os.get_terminal_size().columns, 80)}{config.RESET}"


def render_markdown(text):
    """渲染 Markdown（仅支持 **加粗**）"""
    return re.sub(r"\*\*(.+?)\*\*", f"{config.BOLD}\\1{config.RESET}", text)


def load_system_prompt() -> str:
    """加载系统提示词。"""
    return _SYSTEM_PROMPT_TEMPLATE.replace("{cwd}", os.getcwd())


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


def _financial_report_analysis_loop(query: str):
    """财报分析循环。"""
    import financial_report_analysis

    print(f"\n{config.YELLOW}⏺ 进入财报分析模式{config.RESET}")
    print(f"{config.DIM}  分析课题: {query}{config.RESET}\n")

    report = financial_report_analysis.run(query)

    # 保存为 Markdown 文件
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(config.BASE_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"financial_report_{ts}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{separator()}")
    print(report)
    print(separator())
    print(f"{config.GREEN}⏺ 报告已保存: {report_path}{config.RESET}")


# 斜杠命令注册表
_SLASH_COMMANDS = {
    "/fra": _financial_report_analysis_loop,
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
