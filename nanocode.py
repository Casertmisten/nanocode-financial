#!/usr/bin/env python3
"""nanocode-financial - 个人金融智能分析助手"""

import json
import os
import re
import urllib.request

import config
import router
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
            if user_input in ("/q", "exit"):
                break
            if user_input == "/clear":
                messages = []
                print(f"{config.GREEN}⏺ Cleared conversation{config.RESET}")
                continue

            messages.append({"role": "user", "content": user_input})

            # 路由判断
            route = router.route_query(user_input)
            route_label = "深度研究" if route == "deep_research" else "知识问答"
            print(f"{config.YELLOW}⏺ 路由 → {route_label}{config.RESET}")

            # Agentic loop
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

            print()

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as err:
            print(f"{config.RED}⏺ Error: {err}{config.RESET}")


if __name__ == "__main__":
    main()
