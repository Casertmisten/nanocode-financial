#!/usr/bin/env python3
"""nanocode - minimal claude code alternative"""

import glob as globlib
import json
import os
import re
import subprocess
import urllib.request
import dotenv

# 配置环境变量
dotenv.load_dotenv()
OPENROUTER_KEY = os.environ.get("ALI_API_KEY")  # 兼容Alpaca API Key环境变量
API_URL = os.environ.get("ALI_API_URL")
MODEL = os.environ.get("ALI_MODEL")

# ANSI colors
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = (
    "\033[34m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
)


# --- Tool implementations ---


def read(args):
    lines = open(args["path"]).readlines()
    offset = args.get("offset", 0)
    limit = args.get("limit", len(lines))
    selected = lines[offset : offset + limit]
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))

def write(args):
    with open(args["path"], "w") as f:
        f.write(args["content"])
    return "ok"

def edit(args):
    text = open(args["path"]).read()
    old, new = args["old"], args["new"]
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    replacement = (
        text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
    )
    with open(args["path"], "w") as f:
        f.write(replacement)
    return "ok"

def glob(args):
    pattern = (args.get("path", ".") + "/" + args["pat"]).replace("//", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(
        files,
        key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0,
        reverse=True,
    )
    return "\n".join(files) or "none"

def grep(args):
    pattern = re.compile(args["pat"])
    hits = []
    for filepath in globlib.glob(args.get("path", ".") + "/**", recursive=True):
        try:
            for line_num, line in enumerate(open(filepath), 1):
                if pattern.search(line):
                    hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"

def bash(args):
    proc = subprocess.Popen(
        args["cmd"], shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )
    output_lines = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(f"  {DIM}│ {line.rstrip()}{RESET}", flush=True)
                output_lines.append(line)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        output_lines.append("\n(timed out after 30s)")
    return "".join(output_lines).strip() or "(empty)"


# --- Tool definitions: (description, schema, function) ---
# 工具定义：（描述，参数，函数）

TOOLS = {
    "read": (
        "Read file with line numbers (file path, not directory)",
        {"path": "string", "offset": "number?", "limit": "number?"},
        read,
    ),
    "write": (
        "Write content to file",
        {"path": "string", "content": "string"},
        write,
    ),
    "edit": (
        "Replace old with new in file (old must be unique unless all=true)",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "glob": (
        "Find files by pattern, sorted by mtime",
        {"pat": "string", "path": "string?"},
        glob,
    ),
    "grep": (
        "Search files for regex pattern",
        {"pat": "string", "path": "string?"},
        grep,
    ),
    "bash": (
        "Run shell command",
        {"cmd": "string"},
        bash,
    ),
}


def run_tool(name, args):
    '''运行工具'''
    
    try:
        return TOOLS[name][2](args)
    except Exception as err:
        return f"error: {err}"


def make_schema():
    '''将 TOOLS 注册表转换为 OpenAI 兼容的函数调用格式列表，用于传给模型告知有什么工具'''
    result = []
    for name, (description, params, _fn) in TOOLS.items():
        # 遍历每个工具的参数，构建 JSON Schema 属性定义
        properties = {}
        required = []
        for param_name, param_type in params.items():
            # 参数类型以 "?" 结尾表示可选，如 "number?" -> 可选整数
            is_optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")
            properties[param_name] = {
                "type": "integer" if base_type == "number" else base_type
            }
            if not is_optional:
                required.append(param_name)
        # 组装为 OpenAI tools 格式
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return result


def call_api(messages, system_prompt):
    '''调用API'''
    all_messages = [{"role": "system", "content": system_prompt}] + messages
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(
            {
                "model": MODEL,
                "max_tokens": 8192,
                "messages": all_messages,
                "tools": make_schema(),
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_KEY}",
        },
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read())


def separator():
    '''分隔符'''
    return f"{DIM}{'─' * min(os.get_terminal_size().columns, 80)}{RESET}"


def render_markdown(text):
    '''渲染Markdown（仅支持**加粗**）'''
    return re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)


def main():
    '''主循环：读取用户输入，驱动 agentic loop 与模型交互'''
    # 打印欢迎信息：模型名称、当前工作目录
    print(f"{BOLD}nanocode{RESET} | {DIM}{MODEL} (ZhipuAI) | {os.getcwd()}{RESET}\n")
    messages = []  # 对话历史，每轮追加 user/assistant/tool 消息
    system_prompt = f"Concise coding assistant. cwd: {os.getcwd()}"

    while True:
        try:
            print(separator())
            user_input = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
            print(separator())
            # 空输入跳过
            if not user_input:
                continue
            # 退出命令
            if user_input in ("/q", "exit"):
                break
            # 清空对话历史
            if user_input == "/clear":
                messages = []
                print(f"{GREEN}⏺ Cleared conversation{RESET}")
                continue

            # 将用户消息加入对话历史
            messages.append({"role": "user", "content": user_input})
            
            print("debug：整个messages----------------------------------------------------")
            print(messages)
            print("debug-----------------------------------------------------------------")

            # agentic loop：反复调用 API，直到模型不再请求工具调用
            while True:
                response = call_api(messages, system_prompt)
                msg = response["choices"][0]["message"]
                tool_calls = msg.get("tool_calls") or []
                tool_results = []

                # 打印模型的文本回复
                if msg.get("content"):
                    print(f"\n{CYAN}⏺{RESET} {render_markdown(msg['content'])}")

                # 逐个执行模型请求的工具调用
                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])
                    arg_preview = str(list(tool_args.values())[0])[:50]
                    print(
                        f"\n{GREEN}⏺ {tool_name.capitalize()}{RESET}({DIM}{arg_preview}{RESET})"
                    )

                    # 本地执行工具，并将结果截断预览打印到终端
                    result = run_tool(tool_name, tool_args)
                    result_lines = result.split("\n")
                    preview = result_lines[0][:60]
                    if len(result_lines) > 1:
                        preview += f" ... +{len(result_lines) - 1} lines"
                    elif len(result_lines[0]) > 60:
                        preview += "..."
                    print(f"  {DIM}⎿  {preview}{RESET}")

                    # 将工具执行结果按 OpenAI 格式封装，后续喂回模型
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        }
                    )

                # 将模型的回复（含 tool_calls）追加到对话历史
                messages.append(msg)

                # 没有工具调用则本轮结束，否则把工具结果追加到历史继续循环
                if not tool_results:
                    break
                messages.extend(tool_results)

            print()

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as err:
            print(f"{RED}⏺ Error: {err}{RESET}")


if __name__ == "__main__":
    main()
