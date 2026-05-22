#!/usr/bin/env python3
"""nanocode-financial - 个人金融智能分析助手-CLI 版本"""

import datetime
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
from prompts.main_prompts import system_prompt as _SYSTEM_PROMPT_TEMPLATE
from tools import make_schema, run_tool
import memory.profile as _profile_mod
import memory.session_memory as _session_mem
from memory.context import estimate_tokens, _serialize_messages_for_summary


# --- 三层权限系统 ---

DENY_LIST = [
    "rm -rf /", "sudo", "shutdown", "reboot",
    "mkfs", "dd if=", "> /dev/sda",
]

_FILE_PATH_TOOLS = {"read", "write", "edit", "glob", "grep"}

# 并发安全工具：无副作用、只读、或内部线程安全
_CONCURRENT_SAFE_TOOLS = {
    "read", "glob", "grep",
    "stock_list", "stock_basic_info", "stock_quotes", "batch_stock_quotes",
    "stock_historical", "stock_financial",
    "market_status", "market_news", "stock_news",
    "web_search", "rag_query",
}

MAX_BATCH_SIZE = config.TOOL_BATCH_SIZE


def _check_deny_list(command: str) -> str | None:
    """第一层：硬拒绝检查。匹配到返回错误信息，否则返回 None。"""
    for pattern in DENY_LIST:
        if pattern in command:
            return f"已拦截: '{pattern}' 在禁止列表中"
    return None


def _is_path_in_workdir(path: str) -> bool:
    """判断路径是否在当前工作目录内。"""
    workdir = os.path.realpath(os.getcwd())
    target = os.path.realpath(os.path.abspath(path))
    return target == workdir or target.startswith(workdir + os.sep)


def _check_permission(name: str, args: dict) -> tuple[bool, str]:
    """三层权限检查。返回 (allowed, message)。"""
    # 第一层：bash 硬拒绝
    if name == "bash":
        deny_msg = _check_deny_list(args.get("cmd", ""))
        if deny_msg:
            return False, deny_msg

    # 第二层 + 第三层：文件路径检查
    if name in _FILE_PATH_TOOLS:
        path = args.get("path", ".")
        if path and path != ".":
            if not _is_path_in_workdir(path):
                # 第三层：询问用户
                print(f"\n{config.YELLOW}⏺ 权限请求: {name} 操作路径 '{path}' 在工作目录外{config.RESET}")
                confirm = input(f"  {config.BOLD}允许执行? (yes/no): {config.RESET}").strip().lower()
                if confirm != "yes":
                    return False, f"用户拒绝: {name} 操作外部路径 '{path}'"

    return True, ""


# --- 三层记忆注入 ---

def _count_rounds(messages: list[dict]) -> int:
    """统计对话轮数。"""
    count = 0
    for msg in messages:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            count += 1
    return count // 2


def _inject_memory(user_message: str) -> str:
    """L1 用户画像 + L2 跨会话记忆注入，返回追加到系统提示词的文本。"""
    parts = []
    try:
        profile = _profile_mod.load_profile()
        profile_md = _profile_mod.render_profile_markdown(profile)
        if profile_md:
            parts.append(profile_md)
    except Exception:
        pass
    try:
        memories = _session_mem.retrieve_memories(user_message)
        memories_md = _session_mem.render_memories_markdown(memories)
        if memories_md:
            parts.append(memories_md)
    except Exception:
        pass
    return "\n\n".join(parts)


def _compress_messages(messages: list[dict]) -> list[dict]:
    """L3 会话压缩（纯内存，无 DB 持久化）。"""
    max_tokens = config.MEMORY_SESSION_MAX_TOKENS
    compress_rounds = config.MEMORY_COMPRESS_ROUNDS
    min_keep_rounds = config.MEMORY_MIN_KEEP_ROUNDS
    result = list(messages)

    while estimate_tokens(result) > max_tokens:
        if _count_rounds(result) <= min_keep_rounds:
            break

        to_compress, remaining = [], []
        round_count, in_round = 0, False
        for msg in result:
            if msg.get("role") == "system":
                remaining.append(msg)
                continue
            if round_count < compress_rounds:
                to_compress.append(msg)
                if msg.get("role") == "user":
                    in_round = True
                if in_round and msg.get("role") == "assistant":
                    round_count += 1
                    in_round = False
            else:
                remaining.append(msg)

        if not to_compress or round_count == 0:
            break

        text = _serialize_messages_for_summary(to_compress)
        try:
            from memory.prompts import COMPRESS_PROMPT
            import llm
            summary = llm.call_llm("你是一个对话压缩助手。", COMPRESS_PROMPT.format(messages=text))
        except Exception:
            break

        result = [{"role": "system", "content": f"[对话摘要] {summary}"}] + remaining

    return result


def _save_memory(session_id: str, messages: list[dict]):
    """对话结束后保存 L2 跨会话记忆 + L1 画像候选。"""
    if not messages:
        return
    text = _serialize_messages_for_summary(messages)
    try:
        summary, topics, candidates = _session_mem.generate_summary(text)
        _session_mem.save_session_memory(session_id, summary, topics, "cli")
    except Exception:
        return
    if candidates:
        try:
            _profile_mod.add_candidate(session_id, candidates)
        except Exception:
            pass


def call_api(messages, system_prompt):
    """调用 LLM API (OpenAI Chat Completions compatible)."""
    import httpx
    body = {
        "model": config.MODEL,
        "max_tokens": config.LLM_MAX_TOKENS,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "tools": make_schema(),
    }
    resp = httpx.post(config.API_URL, json=body, headers={
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json",
    }, timeout=float(config.LLM_TIMEOUT))
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


def _print_tool_result(result: str):
    """打印工具执行结果预览。"""
    result_lines = result.split("\n")
    preview = result_lines[0][:60]
    if len(result_lines) > 1:
        preview += f" ... +{len(result_lines) - 1} lines"
    elif len(result_lines[0]) > 60:
        preview += "..."
    print(f"  {config.DIM}⎿  {preview}{config.RESET}")


def _run_agentic_loop(messages, system_prompt):
    """执行 agentic loop，直到模型不再发起工具调用。支持三层权限检查和批量并发执行。"""
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for turn in range(config.AGENT_MAX_TURNS):
        response = call_api(messages, system_prompt)

        # 累计 token 用量（含工具调用轮次）
        usage = response.get("usage") or {}
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)

        msg = response["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []

        if msg.get("content"):
            print(f"\n{config.CYAN}⏺{config.RESET} {render_markdown(msg['content'])}")

        if not tool_calls:
            messages.append(msg)
            break

        messages.append(msg)

        # 分批并发执行工具调用
        all_tool_results = {}
        for batch_start in range(0, len(tool_calls), MAX_BATCH_SIZE):
            batch = tool_calls[batch_start:batch_start + MAX_BATCH_SIZE]

            # 权限检查（需要用户交互，必须顺序执行）
            approved = []
            for tc in batch:
                tool_name = tc["function"]["name"]
                tool_args = json.loads(tc["function"]["arguments"])

                allowed, deny_msg = _check_permission(tool_name, tool_args)
                if not allowed:
                    print(f"  {config.RED}✗ {deny_msg}{config.RESET}")
                    all_tool_results[tc["id"]] = f"error: {deny_msg}"
                else:
                    arg_values = list(tool_args.values())
                    arg_preview = str(arg_values[0])[:50] if arg_values else ""
                    print(
                        f"\n{config.GREEN}⏺ {tool_name.capitalize()}{config.RESET}"
                        f"({config.DIM}{arg_preview}{config.RESET})"
                    )
                    approved.append((tc, tool_name, tool_args))

            if not approved:
                continue

            # 执行工具：按并发安全性分组
            safe = [(tc, n, a) for tc, n, a in approved if n in _CONCURRENT_SAFE_TOOLS]
            unsafe = [(tc, n, a) for tc, n, a in approved if n not in _CONCURRENT_SAFE_TOOLS]

            # 并发执行安全工具
            if len(safe) == 1:
                tc, tool_name, tool_args = safe[0]
                result = run_tool(tool_name, tool_args)
                all_tool_results[tc["id"]] = result
                _print_tool_result(result)
            elif len(safe) > 1:
                with ThreadPoolExecutor(max_workers=len(safe)) as executor:
                    futures = {}
                    for tc, tool_name, tool_args in safe:
                        future = executor.submit(run_tool, tool_name, tool_args)
                        futures[future] = tc

                    for future in as_completed(futures):
                        tc = futures[future]
                        result = future.result()
                        all_tool_results[tc["id"]] = result
                        _print_tool_result(result)

            # 串行执行不安全工具（write/edit/bash/rag_ingest 等有副作用的操作）
            for tc, tool_name, tool_args in unsafe:
                result = run_tool(tool_name, tool_args)
                all_tool_results[tc["id"]] = result
                _print_tool_result(result)

        # 按原始顺序生成 tool results
        tool_results = []
        for tc in tool_calls:
            if tc["id"] in all_tool_results:
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": all_tool_results[tc["id"]],
                })

        if not tool_results:
            break
        messages.extend(tool_results)
    else:
        print(f"{config.YELLOW}⏺ 已达最大轮次 {config.AGENT_MAX_TURNS}{config.RESET}")

    # 打印本轮 token 用量
    if total_usage["total_tokens"] > 0:
        print(f"{config.DIM}  Tokens: {total_usage['prompt_tokens']}+{total_usage['completion_tokens']}={total_usage['total_tokens']}{config.RESET}")


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
    session_id = f"cli_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

            # 注入知识库文档列表到系统提示词
            turn_prompt = system_prompt
            try:
                import db
                docs = db.get_document_titles()
                if docs:
                    doc_lines = "\n".join(f"- {d['title']} ({d['file_type']})" for d in docs)
                else:
                    doc_lines = "（知识库为空，暂无文档）"
                turn_prompt = system_prompt.replace("{documents}", doc_lines)
            except Exception:
                turn_prompt = system_prompt.replace("{documents}", "（知识库为空，暂无文档）")

            # 三层记忆注入
            try:
                memory_text = _inject_memory(user_input)
                if memory_text:
                    turn_prompt += "\n\n" + memory_text
            except Exception:
                pass

            # L3: 压缩超长对话
            try:
                messages[:] = _compress_messages(messages)
            except Exception:
                pass

            print(f"{config.YELLOW}⏺ 知识问答{config.RESET}")
            _run_agentic_loop(messages, turn_prompt)
            print()

            # 保存记忆
            try:
                _save_memory(session_id, messages)
            except Exception:
                pass

        except SystemExit:
            break
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as err:
            print(f"{config.RED}⏺ Error: {err}{config.RESET}")


if __name__ == "__main__":
    main()
