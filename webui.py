#!/usr/bin/env python3
"""nanocode-financial Web UI - 基于 Gradio 的前端交互页面."""

import json
import os
import urllib.request
import uuid
from datetime import datetime

import gradio as gr

import config
from tools import make_schema, run_tool


def call_api(messages, system_prompt):
    """调用 LLM API (OpenAI Chat Completions compatible)."""
    all_messages = [{"role": "system", "content": system_prompt}] + messages
    req = urllib.request.Request(
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
    response = urllib.request.urlopen(req)
    return json.loads(response.read())


def load_system_prompt() -> str:
    """加载系统提示词，优先从 prompts/rag_system.txt，否则用默认。"""
    prompt_path = os.path.join(config.BASE_DIR, "prompts", "rag_system.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            return f.read().replace("{cwd}", os.getcwd())
    return f"你是一个金融分析助手。当前工作目录: {os.getcwd()}"


SYSTEM_PROMPT = load_system_prompt()

SESSIONS_DIR = os.path.join(config.BASE_DIR, "data", "sessions")

CUSTOM_CSS = """
.title-bar { text-align: center; padding: 8px 0; }
.sidebar { background: #f7f7f8; border-radius: 12px; padding: 12px; min-height: 600px; }
.sidebar .new-chat-btn button { width: 100%; border-radius: 8px; font-size: 14px; padding: 10px; }
.group-header { font-size: 0.8em; color: #666; font-weight: 600; padding: 8px 4px 2px; margin: 0; }
.session-list { padding: 0 4px; }
.session-list label { padding: 6px 8px !important; border-radius: 8px !important; font-size: 0.9em; }
.session-list label:hover { background: #ececec !important; }
.session-list label.selected { background: #e3e3e3 !important; }
.delete-btn button { width: 100%; margin-top: 8px; }
"""

GROUP_ORDER = ["today", "yesterday", "week", "month", "older"]
GROUP_LABELS = {
    "today": "📅 今天",
    "yesterday": "📅 昨天",
    "week": "📅 7天内",
    "month": "📅 30天内",
    "older": "📅 更早",
}


def _ensure_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _save_session(session_id: str, messages: list, title: str):
    _ensure_sessions_dir()
    data = {
        "id": session_id,
        "title": title,
        "messages": messages,
        "updated_at": datetime.now().isoformat(),
    }
    with open(_session_path(session_id), "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return {"id": session_id, "title": "新会话", "messages": []}
    with open(path, "r") as f:
        return json.load(f)


def _list_sessions() -> list[dict]:
    _ensure_sessions_dir()
    sessions = []
    for fname in sorted(os.listdir(SESSIONS_DIR), reverse=True):
        if fname.endswith(".json"):
            with open(os.path.join(SESSIONS_DIR, fname), "r") as f:
                data = json.load(f)
                sessions.append(
                    {
                        "id": data.get("id", fname[:-5]),
                        "title": data.get("title", "未命名"),
                        "updated_at": data.get("updated_at", ""),
                    }
                )
    return sessions


def _group_sessions(sessions: list[dict]) -> dict:
    """将会话按更新时间分组."""
    now = datetime.now()
    groups = {k: [] for k in GROUP_ORDER}
    for s in sessions:
        try:
            updated = datetime.fromisoformat(s["updated_at"])
        except (ValueError, TypeError):
            groups["older"].append(s)
            continue
        days_ago = (now - updated).days
        if days_ago == 0:
            groups["today"].append(s)
        elif days_ago == 1:
            groups["yesterday"].append(s)
        elif days_ago < 7:
            groups["week"].append(s)
        elif days_ago < 30:
            groups["month"].append(s)
        else:
            groups["older"].append(s)
    return groups


def _build_display_messages(session_messages: list) -> list:
    """将 session_messages 转换为 Gradio 6 messages 格式."""
    display = []
    for msg in session_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        ts = msg.get("timestamp", "")
        if role == "user":
            display.append(
                {"role": "user", "content": f"{content}\n\n*— {ts}*" if ts else content}
            )
        elif role == "assistant":
            display.append({"role": "assistant", "content": content or ""})
        elif role == "tool_call":
            tool_info = (
                f"🔧 **工具调用**: `{content}`\n\n"
                f"<details><summary>查看结果</summary>\n\n"
                f"```\n{msg.get('result', '')}\n```\n</details>"
            )
            display.append({"role": "assistant", "content": tool_info})
    return display


def _run_agentic_loop(user_message: str, session_messages: list):
    """执行 agentic loop，返回更新后的 session_messages."""
    ts = _now_str()
    session_messages.append(
        {"role": "user", "content": user_message, "timestamp": ts, "sender": "user"}
    )

    api_messages = []
    for m in session_messages:
        if m["role"] in ("user", "assistant") and m.get("content"):
            api_messages.append({"role": m["role"], "content": m["content"]})
        elif m["role"] == "tool_result":
            api_messages.append(m)

    while True:
        try:
            response = call_api(api_messages, SYSTEM_PROMPT)
        except Exception as err:
            ts2 = _now_str()
            session_messages.append(
                {"role": "assistant", "content": f"API 调用失败: {err}", "timestamp": ts2, "sender": "assistant"}
            )
            break

        msg = response["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        tool_results = []

        if msg.get("content"):
            ts2 = _now_str()
            session_messages.append(
                {"role": "assistant", "content": msg["content"], "timestamp": ts2, "sender": "assistant"}
            )
            api_messages.append({"role": "assistant", "content": msg["content"]})

        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_args = json.loads(tc["function"]["arguments"])
            ts3 = _now_str()
            arg_preview = str(list(tool_args.values())[0])[:50] if tool_args else ""
            call_desc = f"{tool_name}({arg_preview})"
            result = run_tool(tool_name, tool_args)
            session_messages.append(
                {"role": "tool_call", "content": call_desc, "result": result[:500], "timestamp": ts3, "sender": "tool"}
            )
            tool_results.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        api_messages.append(msg)
        if not tool_results:
            break
        api_messages.extend(tool_results)
        for tr in tool_results:
            session_messages.append({"role": "tool_result", **tr, "timestamp": _now_str(), "sender": "tool"})

    return session_messages


def _auto_title(messages: list) -> str:
    """从第一条用户消息自动生成会话标题."""
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            title = m["content"][:30]
            return title + "..." if len(m["content"]) > 30 else title
    return "新会话"


def _get_group_updates():
    """获取所有分组的更新（用于刷新列表）."""
    sessions = _list_sessions()
    grouped = _group_sessions(sessions)
    updates = []
    for key in GROUP_ORDER:
        items = grouped[key]
        if items:
            choices = [(s["title"], s["id"]) for s in items]
            updates.append(gr.update(visible=True))
            updates.append(gr.update(choices=choices, value=None, visible=True))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(choices=[], value=None, visible=False))
    return updates


def build_ui():
    """构建 Gradio 界面."""
    with gr.Blocks(title="nanocode-financial 金融智能助手") as demo:
        gr.Markdown(
            "# 📊 nanocode-financial 金融智能助手\n"
            f"模型: `{config.MODEL}` | 基于 RAG 的金融知识问答",
            elem_classes=["title-bar"],
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=260, elem_classes=["sidebar"]):
                new_btn = gr.Button("➕ 开启新对话", variant="primary", elem_classes=["new-chat-btn"])

                group_cols = {}
                group_radios = {}
                for key in GROUP_ORDER:
                    col = gr.Column(visible=False, elem_classes=[f"group-{key}"])
                    with col:
                        gr.Markdown(GROUP_LABELS[key], elem_classes=["group-header"])
                        radio = gr.Radio(
                            choices=[], interactive=True, show_label=False,
                            elem_classes=["session-list"],
                        )
                    group_cols[key] = col
                    group_radios[key] = radio

                delete_btn = gr.Button("🗑️ 删除选中会话", size="sm", elem_classes=["delete-btn"])

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="对话", height=520, buttons=["copy"])

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入金融问题，例如：最近A股市场表现如何？",
                        show_label=False, scale=5, lines=2, autofocus=True,
                    )
                    send_btn = gr.Button("发送 ➤", variant="primary", scale=1, min_width=80)

        session_id_state = gr.State(str(uuid.uuid4())[:8])
        session_messages_state = gr.State([])
        session_title_state = gr.State("新会话")

        all_group_outputs = []
        for key in GROUP_ORDER:
            all_group_outputs.append(group_cols[key])
            all_group_outputs.append(group_radios[key])

        def _refresh_all():
            updates = _get_group_updates()
            return updates

        def _do_send(user_message, session_messages, session_id, session_title):
            if not user_message.strip():
                return [], session_messages, session_id, session_title, user_message
            session_messages = _run_agentic_loop(user_message.strip(), session_messages)
            if session_title == "新会话":
                session_title = _auto_title(session_messages)
            _save_session(session_id, session_messages, session_title)
            display = _build_display_messages(session_messages)
            return display, session_messages, session_id, session_title, ""

        def _do_new():
            sid = str(uuid.uuid4())[:8]
            return sid, [], "新会话", [], ""

        def _make_radio_handler(group_key):
            other_keys = [k for k in GROUP_ORDER if k != group_key]
            other_radios = [group_radios[k] for k in other_keys]

            def handler(selected_id, sm, sid):
                if not selected_id:
                    result = [sm, sid, "新会话", [], ""]
                    result += [gr.update(value=None)] * len(other_radios)
                    return result
                data = _load_session(selected_id)
                msgs = data.get("messages", [])
                title = data.get("title", "未命名")
                display = _build_display_messages(msgs)
                result = [msgs, selected_id, title, display, ""]
                result += [gr.update(value=None)] * len(other_radios)
                return result
            return handler

        def _do_delete(session_messages, session_id, *radio_values):
            selected_id = None
            for key in GROUP_ORDER:
                val = radio_values[GROUP_ORDER.index(key)]
                if val:
                    selected_id = val
                    break
            if selected_id:
                path = _session_path(selected_id)
                if os.path.exists(path):
                    os.remove(path)
            sid = str(uuid.uuid4())[:8]
            updates = _get_group_updates()
            return sid, [], "新会话", [], "" if selected_id else session_messages, updates

        for key in GROUP_ORDER:
            radio = group_radios[key]
            other_keys = [k for k in GROUP_ORDER if k != key]
            other_radios = [group_radios[k] for k in other_keys]

            handler = _make_radio_handler(key)
            radio.change(
                fn=handler,
                inputs=[radio, session_messages_state, session_id_state],
                outputs=[session_messages_state, session_id_state, session_title_state, chatbot, msg_input] + other_radios,
            ).then(
                fn=lambda: _get_group_updates(),
                outputs=all_group_outputs,
            )

        send_btn.click(
            fn=_do_send,
            inputs=[msg_input, session_messages_state, session_id_state, session_title_state],
            outputs=[chatbot, session_messages_state, session_id_state, session_title_state, msg_input],
        ).then(
            fn=lambda: _get_group_updates(),
            outputs=all_group_outputs,
        )

        msg_input.submit(
            fn=_do_send,
            inputs=[msg_input, session_messages_state, session_id_state, session_title_state],
            outputs=[chatbot, session_messages_state, session_id_state, session_title_state, msg_input],
        ).then(
            fn=lambda: _get_group_updates(),
            outputs=all_group_outputs,
        )

        new_btn.click(
            fn=_do_new,
            outputs=[session_id_state, session_messages_state, session_title_state, chatbot, msg_input],
        ).then(
            fn=lambda: _get_group_updates(),
            outputs=all_group_outputs,
        )

        delete_btn.click(
            fn=_do_delete,
            inputs=[session_messages_state, session_id_state] + [group_radios[k] for k in GROUP_ORDER],
            outputs=[session_id_state, session_messages_state, session_title_state, chatbot, msg_input] + all_group_outputs,
        )

        demo.load(fn=lambda: _get_group_updates(), outputs=all_group_outputs)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=CUSTOM_CSS,
    )
