// 统一 API 工具层
const API_BASE = '/api';

async function api(method, path, body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(`${API_BASE}${path}`, opts);
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`API ${res.status}: ${text}`);
    }
    return res.json();
}

/**
 * SSE POST 请求 — 用于对话流式输出
 * @param {string} path - API 路径
 * @param {object} body - 请求体
 * @param {function} onToken - 收到 token 事件回调 (content: string)
 * @param {function} onToolStart - 工具开始回调 (tool: string, args: object)
 * @param {function} onToolEnd - 工具结束回调 (tool: string, result: string)
 * @param {function} onDone - 完成回调 (messageId: number)
 * @param {function} onError - 错误回调 (error: Error)
 */
async function ssePost(path, body, { onToken, onToolStart, onToolEnd, onDone, onError }) {
    const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!res.ok) {
        const text = await res.text();
        if (onError) onError(new Error(`API ${res.status}: ${text}`));
        return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = '';
        for (const line of lines) {
            if (line.startsWith('event: ')) {
                currentEvent = line.slice(7);
            } else if (line.startsWith('data: ')) {
                const dataStr = line.slice(6);
                try {
                    const data = JSON.parse(dataStr);
                    switch (currentEvent) {
                        case 'token':
                            if (onToken) onToken(data.content);
                            break;
                        case 'tool_start':
                            if (onToolStart) onToolStart(data.tool, data.args);
                            break;
                        case 'tool_end':
                            if (onToolEnd) onToolEnd(data.tool, data.result);
                            break;
                        case 'done':
                            if (onDone) onDone(data.message_id);
                            break;
                    }
                } catch (e) {
                    // 忽略解析错误
                }
                currentEvent = '';
            }
        }
    }
}
