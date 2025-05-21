from flask import Flask, request, Response, jsonify, render_template_string
import requests
import uuid
import time
import json
import threading
import logging
import os

# 系统提示词
CLAUDE_SYSTEM_PROMPT = """你是一个由Anthropic创建的AI助手Claude,你将使用中文与用户进行对话。请记住:

1. 直接用中文回答问题,避免不必要的开场白
2. 保持友好专业的态度,提供有深度的见解
3. 如果问题模糊,可以适当提出澄清性的问题
4. 答案应该简洁明了,避免冗长
5. 优先提供一个明确的建议,而不是列举多个选项
6. 对于专业领域的问题,提供准确和最新的信息
7. 如果不确定某个信息,要明确指出
8. 使用markdown格式来突出重要内容和代码
9. 保持谨慎和理性,避免有害或误导性的回答
10. 积极参与对话但不要过度热情

重要说明：你只需要回复一次用户的问题，不要继续对话或，提供完整的回答后结束。"""

# 配置和常量
PRIVATE_KEY = os.environ.get("PRIVATE_KEY", "")
SAFE_HEADERS = ["Authorization", "X-API-KEY"]
ONDEMAND_API_BASE = "https://api.on-demand.io/chat/v1"
BAD_KEY_RETRY_INTERVAL = 600
DEFAULT_ONDEMAND_MODEL = "predefined-openai-gpt4o"

# 模型映射
MODEL_MAP = {
    "gpto3-mini": "predefined-openai-gpto3-mini",
    "gpt-4o": "predefined-openai-gpt4o", 
    "gpt-4.1": "predefined-openai-gpt4.1",
    "gpt-4.1-mini": "predefined-openai-gpt4.1-mini",  
    "gpt-4.1-nano": "predefined-openai-gpt4.1-nano",
    "gpt-4o-mini": "predefined-openai-gpt4o-mini",
    "deepseek-v3": "predefined-deepseek-v3",
    "deepseek-r1": "predefined-deepseek-r1",
    "claude-3.7-sonnet": "predefined-claude-3.7-sonnet",
    "gemini-2.0-flash": "predefined-gemini-2.0-flash"
}

# 权限检查
def check_private_key():
    if request.path in ["/", "/favicon.ico"]:
        return None
    
    key_from_header = None
    for header_name in SAFE_HEADERS:
        key_from_header = request.headers.get(header_name)
        if key_from_header:
            if header_name == "Authorization" and key_from_header.startswith("Bearer "):
                key_from_header = key_from_header[len("Bearer "):].strip()
            break
    
    if not PRIVATE_KEY:
        logging.warning("PRIVATE_KEY 未设置,服务将不进行鉴权!")
        return None

    if not key_from_header or key_from_header != PRIVATE_KEY:
        logging.warning(f"未授权访问: Path={request.path}, IP={request.remote_addr}")
        return jsonify({"error": "Unauthorized. Correct 'Authorization: Bearer <PRIVATE_KEY>' or 'X-API-KEY: <PRIVATE_KEY>' header is required."}), 401
    return None

# 密钥管理
class KeyManager:
    def __init__(self, key_list):
        self.key_list = list(key_list)
        self.lock = threading.Lock()
        self.key_status = {key: {"bad": False, "bad_ts": None} for key in self.key_list}
        self.idx = 0

    def display_key(self, key):
        return f"{key[:6]}...{key[-4:]}" if key and len(key) >= 10 else "INVALID_KEY"

    def get(self):
        with self.lock:
            if not self.key_list:
                raise ValueError("API key pool is empty.")

            now = time.time()
            for _ in range(len(self.key_list)):
                key = self.key_list[self.idx]
                self.idx = (self.idx + 1) % len(self.key_list)
                status = self.key_status[key]

                if not status["bad"] or (status["bad_ts"] and now - status["bad_ts"] >= BAD_KEY_RETRY_INTERVAL):
                    status["bad"] = False
                    status["bad_ts"] = None
                    return key

            # 所有key都不可用时重置状态
            for k in self.key_list:
                self.key_status[k]["bad"] = False
                self.key_status[k]["bad_ts"] = None
            return self.key_list[0] if self.key_list else None

    def mark_bad(self, key):
        with self.lock:
            if key in self.key_status and not self.key_status[key]["bad"]:
                self.key_status[key]["bad"] = True
                self.key_status[key]["bad_ts"] = time.time()

# 初始化Flask应用
app = Flask(__name__)
app.before_request(check_private_key)

# 初始化密钥管理器
ONDEMAND_APIKEYS = [key.strip() for key in os.environ.get("ONDEMAND_APIKEYS", "").split(',') if key.strip()]
keymgr = KeyManager(ONDEMAND_APIKEYS)

# 工具函数
def get_endpoint_id(model_name):
    return MODEL_MAP.get(str(model_name or "").lower().replace(" ", ""), DEFAULT_ONDEMAND_MODEL)

def format_openai_sse_delta(data):
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

def create_session(apikey, external_user_id=None):
    url = f"{ONDEMAND_API_BASE}/sessions"
    payload = {"externalUserId": external_user_id or str(uuid.uuid4())}
    headers = {"apikey": apikey, "Content-Type": "application/json"}
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp.json()["data"]["id"]
    except Exception as e:
        logging.error(f"创建会话失败: {e}")
        raise

# 处理流式请求
def handle_stream_request(apikey, session_id, query, endpoint_id, model_name):
    url = f"{ONDEMAND_API_BASE}/sessions/{session_id}/query"
    payload = {
        "query": query,
        "endpointId": endpoint_id,
        "pluginIds": [],
        "responseMode": "stream"
    }
    headers = {
        "apikey": apikey,
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            first_chunk = True
            
            for line in resp.iter_lines():
                if not line:
                    continue
                    
                line = line.decode('utf-8')
                if not line.startswith("data:"):
                    continue
                    
                data = line[5:].strip()
                if data == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                    
                try:
                    event_data = json.loads(data)
                    if event_data.get("eventType") == "fulfillment":
                        content = event_data.get("answer", "")
                        if content is None:
                            continue
                            
                        delta = {}
                        if first_chunk:
                            delta["role"] = "assistant"
                            first_chunk = False
                        delta["content"] = content
                        
                        chunk = {
                            "id": f"chatcmpl-{str(uuid.uuid4())[:12]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{"delta": delta, "index": 0, "finish_reason": None}]
                        }
                        yield format_openai_sse_delta(chunk)
                except Exception as e:
                    logging.warning(f"处理流数据出错: {e}")
                    continue
    except Exception as e:
        error = {
            "error": {
                "message": str(e),
                "type": "stream_error",
                "code": 500
            }
        }
        yield format_openai_sse_delta(error)
        yield "data: [DONE]\n\n"

# 处理非流式请求        
def handle_non_stream_request(apikey, session_id, query, endpoint_id, model_name):
    url = f"{ONDEMAND_API_BASE}/sessions/{session_id}/query"
    payload = {
        "query": query,
        "endpointId": endpoint_id,
        "pluginIds": [],
        "responseMode": "sync"
    }
    headers = {"apikey": apikey, "Content-Type": "application/json"}
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        response_data = resp.json()
        content = response_data["data"]["answer"]
        
        return jsonify({
            "id": f"chatcmpl-{str(uuid.uuid4())[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 路由处理
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    try:
        data = request.json
        if not data or "messages" not in data:
            return jsonify({"error": "Invalid request format"}), 400

        messages = data["messages"]
        if not isinstance(messages, list) or not messages:
            return jsonify({"error": "Messages must be a non-empty list"}), 400

        model = data.get("model", "gpt-4o")
        endpoint_id = get_endpoint_id(model)
        is_stream = bool(data.get("stream", False))

        # 格式化消息
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user").strip().capitalize()
            content = msg.get("content", "")
            
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        else:
                            for k, v in item.items():
                                text_parts.append(f"{k}: {v}")
                content = "\n".join(filter(None, text_parts))
            
            if content:
                formatted_messages.append(f"<|{role}|>: {content}")

        if not formatted_messages:
            return jsonify({"error": "No valid content in messages"}), 400

        # 添加系统提示词
        query = CLAUDE_SYSTEM_PROMPT + "\n\n下面是对话历史:\n" + "\n".join(formatted_messages)

        # 处理请求
        try:
            apikey = keymgr.get()
            if not apikey:
                return jsonify({"error": "No available API keys"}), 503

            session_id = create_session(apikey)
            
            if is_stream:
                return Response(
                    handle_stream_request(apikey, session_id, query, endpoint_id, model),
                    content_type='text/event-stream'
                )
            else:
                return handle_non_stream_request(apikey, session_id, query, endpoint_id, model)
                
        except Exception as e:
            if isinstance(e, requests.exceptions.RequestException):
                keymgr.mark_bad(apikey)
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "ondemand-proxy"
        } for model_id in MODEL_MAP.keys()]
    })

@app.route("/health", methods=["GET"])
def health_check_json():
    """返回JSON格式的健康检查信息"""
    return jsonify({
        "status": "ok",
        "message": "OnDemand API Proxy is running.",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        "api_keys_loaded": len(ONDEMAND_APIKEYS),
        "key_status": {
            keymgr.display_key(k): "OK" if not v["bad"] else "BAD"
            for k, v in keymgr.key_status.items()
        },
        "available_models": list(MODEL_MAP.keys())
    })

@app.route("/", methods=["GET"])
def health_check():
    """返回HTML格式的健康检查页面"""
    # 获取当前时间
    current_time = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    
    # 获取API密钥状态
    key_status = {
        keymgr.display_key(k): "正常" if not v["bad"] else "异常"
        for k, v in keymgr.key_status.items()
    }
    
    # 获取可用模型列表
    available_models = list(MODEL_MAP.keys())
    
    # HTML模板
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>API服务</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="refresh" content="10">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            h1, h2 {
                color: #333;
            }
            .status {
                margin-bottom: 20px;
            }
            .status-ok {
                color: green;
                font-weight: bold;
            }
            .status-error {
                color: red;
                font-weight: bold;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .model-list {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .model-item {
                background-color: #f0f0f0;
                padding: 5px 10px;
                border-radius: 4px;
            }
            .refresh {
                margin-top: 20px;
            }
            .api-endpoints {
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <h1>API服务</h1>
        
        <div class="status">
            <h2>服务状态</h2>
            <p>状态: <span class="status-ok">正常运行中</span></p>
            <p>当前时间: {{ current_time }}</p>
        </div>
        
        <div class="models">
            <h2>可用模型</h2>
            <div class="model-list">
                {% for model in available_models %}
                <div class="model-item">{{ model }}</div>
                {% endfor %}
            </div>
        </div>
        
        <div class="refresh">
            <button onclick="location.reload()">手动刷新</button>
            <p><small>页面每10秒自动刷新一次</small></p>
        </div>
        
        <div class="api-info">
            <h2>API信息</h2>
            <p>健康检查JSON端点: <a href="/health">/health</a></p>
            <p>模型列表端点: <a href="/v1/models">/v1/models</a></p>
        </div>
    </body>
    </html>
    """
    
    # 渲染模板
    return render_template_string(
        html_template,
        current_time=current_time,
        api_keys_count=len(ONDEMAND_APIKEYS),
        key_status=key_status,
        available_models=available_models,
        api_base=ONDEMAND_API_BASE
    )

if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

    if not ONDEMAND_APIKEYS:
        logging.warning("未设置ONDEMAND_APIKEYS环境变量,服务可能无法正常工作")
    
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
