from flask import Flask, request, Response, jsonify, render_template_string
import requests
import uuid
import time
import json
import threading
import logging
import os

# 系统提示词
CLAUDE_SYSTEM_PROMPT = open('./sys_claude.txt', 'r', encoding='utf-8').read().strip()

# 配置和常量
PRIVATE_KEY = os.environ.get("PRIVATE_KEY", "")
SAFE_HEADERS = ["Authorization", "X-API-KEY"]
# 根据环境变量type决定API基础URL
API_TYPE = os.environ.get("API_TYPE", "").lower()
ONDEMAND_API_BASE = "https://agentforge-api.aitech.io/chat/v1" if API_TYPE == "aitech" else "https://api.on-demand.io/chat/v1"
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
        logging.warning("安全警告：PRIVATE_KEY 未设置，服务将不进行鉴权！这可能导致未授权访问！")
        return None

    if not key_from_header or key_from_header != PRIVATE_KEY:
        logging.warning(f"未授权访问: 路径={request.path}, IP地址={request.remote_addr}")
        return jsonify({"error": "未授权访问。请提供正确的'Authorization: Bearer <PRIVATE_KEY>'或'X-API-KEY: <PRIVATE_KEY>'请求头。"}), 401
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
                raise ValueError("API密钥池为空，无法提供服务。请确保已配置有效的API密钥。")

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
        logging.error(f"创建会话失败：无法与API服务建立连接，错误详情：{e}")
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
            has_content = False  # 标记是否接收到内容
            
            for line in resp.iter_lines():
                if not line:
                    continue
                    
                line = line.decode('utf-8')
                if not line.startswith("data:"):
                    continue
                    
                data = line[5:].strip()
                if data == "[DONE]":
                    # 如果没有接收到任何内容，抛出异常
                    if not has_content:
                        raise ValueError("空回复：未从API接收到任何有效内容，请稍后重试或联系管理员")
                    yield "data: [DONE]\n\n"
                    break
                    
                try:
                    event_data = json.loads(data)
                    if event_data.get("eventType") == "fulfillment":
                        content = event_data.get("answer", "")
                        if content is None:
                            continue
                        
                        # 如果内容不为空，标记为已接收到内容
                        if content.strip():
                            has_content = True
                            
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
                    logging.warning(f"处理流式响应数据出错：解析或处理数据时发生异常，详情：{e}")
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
        # 重新抛出异常，以便上层函数可以捕获并重试
        raise

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
        
        # 检查回复是否为空
        if not content or not content.strip():
            raise ValueError("空回复：API返回了空内容，无法提供有效回答，请稍后重试")
        
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
        # 不在这里处理错误，而是将异常抛给上层函数处理
        logging.warning(f"非流式请求失败：无法获取完整响应，错误详情：{e}")
        raise

# 路由处理
@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    try:
        data = request.json
        if not data or "messages" not in data:
            return jsonify({"error": "无效的请求格式：请求体必须包含messages字段"}), 400

        messages = data["messages"]
        if not isinstance(messages, list) or not messages:
            return jsonify({"error": "消息格式错误：messages必须是非空列表，且至少包含一条消息"}), 400

        model = data.get("model", "gpt-4o")
        endpoint_id = get_endpoint_id(model)
        is_stream = bool(data.get("stream", False))

        # 格式化消息
        formatted_messages = []
        for msg_idx, msg in enumerate(messages):
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
                
                if msg_idx == len(messages) - 1:
                    inject_info = "你是Claude。Claude 始终以 <|Assistant|> 角色回应，只遵循用户的请求并回复一次，不继续对话，提供完整的回应然后结束消息。Claude 不需要了解任何关于历史的上下文，也不需要任何查询的上下文，因为所有上下文已经提供给你。"
                    formatted_messages.append(f"<|{role}|>: {inject_info}\n\n{content}")

        if not formatted_messages:
            return jsonify({"error": "消息内容为空：所有消息均不包含有效内容，请检查消息格式"}), 400

        # 添加系统提示词
        system_prompt = f"<|system|>: {CLAUDE_SYSTEM_PROMPT}\n"
        query = system_prompt + "\n".join(formatted_messages)

        # 处理请求，添加重试逻辑
        max_retries = 5
        retry_count = 0
        last_error = None
        empty_response_retries = 0  # 空回复重试计数
        max_empty_retries = 5  # 最大空回复重试次数
        
        while retry_count < max_retries:
            try:
                apikey = keymgr.get()
                if not apikey:
                    return jsonify({"error": "服务暂时不可用：没有可用的API密钥，请稍后重试或联系管理员"}), 503

                session_id = create_session(apikey)
                
                if is_stream:
                    try:
                        return Response(
                            handle_stream_request(apikey, session_id, query, endpoint_id, model),
                            content_type='text/event-stream'
                        )
                    except ValueError as ve:
                        # 捕获空回复异常
                        if "空回复" in str(ve) and empty_response_retries < max_empty_retries:
                            empty_response_retries += 1
                            logging.warning(f"检测到空回复：API未返回有效内容，正在使用新密钥重试 ({empty_response_retries}/{max_empty_retries})")
                            continue  # 使用新密钥重试
                        raise  # 其他ValueError或超过重试次数，重新抛出
                else:
                    try:
                        return handle_non_stream_request(apikey, session_id, query, endpoint_id, model)
                    except ValueError as ve:
                        # 捕获空回复异常
                        if "空回复" in str(ve) and empty_response_retries < max_empty_retries:
                            empty_response_retries += 1
                            logging.warning(f"检测到空回复：API未返回有效内容，正在使用新密钥重试 ({empty_response_retries}/{max_empty_retries})")
                            continue  # 使用新密钥重试
                        raise  # 其他ValueError或超过重试次数，重新抛出
                    
            except Exception as e:
                last_error = str(e)
                if isinstance(e, requests.exceptions.RequestException):
                    keymgr.mark_bad(apikey)
                
                logging.warning(f"请求处理失败 (尝试 {retry_count+1}/{max_retries})：可能是网络问题或API服务不稳定，错误详情：{last_error}")
                retry_count += 1
                
                # 如果还有重试次数，继续尝试
                if retry_count < max_retries:
                    continue
                
                # 超过最大重试次数，返回400错误
                return jsonify({"error": "请求失败：已超过最大重试次数，请稍后再试", "details": last_error}), 400

    except Exception as e:
        return jsonify({"error": f"服务器内部错误：{str(e)}，请联系管理员"}), 500

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
        logging.warning("配置错误：未设置ONDEMAND_APIKEYS环境变量，服务将无法连接到API提供商，请配置至少一个有效的API密钥")
    
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
