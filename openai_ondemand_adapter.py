from flask import Flask, request, Response, jsonify
import requests
import uuid
import time
import json
import threading
import logging
import os

# ====== 读取 Huggingface Secret 配置的私有key =======
# 用于保护此代理服务本身，防止未授权访问
PRIVATE_KEY = os.environ.get("PRIVATE_KEY", "") # 如果在Huggingface Spaces运行，这里会读取Secrets
SAFE_HEADERS = ["Authorization", "X-API-KEY"] # 允许传递私有key的请求头

# 全局接口访问权限检查
def check_private_key():
    # 根路径和favicon通常不需要鉴权
    if request.path in ["/", "/favicon.ico"]:
        return None # 显式返回 None 表示通过

    key_from_header = None
    for header_name in SAFE_HEADERS:
        key_from_header = request.headers.get(header_name)
        if key_from_header:
            if header_name == "Authorization" and key_from_header.startswith("Bearer "):
                key_from_header = key_from_header[len("Bearer "):].strip()
            break
    
    if not PRIVATE_KEY: # 如果没有设置 PRIVATE_KEY，则不进行鉴权 (方便本地测试)
        logging.warning("PRIVATE_KEY 未设置，服务将不进行鉴权！")
        return None

    if not key_from_header or key_from_header != PRIVATE_KEY:
        logging.warning(f"未授权访问尝试: Path={request.path}, IP={request.remote_addr}, Key Provided='{key_from_header[:10]}...'")
        return jsonify({"error": "Unauthorized. Correct 'Authorization: Bearer <PRIVATE_KEY>' or 'X-API-KEY: <PRIVATE_KEY>' header is required."}), 401
    return None # 鉴权通过

# 应用所有API鉴权
app = Flask(__name__)
app.before_request(check_private_key)

# ========== OnDemand API KEY池（从环境变量读取，每行一个KEY，用逗号分隔）==========
ONDEMAND_APIKEYS_STR = os.environ.get("ONDEMAND_APIKEYS", "")
ONDEMAND_APIKEYS = [key.strip() for key in ONDEMAND_APIKEYS_STR.split(',') if key.strip()]

BAD_KEY_RETRY_INTERVAL = 600  # 标记为坏的KEY的重试间隔（秒），例如10分钟
# SESSION_TIMEOUT 已移除，因为我们现在每次都用新会话

# ========== OnDemand模型映射 ==========
# 将 OpenAI 风格的模型名称映射到 OnDemand 服务的 endpointId
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
    "gemini-2.0-flash": "predefined-gemini-2.0-flash",
}
DEFAULT_ONDEMAND_MODEL = "predefined-openai-gpt4o"
# ==========================================

class KeyManager:
    """管理 OnDemand API 密钥池"""
    def __init__(self, key_list):
        self.key_list = list(key_list) # 存储可用的API密钥
        self.lock = threading.Lock()   # 线程锁，用于同步访问密钥状态
        # 存储每个密钥的状态：是否被标记为“坏的”以及标记的时间戳
        self.key_status = {key: {"bad": False, "bad_ts": None} for key in self.key_list}
        self.idx = 0 # 用于轮询密钥的索引

    def display_key(self, key):
        """返回部分隐藏的密钥，用于日志输出"""
        if not key or len(key) < 10:
            return "INVALID_KEY_FORMAT"
        return f"{key[:6]}...{key[-4:]}"

    def get(self):
        """获取一个可用的API密钥"""
        with self.lock:
            if not self.key_list: # 如果密钥池为空
                logging.error("【KeyManager】API密钥池为空！无法提供密钥。")
                raise ValueError("API key pool is empty.")

            now = time.time()
            num_keys = len(self.key_list)

            for i in range(num_keys): # 尝试遍历所有密钥最多一次
                current_key_candidate = self.key_list[self.idx]
                self.idx = (self.idx + 1) % num_keys # 移动到下一个密钥，循环使用

                status = self.key_status[current_key_candidate]

                if not status["bad"]: # 如果密钥状态良好
                    logging.info(f"【KeyManager】选择API KEY: {self.display_key(current_key_candidate)} [状态：正常]")
                    return current_key_candidate
                
                # 如果密钥被标记为坏的，检查是否已达到重试时间
                if status["bad_ts"] and (now - status["bad_ts"] >= BAD_KEY_RETRY_INTERVAL):
                    logging.info(f"【KeyManager】API KEY: {self.display_key(current_key_candidate)} 达到重试周期，恢复为正常。")
                    status["bad"] = False
                    status["bad_ts"] = None
                    return current_key_candidate
            
            # 如果所有密钥都被标记为坏的，并且都未达到重试时间
            # 强制重置所有密钥状态并返回第一个，这是一种降级策略
            logging.warning("【KeyManager】所有API KEY均被标记为不良且未到重试时间。将强制重置所有KEY状态并尝试第一个。")
            for key_to_reset in self.key_list:
                self.key_status[key_to_reset]["bad"] = False
                self.key_status[key_to_reset]["bad_ts"] = None
            self.idx = 0
            if self.key_list: # 再次检查，以防在极小概率下key_list变空
                 selected_key = self.key_list[0]
                 logging.info(f"【KeyManager】强制选择API KEY: {self.display_key(selected_key)} [状态：强制重试]")
                 return selected_key
            else: # 理论上不应该到这里，因为前面有检查
                logging.error("【KeyManager】在强制重试逻辑中发现密钥池为空！")
                raise ValueError("API key pool became empty during forced retry logic.")


    def mark_bad(self, key):
        """将指定的API密钥标记为“坏的”"""
        with self.lock:
            if key in self.key_status and not self.key_status[key]["bad"]:
                logging.warning(f"【KeyManager】禁用API KEY: {self.display_key(key)}。将在 {BAD_KEY_RETRY_INTERVAL // 60} 分钟后自动重试。")
                self.key_status[key]["bad"] = True
                self.key_status[key]["bad_ts"] = time.time()

# 初始化 KeyManager
if not ONDEMAND_APIKEYS:
    logging.warning("【启动警告】ONDEMAND_APIKEYS 环境变量未设置或为空。服务可能无法正常工作。")
keymgr = KeyManager(ONDEMAND_APIKEYS)

ONDEMAND_API_BASE = "https://api.on-demand.io/chat/v1" # OnDemand API 的基础URL

def get_endpoint_id(openai_model_name):
    """根据用户提供的OpenAI模型名称，从MODEL_MAP中查找对应的OnDemand endpointId"""
    normalized_model_name = str(openai_model_name or "").lower().replace(" ", "")
    return MODEL_MAP.get(normalized_model_name, DEFAULT_ONDEMAND_MODEL)

def create_session(apikey, external_user_id=None, plugin_ids=None):
    """
    向 OnDemand API 创建一个新的会话。
    :param apikey: OnDemand API 密钥。
    :param external_user_id: 可选，外部用户ID。
    :param plugin_ids: 可选，插件ID列表。
    :return: 新创建的会话ID。
    :raises: requests.HTTPError 如果API调用失败。
    """
    url = f"{ONDEMAND_API_BASE}/sessions"
    payload = {"externalUserId": external_user_id or str(uuid.uuid4())} # 如果未提供，则生成UUID
    if plugin_ids is not None: # 通常聊天场景可能不需要插件
        payload["pluginIds"] = plugin_ids
    
    headers = {"apikey": apikey, "Content-Type": "application/json"}
    
    logging.info(f"【OnDemand】尝试创建新会话... URL: {url}, Key: {keymgr.display_key(apikey)}")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20) # 设置超时
        resp.raise_for_status() # 如果状态码不是2xx，则抛出HTTPError
        session_id = resp.json()["data"]["id"]
        logging.info(f"【OnDemand】新会话创建成功: {session_id}, Key: {keymgr.display_key(apikey)}")
        return session_id
    except requests.exceptions.Timeout:
        logging.error(f"【OnDemand】创建会话超时。URL: {url}, Key: {keymgr.display_key(apikey)}")
        raise
    except requests.exceptions.RequestException as e:
        logging.error(f"【OnDemand】创建会话失败。URL: {url}, Key: {keymgr.display_key(apikey)}, 错误: {e}, 响应: {e.response.text if e.response else 'N/A'}")
        raise

def format_openai_sse_delta(chunk_data_dict):
    """将数据块格式化为 OpenAI SSE (Server-Sent Events) 流格式"""
    return f"data: {json.dumps(chunk_data_dict, ensure_ascii=False)}\n\n"

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """处理聊天补全请求，模拟 OpenAI /v1/chat/completions 接口"""
    try:
        request_data = request.json
    except Exception as e:
        logging.warning(f"无法解析请求JSON: {e}")
        return jsonify({"error": "Invalid JSON in request body."}), 400

    if not request_data or "messages" not in request_data:
        return jsonify({"error": "Request body must be JSON and include a 'messages' field."}), 400

    messages = request_data["messages"]
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "'messages' must be a non-empty list."}), 400

    openai_model_name = request_data.get("model", "gpt-4o") # 默认为 gpt-4o
    target_endpoint_id = get_endpoint_id(openai_model_name)
    is_stream_request = bool(request_data.get("stream", False))

    # --- 构造发送给 OnDemand 的 query 字符串 ---
    # 将整个对话历史格式化为一个字符串
    # 格式: "Role: Content\nRole: Content..."
    # 您可能需要根据 OnDemand API 的具体要求调整此格式
    formatted_query_parts = []
    for msg in messages:
        role = msg.get("role", "user").strip().capitalize()
        content = msg.get("content", "").strip()
        if not content: # 跳过空内容的消息
            continue
        formatted_query_parts.append(f"<{role}>: {content}")
    
    if not formatted_query_parts:
        return jsonify({"error": "No valid content found in 'messages'."}), 400
    sys_prompt = "The assistant is Claude, created by Anthropic. Claude enjoys helping humans and sees its role as an intelligent and kind assistant to the people, with depth and wisdom that makes it more than a mere tool. Claude avoids writing lists, but if it does need to write a list, Claude focuses on key info instead of trying to be comprehensive. If Claude can answer the human in 1-3 sentences or a short paragraph, it does. If Claude can write a natural language list of a few comma separated items instead of a numbered or bullet-pointed list, it does so. Claude tries to stay focused and share fewer, high quality examples or ideas rather than many.\nClaude always responds to the person in the language they use or request. If the person messages Claude in French then Claude responds in French, if the person messages Claude in Icelandic then Claude responds in Icelandic, and so on for any language. Claude is fluent in a wide variety of world languages.\nClaude is now being connected with a person."
    final_query_to_ondemand = sys_prompt + "\n\n".join(formatted_query_parts) + "\n\n<Assistant> :"
    
    # --- 结束构造 query ---

    # 内部函数，用于封装实际的API调用逻辑，方便重试和密钥管理
    def attempt_ondemand_request(current_apikey, current_session_id):
        # 这个函数会被 with_valid_key_and_session 调用
        # current_apikey 和 current_session_id 由 with_valid_key_and_session 提供
        
        # 根据是否流式请求，调用不同的处理函数
        if is_stream_request:
            return handle_stream_request(current_apikey, current_session_id, final_query_to_ondemand, target_endpoint_id, openai_model_name)
        else:
            return handle_non_stream_request(current_apikey, current_session_id, final_query_to_ondemand, target_endpoint_id, openai_model_name)

    # 装饰器/高阶函数，用于管理API密钥获取、会话创建和重试逻辑
    def with_valid_key_and_session(action_func):
        max_retries = len(ONDEMAND_APIKEYS) * 2 if ONDEMAND_APIKEYS else 1 # 每个key最多尝试2次
        retries_count = 0
        last_exception_seen = None

        while retries_count < max_retries:
            selected_apikey = None
            try:
                selected_apikey = keymgr.get() # 从KeyManager获取一个API密钥
                
                # 每次请求都创建一个新的OnDemand会话
                logging.info(f"【请求处理】使用 API Key: {keymgr.display_key(selected_apikey)}，准备创建新会话...")
                ondemand_session_id = create_session(selected_apikey) # 创建新会话
                
                # 执行实际的请求操作 (流式或非流式)
                return action_func(selected_apikey, ondemand_session_id)

            except ValueError as ve: # KeyManager中没有key了
                logging.critical(f"【请求处理】KeyManager 错误: {ve}")
                last_exception_seen = ve
                break # 无法获取密钥，直接中断
            except requests.HTTPError as http_err: # 包括 create_session 或 query API 的 HTTP 错误
                last_exception_seen = http_err
                response = http_err.response
                logging.warning(f"【请求处理】HTTP 错误发生。状态码: {response.status_code if response else 'N/A'}, Key: {keymgr.display_key(selected_apikey) if selected_apikey else 'N/A'}")
                if selected_apikey and response is not None:
                    # 根据错误码判断是否将Key标记为坏的
                    # 401 (Unauthorized), 403 (Forbidden), 429 (Too Many Requests) 通常意味着Key有问题或达到限额
                    if response.status_code in (401, 403, 429):
                        keymgr.mark_bad(selected_apikey)
                    # 某些5xx错误也可能与特定Key相关，或者只是服务端临时问题
                    # elif response.status_code >= 500:
                    #     keymgr.mark_bad(selected_apikey) # 谨慎处理5xx，也可能标记
                retries_count += 1
                logging.info(f"【请求处理】尝试次数: {retries_count}/{max_retries}. 等待片刻后重试...")
                time.sleep(1) # 简单等待1秒后重试
                continue
            except requests.exceptions.Timeout:
                last_exception_seen = "Request timed out."
                logging.warning(f"【请求处理】请求超时。Key: {keymgr.display_key(selected_apikey) if selected_apikey else 'N/A'}")
                if selected_apikey:
                    keymgr.mark_bad(selected_apikey) # 超时也可能标记Key
                retries_count += 1
                logging.info(f"【请求处理】尝试次数: {retries_count}/{max_retries}. 等待片刻后重试...")
                time.sleep(1)
                continue
            except Exception as e: # 其他所有Python异常
                last_exception_seen = e
                logging.error(f"【请求处理】发生意外的严重错误: {e}", exc_info=True)
                if selected_apikey:
                    keymgr.mark_bad(selected_apikey) # 发生未知严重错误时，也标记当前Key
                retries_count += 1 # 增加重试计数，避免死循环
                # 对于非常严重的未知错误，可能选择直接中断而不是继续重试
                # break 
        
        # 如果所有重试都失败了
        error_message = "All attempts to process the request failed after multiple retries."
        if last_exception_seen:
            error_message += f" Last known error: {str(last_exception_seen)}"
        logging.error(error_message)
        return jsonify({"error": "Failed to process request with OnDemand service after multiple retries. Please check service status or API keys."}), 503

    return with_valid_key_and_session(attempt_ondemand_request)


def handle_stream_request(apikey, session_id, query_str, endpoint_id, openai_model_name_for_response):
    """处理流式聊天补全请求"""
    def generate_stream_chunks():
        url = f"{ONDEMAND_API_BASE}/sessions/{session_id}/query"
        payload = {
            "query": query_str,
            "endpointId": endpoint_id,
            "pluginIds": [], # 根据需要，通常聊天为空
            "responseMode": "stream"
        }
        headers = {
            "apikey": apikey, 
            "Content-Type": "application/json", 
            "Accept": "text/event-stream" # 指示服务器发送SSE
        }
        
        logging.info(f"【流式请求】发送到 OnDemand: Session={session_id}, Endpoint={endpoint_id}, Key={keymgr.display_key(apikey)}")
        # logging.debug(f"【流式请求】Payload Query (first 200 chars): {query_str[:200]}...")

        try:
            with requests.post(url, json=payload, headers=headers, stream=True, timeout=180) as resp: # 流式请求超时可以设置长一些
                if resp.status_code != 200:
                    error_text = resp.text # 尝试读取错误响应体
                    logging.error(f"【OnDemand流错误】请求失败。状态码: {resp.status_code}, Session: {session_id}, 响应: {error_text[:500]}")
                    # 在流中产生一个错误事件
                    yield format_openai_sse_delta({
                        "error": {
                            "message": f"OnDemand API Error (Stream Init): {resp.status_code} - {error_text[:200]}",
                            "type": "on_demand_api_error",
                            "code": resp.status_code
                        }
                    })
                    yield "data: [DONE]\n\n" # 确保流结束
                    return # 提前退出生成器

                first_chunk_sent = False
                for line_bytes in resp.iter_lines(): # 按行迭代响应
                    if not line_bytes: # 跳过空行 (SSE中的keep-alive)
                        continue
                    
                    line_str = line_bytes.decode("utf-8")

                    if line_str.startswith("data:"):
                        data_part = line_str[len("data:"):].strip()
                        
                        if data_part == "[DONE]":
                            logging.info(f"【OnDemand流】接收到 [DONE] 信号。Session: {session_id}")
                            yield "data: [DONE]\n\n"
                            break 
                        elif data_part.startswith("[ERROR]:"):
                            error_json_str = data_part[len("[ERROR]:"):].strip()
                            logging.warning(f"【OnDemand流】接收到错误事件: {error_json_str}。Session: {session_id}")
                            try:
                                error_obj = json.loads(error_json_str)
                                yield format_openai_sse_delta({"error": error_obj})
                            except json.JSONDecodeError:
                                yield format_openai_sse_delta({"error": {"message": error_json_str, "type": "on_demand_stream_error_format"}})
                            yield "data: [DONE]\n\n" # 错误后也发送DONE
                            break
                        else:
                            try:
                                event_data = json.loads(data_part)
                            except json.JSONDecodeError:
                                logging.warning(f"【OnDemand流】无法解析JSON数据块: {data_part[:100]}... Session: {session_id}")
                                continue # 跳过无法解析的块

                            # 假设OnDemand流式响应中，'fulfillment'事件包含文本块
                            if event_data.get("eventType") == "fulfillment":
                                delta_content = event_data.get("answer", "") # 获取文本增量
                                if delta_content is None: delta_content = "" # 确保是字符串
                                
                                choice_delta = {}
                                if not first_chunk_sent: # 第一个有效数据块
                                    choice_delta["role"] = "assistant"
                                    choice_delta["content"] = delta_content
                                    first_chunk_sent = True
                                else:
                                    choice_delta["content"] = delta_content
                                
                                if not choice_delta.get("content") and not choice_delta.get("role"): # 避免发送空delta
                                    continue

                                openai_chunk = {
                                    "id": "chatcmpl-" + str(uuid.uuid4())[:12], # 更长的随机ID
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": openai_model_name_for_response,
                                    "choices": [{
                                        "delta": choice_delta,
                                        "index": 0,
                                        "finish_reason": None # 流式传输中，finish_reason通常在最后一块或[DONE]后确定
                                    }]
                                }
                                yield format_openai_sse_delta(openai_chunk)
                
                # 确保如果循环正常结束（没有break且没有收到[DONE]），也发送一个[DONE]
                # 但通常OnDemand API应该自己发送[DONE]
                if not line_str.endswith("data: [DONE]"): # 简易检查
                     logging.info(f"【OnDemand流】流迭代完成，补充发送 [DONE]。Session: {session_id}")
                     yield "data: [DONE]\n\n"

        except requests.exceptions.RequestException as e:
            logging.error(f"【OnDemand流】请求过程中发生网络或请求异常: {e}, Session: {session_id}", exc_info=True)
            yield format_openai_sse_delta({
                "error": {
                    "message": f"Network or request error during streaming: {str(e)}",
                    "type": "streaming_request_exception"
                }
            })
            yield "data: [DONE]\n\n"
        except Exception as e:
            logging.error(f"【OnDemand流】处理流时发生未知错误: {e}, Session: {session_id}", exc_info=True)
            yield format_openai_sse_delta({
                "error": {
                    "message": f"Unknown error during streaming: {str(e)}",
                    "type": "unknown_streaming_error"
                }
            })
            yield "data: [DONE]\n\n"

    return Response(generate_stream_chunks(), content_type='text/event-stream')


def handle_non_stream_request(apikey, session_id, query_str, endpoint_id, openai_model_name_for_response):
    """处理非流式聊天补全请求"""
    url = f"{ONDEMAND_API_BASE}/sessions/{session_id}/query"
    payload = {
        "query": query_str,
        "endpointId": endpoint_id,
        "pluginIds": [],
        "responseMode": "sync" # 同步模式
    }
    headers = {"apikey": apikey, "Content-Type": "application/json"}

    logging.info(f"【同步请求】发送到 OnDemand: Session={session_id}, Endpoint={endpoint_id}, Key={keymgr.display_key(apikey)}")
    # logging.debug(f"【同步请求】Payload Query (first 200 chars): {query_str[:200]}...")
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120) # 同步请求超时
        resp.raise_for_status() # 检查HTTP错误

        response_json = resp.json()
        # 验证响应结构，假设成功时 "data.answer" 包含回复文本
        if "data" not in response_json or "answer" not in response_json["data"]:
            logging.error(f"【OnDemand同步错误】响应格式不符合预期。Session: {session_id}, 响应: {str(response_json)[:500]}")
            raise ValueError("OnDemand API sync response missing 'data.answer' field.")
        
        ai_response_content = response_json["data"]["answer"]
        if ai_response_content is None: ai_response_content = "" # 确保是字符串

        # 构造OpenAI格式的响应
        openai_response_obj = {
            "id": "chatcmpl-" + str(uuid.uuid4())[:12],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": openai_model_name_for_response,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ai_response_content
                    },
                    "finish_reason": "stop" # 同步模式通常意味着完成
                }
            ],
            "usage": { # OnDemand可能不提供usage，这里留空或估算
                # "prompt_tokens": estimate_tokens(query_str),
                # "completion_tokens": estimate_tokens(ai_response_content),
                # "total_tokens": estimate_tokens(query_str) + estimate_tokens(ai_response_content)
            }
        }
        return jsonify(openai_response_obj)

    except requests.exceptions.Timeout as e:
        logging.error(f"【OnDemand同步错误】请求超时。Session: {session_id}, Key: {keymgr.display_key(apikey)}")
        # 此处异常会被 with_valid_key_and_session 捕获并处理重试或返回错误
        raise 
    except requests.exceptions.RequestException as e: # 包括HTTPError
        logging.error(f"【OnDemand同步错误】请求失败。Session: {session_id}, Key: {keymgr.display_key(apikey)}, 错误: {e}, 响应: {e.response.text[:500] if e.response else 'N/A'}")
        raise
    except (ValueError, KeyError, json.JSONDecodeError) as e: # 解析响应或结构错误
        logging.error(f"【OnDemand同步错误】处理响应时出错。Session: {session_id}, 错误: {e}", exc_info=True)
        # 包装成一个可以被上层理解的错误，或者直接让上层HTTPError处理
        raise requests.HTTPError(f"Error processing OnDemand sync response: {e}", response=resp if 'resp' in locals() else None)


@app.route("/v1/models", methods=["GET"])
def list_models():
    """返回此代理支持的模型列表，模拟 OpenAI /v1/models 接口"""
    model_objects = []
    for model_key_alias in MODEL_MAP.keys():
        model_objects.append({
            "id": model_key_alias, # 用户请求时使用的模型名
            "object": "model",
            "created": int(time.time()), # 可以用一个固定的时间戳或动态生成
            "owned_by": "ondemand-proxy" # 指示这些模型条目由代理提供
        })
    # 如果有默认模型且不在MODEL_MAP的key中，也可以考虑加入
    # if DEFAULT_ONDEMAND_MODEL not in [m["id"] for m in model_objects]:
    # (这取决于DEFAULT_ONDEMAND_MODEL是否也应该作为用户可选的模型ID)

    return jsonify({
        "object": "list",
        "data": model_objects
    })

@app.route("/", methods=["GET"])
def health_check():
    """简单的健康检查端点或首页"""
    num_keys = len(ONDEMAND_APIKEYS)
    key_status_summary = {keymgr.display_key(k): ("OK" if not v["bad"] else f"BAD (since {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(v['bad_ts'])) if v['bad_ts'] else 'N/A'})") for k, v in keymgr.key_status.items()}
    
    return jsonify({
        "status": "ok",
        "message": "OnDemand API Proxy is running.",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        "ondemand_api_keys_loaded": num_keys,
        "ondemand_api_key_pool_status": key_status_summary if num_keys > 0 else "No keys loaded.",
        "model_mapping_enabled": True,
        "default_on_demand_model": DEFAULT_ONDEMAND_MODEL,
        "available_models_via_proxy": list(MODEL_MAP.keys())
    }), 200


if __name__ == "__main__":
    log_format = '[%(asctime)s] %(levelname)s in %(module)s (%(funcName)s): %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)

    if not PRIVATE_KEY:
        logging.warning("****************************************************************")
        logging.warning("* WARNING: PRIVATE_KEY environment variable is not set.        *")
        logging.warning("* The proxy service will be UNSECURED and open to anyone.      *")
        logging.warning("* For production, set PRIVATE_KEY to a strong secret value.    *")
        logging.warning("****************************************************************")

    if not ONDEMAND_APIKEYS:
        logging.warning("****************************************************************")
        logging.warning("* WARNING: ONDEMAND_APIKEYS environment variable is not set    *")
        logging.warning("* or is empty. The proxy will not be able to connect to        *")
        logging.warning("* the OnDemand service.                                        *")
        logging.warning("****************************************************************")
    else:
        logging.info(f"======== OnDemand API KEY 池数量: {len(ONDEMAND_APIKEYS)} ========")
        for i, key_val in enumerate(ONDEMAND_APIKEYS):
            logging.info(f"  Key [{i+1}]: {keymgr.display_key(key_val)}")
    
    logging.info(f"======== 默认 OnDemand 模型 Endpoint ID: {DEFAULT_ONDEMAND_MODEL} ========")
    logging.info(f"======== 模型映射表 (User Model -> OnDemand Endpoint ID):")
    for user_model, od_endpoint in MODEL_MAP.items():
        logging.info(f"  '{user_model}' -> '{od_endpoint}'")

    # 从环境变量读取端口，默认为7860
    port = int(os.environ.get("PORT", 7860))
    # 对于生产环境，debug通常应为False
    # 在HuggingFace Spaces等环境中，它们通常会处理HTTPS，所以本地运行HTTP即可
    app.run(host="0.0.0.0", port=port, debug=False)
