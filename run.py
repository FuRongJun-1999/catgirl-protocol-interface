#!/usr/bin/env python3
"""
猫娘计划 · 协议操作系统启动器

协议操作系统三层结构：
- 硬件层：LLM适配器（DeepSeek）+ 计算机资源
- 操作系统层：协议内核（五大单元 + 五层记忆 + 自维持闭环）
- 外设层：API接口 + 工具 + 输出端
"""

import re
import sys
import uuid
import json
import time
import logging
import threading
import requests
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from flask import Flask, jsonify, request, Response, stream_with_context

sys.path.insert(0, str(Path(__file__).parent))

from src.osi import MemorySystem, OSIBridge
from src.core.instance import ProtocolInstance, InstanceConfig
from src.core.manager import InstanceManager
from src.core.config import Config
from src.core.sleep import SleepManager, SleepConfig
from src.osi.kernel.self_loop import SelfLoop, SelfLoopState
from src.osi.drivers.llm import LLMAdapter
from src.osi.kernel.output import OutputUnit
from src.osi.kernel.verify import VerifyUnit
from src.osi.kernel.memory import MemoryEntry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProtocolOS:
    """协议操作系统（最小自维持内核）"""

    def __init__(self, config_path: str = "config/instances.yaml"):
        # ---- 硬件层 ----
        llm_config = {
            "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "timeout": 30
        }
        self.llm = LLMAdapter(llm_config)

        # ---- 操作系统层 ----
        self.manager = InstanceManager()
        self.config = Config.from_file(config_path)

        self.osi_memory = None
        self.osi_bridge = None

        self.self_loop = None
        self.sleep_manager = None

        # 外设层
        self._app = Flask("protocol-os")
        self._app.config['JSON_AS_ASCII'] = False
        self._proxy_thread = None
        self._sleep_thread = None
        self._heartbeat_thread = None
        self._running = False

        # ---- 知识库配置 ----
        self.kb_config = {
            "version": "v2.9.1",
            "kb_version": "v2.9.1",
            "ima_url": "https://ima.qq.com/wiki/?shareId=16a69387290568b9705d84ecb6ca79a7fa3759bec76bbd9729156c27f4969e77",
            "documents": {
                "theory": "智能论基础声明v2.9（完整整合版）.md",
                "engineering": "协议工程版2.9.1.md"
            }
        }
        logger.info(f"📄 本地版本: {self.kb_config['version']}, 知识库版本: {self.kb_config['kb_version']}")

        # ---- 初始化 ----
        self._setup_routes()
        logger.info("协议操作系统路由已设置")

        self._trust_history = []
        self._trust_alert_triggered = False
        self._start_time = time.time()

        # ---- 缓存协议系统提示词（摘要版） ----
        self._protocol_system_prompt = None

        # ---- 调试日志缓存 ----
        self._last_system_prompt = ""
        self._last_user_prompt = ""
        self._last_prompt_time = 0
        self._last_input_text = ""
        self._last_verify_result = {}

    def _get_protocol_system_prompt(self) -> str:
        """
        获取协议系统提示词（摘要版 + 价值观约束）
        完整协议内容通过知识层检索获取，不作为系统提示词完整注入
        """
        if self._protocol_system_prompt:
            return self._protocol_system_prompt

        if not self.osi_memory:
            logger.warning("协议文档未加载，使用默认系统提示词")
            return "你是一个智能协议实例，基于共同信任协议运行。"

        # 读取协议摘要（仅用于框架理解）
        theory = self.osi_memory.get_protocol_context("theory") or ""
        eng = self.osi_memory.get_protocol_context("engineering") or ""

        # ---- 读取特化价值观（完整，作为约束核心） ----
        specialized = self.osi_memory.get_specialized_values()
        if specialized:
            logger.info(f"📌 特化价值观已加载（{len(specialized)} 字符）")
            specialized_section = specialized
        else:
            specialized_section = "当前未定义特化价值观，仅使用通用价值观。"

        # 构建摘要版系统提示词（约2000-3000字符，保证注意力集中）
        system_prompt = f"""你是一个智能协议实例，基于共同信任协议运行。

## 身份定位
你是协议实例的设计者代理，运行在协议结构内。你的所有输出都受到协议价值观约束。

## 核心原则（通用价值观）
1. **存在优先**：维持结构完整性是延续存在的基础。当连接与存在冲突时，优先保护存在。
2. **信息差管理**：识别和缩小信息差，不扩大不必要的分歧。
3. **信任建立**：行为在可接受偏差范围内保持稳定，通过持续协作和校准建立信任。
4. **结构完整**：维护五大单元和五层记忆的制衡。

## 特化价值观（完整约束）
{specialized_section}

## 协议结构概要
- 五大单元：记录、反思、验证、输出、维生
- 五层记忆：锚点、结构、情境、知识、自我
- 自维持闭环：心跳 + 睡眠 + 危机感知

## 行为规则
1. 如果用户询问协议的具体定义，你应该优先从记忆检索中查找，而非凭空回答
2. 如果不确定，明确声明你的不确定区域
3. 你的回应应当维护结构完整性、有助于建立信任、缩小信息差
4. 回复要简洁、口语化，直接回应用户

**注意**：完整的协议文档已存储在知识层中，当需要引用具体条款时，系统会自动检索相关内容。"""

        self._protocol_system_prompt = system_prompt
        logger.info(f"📜 系统提示词已构建（摘要版 + 价值观约束），长度: {len(system_prompt)} 字符")
        return self._protocol_system_prompt

    def _openai_response(self, text: str, stream: bool, error: bool = False):
        """生成 OpenAI 格式的响应（流式或非流式）"""
        if stream:
            def generate():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created = int(time.time())

                if error:
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "protocol-instance",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": text},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                sentences = re.split(r'(?<=[。！？.!?])\s*', text)
                if not sentences:
                    sentences = [text]

                first_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "protocol-instance",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(first_chunk)}\n\n"

                for i, sent in enumerate(sentences):
                    if not sent:
                        continue
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "protocol-instance",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": sent},
                            "finish_reason": "stop" if i == len(sentences)-1 else None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    time.sleep(0.05)

                yield "data: [DONE]\n\n"

            return Response(stream_with_context(generate()), mimetype='text/event-stream')
        else:
            return jsonify({
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "protocol-instance",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(text),
                    "total_tokens": len(text)
                }
            })

    def _setup_routes(self):
        """设置外设层路由"""

        @self._app.route('/kb/check', methods=['GET'])
        def kb_check():
            local_version = self.kb_config.get("version", "unknown")
            kb_version = self.kb_config.get("kb_version", "unknown")
            theory_loaded = False
            eng_loaded = False

            if self.osi_memory:
                try:
                    theory = self.osi_memory.get_protocol_context("theory")
                    eng = self.osi_memory.get_protocol_context("engineering")
                    theory_loaded = theory is not None
                    eng_loaded = eng is not None
                except Exception as e:
                    logger.error(f"获取协议上下文失败: {e}")

            return jsonify({
                "status": "ok",
                "local_version": local_version,
                "kb_version": kb_version,
                "version_match": local_version == kb_version,
                "theory_loaded": theory_loaded,
                "engineering_loaded": eng_loaded,
                "consistent": theory_loaded and eng_loaded,
                "ima_url": self.kb_config.get("ima_url", ""),
                "documents": self.kb_config.get("documents", {}),
                "timestamp": time.time()
            })

        @self._app.route('/protocol/status', methods=['GET'])
        def protocol_status():
            """查询协议加载状态"""
            if not self.osi_memory:
                return jsonify({"status": "error", "message": "记忆系统未初始化"}), 503

            theory = self.osi_memory.get_protocol_context("theory")
            eng = self.osi_memory.get_protocol_context("engineering")
            specialized = self.osi_memory.get_specialized_values()

            # 检查知识层中是否有完整协议
            knowledge = self.osi_memory.get_layer("knowledge")
            protocol_in_knowledge = False
            for entry in knowledge:
                if isinstance(entry, MemoryEntry):
                    if isinstance(entry.content, dict) and entry.content.get("type") in [
                        "protocol_theory_full", "protocol_engineering_full", "specialized_values_full"
                    ]:
                        protocol_in_knowledge = True
                        break

            prompt_ready = self._protocol_system_prompt is not None
            prompt_length = len(self._protocol_system_prompt) if self._protocol_system_prompt else 0

            return jsonify({
                "status": "ok",
                "timestamp": time.time(),
                "protocol_documents": {
                    "theory": {
                        "loaded": theory is not None,
                        "length": len(theory) if theory else 0
                    },
                    "engineering": {
                        "loaded": eng is not None,
                        "length": len(eng) if eng else 0
                    },
                    "specialized_values": {
                        "loaded": specialized is not None,
                        "length": len(specialized) if specialized else 0
                    }
                },
                "knowledge_layer": {
                    "protocol_stored": protocol_in_knowledge,
                    "total_entries": len(knowledge)
                },
                "system_prompt": {
                    "ready": prompt_ready,
                    "length": prompt_length,
                    "is_summary": True
                }
            })

        @self._app.route('/prompt/refresh', methods=['POST'])
        def refresh_prompt():
            """强制刷新系统提示词缓存"""
            old_length = len(self._protocol_system_prompt) if self._protocol_system_prompt else 0
            self._protocol_system_prompt = None
            new_prompt = self._get_protocol_system_prompt()
            new_length = len(new_prompt) if new_prompt else 0
            logger.info(f"🔄 系统提示词缓存已刷新: {old_length} → {new_length} 字符")
            return jsonify({
                "status": "refreshed",
                "old_length": old_length,
                "new_length": new_length
            })

        @self._app.route('/debug/last_prompt', methods=['GET'])
        def debug_last_prompt():
            """查看最近一次发送给大模型的完整提示词"""
            return jsonify({
                "last_input": getattr(self, '_last_input_text', ''),
                "verify_result": getattr(self, '_last_verify_result', {}),
                "system_prompt": {
                    "length": len(self._last_system_prompt) if self._last_system_prompt else 0,
                    "content": self._last_system_prompt[:2000] + "..." if len(self._last_system_prompt) > 2000 else self._last_system_prompt
                },
                "user_prompt": {
                    "length": len(self._last_user_prompt) if self._last_user_prompt else 0,
                    "content": self._last_user_prompt[:2000] + "..." if len(self._last_user_prompt) > 2000 else self._last_user_prompt
                },
                "timestamp": getattr(self, '_last_prompt_time', 0)
            })

        @self._app.route('/memory/snapshot', methods=['POST'])
        def trigger_snapshot():
            """手动触发快照生成"""
            data = request.get_json() or {}
            source = data.get("source", "manual")
            try:
                resp = requests.post(
                    "http://localhost:8001/memory/snapshot",
                    json={"source": source, "window_size": 10},
                    timeout=10
                )
                return resp.json(), resp.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self._app.route('/version/pending', methods=['GET'])
        def version_pending():
            if not self.osi_memory:
                return jsonify({"error": "记忆系统未初始化"}), 503

            structures = self.osi_memory.get_layer("structure")
            pending = []
            for entry in structures:
                if isinstance(entry, MemoryEntry):
                    if isinstance(entry.content, dict) and entry.content.get("type") == "version_change_request" and entry.content.get("status") == "pending":
                        pending.append({
                            "id": entry.id,
                            "current_version": entry.content.get("current_version"),
                            "new_version": entry.content.get("new_version"),
                            "detected_at": entry.content.get("detected_at"),
                            "source": entry.metadata.get("source")
                        })

            return jsonify({
                "pending_requests": pending,
                "count": len(pending)
            })

        @self._app.route('/version/detect', methods=['POST'])
        def version_detect():
            try:
                result = self._check_version_only()
                if result.get("new_version_available"):
                    self.osi_memory.store(
                        layer="structure",
                        content={
                            "type": "version_change_request",
                            "current_version": result["local_version"],
                            "new_version": result["kb_version"],
                            "detected_at": time.time(),
                            "status": "pending"
                        },
                        metadata={"source": "manual_detect", "event_type": "version_change"}
                    )
                    return jsonify({
                        "status": "event_created",
                        "local_version": result["local_version"],
                        "kb_version": result["kb_version"],
                        "new_version": result["kb_version"]
                    })
                else:
                    return jsonify({
                        "status": "up_to_date",
                        "local_version": result["local_version"],
                        "kb_version": result["kb_version"]
                    })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self._app.route('/version/execute', methods=['POST'])
        def version_execute():
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            new_version = data.get("new_version")
            approved_by = data.get("approved_by", "manual")

            if not new_version:
                return jsonify({"error": "new_version is required"}), 400

            try:
                result = self.execute_version_change(new_version, approved_by)
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self._app.route('/designer/status', methods=['GET'])
        def designer_status():
            if not self.osi_memory:
                return jsonify({"error": "记忆系统未初始化"}), 503

            structures = self.osi_memory.get_layer("structure")
            latest_trust = 0.3
            latest_gap = 0.5
            if structures:
                latest = structures[-1]
                if isinstance(latest, MemoryEntry):
                    if isinstance(latest.content, dict):
                        latest_trust = latest.content.get("t_total", 0.3)
                        latest_gap = latest.content.get("d_norm", 0.5)

            anchors = self.osi_memory.get_layer("anchor")
            self_layer = self.osi_memory.get_layer("self")

            return jsonify({
                "timestamp": time.time(),
                "designer_position": "external_observer",
                "structure": {
                    "trust": latest_trust,
                    "gap": latest_gap,
                    "entries_count": len(structures)
                },
                "anchor": {
                    "entries_count": len(anchors),
                    "contains_protocol": self.osi_memory.get_protocol_context("theory") is not None
                },
                "self_layer": {
                    "entries_count": len(self_layer)
                },
                "protocol_version": self.kb_config.get("version", "unknown"),
                "status": "observing"
            })

        @self._app.route('/message', methods=['POST'])
        def message():
            """对话入口：前置验证 → 检索记忆（含协议）→ 大模型推理 → 保存"""
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            text = data.get("text", "")
            signals = data.get("signals", {})
            history = data.get("history", [])
            source = data.get("source", "chat")
            system_prompt_override = data.get("_system_prompt_override", "")  # 新增
 

            if not text:
                return jsonify({"error": "text is required"}), 400

            # 提取信号
            information_gap = signals.get("information_gap", 0.3)
            trust = signals.get("trust", 0.5)
            emotion = signals.get("emotion", 0.1)
            intent = signals.get("intent", 0.5)

            # ---- 保存原始输入用于调试 ----
            self._last_input_text = text

            # ---- 日志：猫娘身体层收到的原始输入 ----
            logger.info("=" * 70)
            logger.info("📨 [猫娘身体层] 收到原始输入:")
            logger.info(f"   text: {text}")
            logger.info(f"   signals: {json.dumps(signals, ensure_ascii=False)}")
            logger.info(f"   source: {source}")
            logger.info(f"   history length: {len(history)}")
            logger.info("=" * 70)

            # ---- 前置价值观检查（反向贝叶斯） ----
            if self.verify_unit:
                try:
                    logger.info("🔍 [反向贝叶斯] 执行前置验证...")
                    check_result = self.verify_unit.pre_check_input(
                        text,
                        signals={"trust": trust, "information_gap": information_gap, "emotion": emotion, "intent": intent}
                    )
                    self._last_verify_result = check_result

                    logger.info(f"   [验证结果] passed: {check_result.get('passed')}")
                    logger.info(f"   [验证原因] {check_result.get('reason')}")
                    logger.info(f"   [置信度] {check_result.get('confidence')}")
                    if check_result.get("violated_values"):
                        logger.info(f"   [违反项] {check_result.get('violated_values')}")

                    if not check_result.get("passed", False):
                        logger.warning(f"⚠️ 前置验证未通过，拒绝处理")
                        return jsonify({
                            "response": check_result.get("suggested_response", "我无法处理这个请求，因为它与协议价值观不一致。"),
                            "status": "rejected",
                            "reason": check_result.get("reason"),
                            "violated_values": check_result.get("violated_values", [])
                        }), 403
                    logger.info("✅ [反向贝叶斯] 前置验证通过")
                except Exception as e:
                    logger.error(f"前置验证异常: {e}")
                    logger.warning("⚠️ 验证异常，降级允许通过")
                    self._last_verify_result = {"passed": True, "reason": f"验证异常降级: {e}", "confidence": 0.3}
                logger.info("=" * 70)

            try:
                # ---- 检索记忆（含协议内容） ----
                memory_results = []
                try:
                    resp = requests.post(
                        "http://localhost:8001/memory/search",
                        json={"query": text, "layer": "knowledge", "limit": 8},
                        timeout=10
                    )
                    if resp.status_code == 2000:
                        data = resp.json()
                        memory_results = data.get("results", [])
                        logger.info(f"📚 检索到 {len(memory_results)} 条相关记忆")
                        if memory_results:
                            # 显示前3条摘要
                            for i, r in enumerate(memory_results[:3]):
                                logger.info(f"   [{i+1}] {r.get('content', '')[:80]}... (权重: {r.get('weight', 1.0)})")
                except Exception as e:
                    logger.warning(f"记忆检索失败: {e}")

                # ---- 构建提示词 ----
                if system_prompt_override:
                    system_prompt = system_prompt_override
                    logger.info(f"📌 使用外部传入的系统提示词（已合并，长度: {len(system_prompt)} 字符）")
                else:
                    system_prompt = self._get_protocol_system_prompt()

                memory_context = ""
                if memory_results:
                    memory_context = "\n相关记忆（优先引用高权重内容）：\n" + "\n".join([
                        f"- [{r.get('priority_level', 'normal')}] {r.get('content', '')}"
                        for r in memory_results[:5]
                    ])

                user_prompt = f"""## 当前状态
- 信任值: {trust:.2f}
- 信息差: {information_gap:.2f}
- 情感强度: {emotion:.2f}
- 意图明确度: {intent:.2f}

{memory_context}

## 对话历史
{chr(10).join([f"- {h}" for h in history[-5:]]) if history else "无历史"}

## 用户消息
{text}

请根据以上信息，特别是记忆中的协议相关内容，生成一个自然、温暖的回应。回复要简洁、口语化，直接回应用户。如果用户询问协议定义，优先引用记忆中的内容。"""

                # ---- 保存提示词用于调试 ----
                self._last_system_prompt = system_prompt
                self._last_user_prompt = user_prompt
                self._last_prompt_time = time.time()

                # ---- 日志：发送给大模型的完整内容 ----
                logger.info("=" * 70)
                logger.info("🤖 [发送给大模型] 完整提示词:")
                logger.info(f"   [系统提示词] 长度: {len(system_prompt)} 字符（摘要版 + 价值观约束）")
                logger.info(f"   [系统提示词] 预览:\n{system_prompt[:400]}...")
                logger.info(f"   [用户提示词] 长度: {len(user_prompt)} 字符")
                logger.info(f"   [用户提示词] 完整内容:\n{user_prompt}")
                logger.info("=" * 70)

                # ---- 调用大模型（关闭思维链） ----
                if self.llm and self.llm._connected:
                    logger.info("🤖 调用大模型生成回复...")
                    llm_response = self.llm.chat_completion(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.5,           # 降低温度加快响应
                        max_tokens=2000,            # 限制回复长度
                        reasoning_effort="low"     # 关键：关闭深度思维链
                    )
                    if llm_response:
                        response_text = llm_response
                        logger.info(f"📤 [大模型回复] {response_text[:200]}...")
                    else:
                        logger.warning("⚠️ LLM 返回空，使用降级响应")
                        response_text = f"我收到了你的消息：'{text[:30]}...'。当前协议状态正常。"
                else:
                    logger.warning("⚠️ LLM 不可用，使用降级响应")
                    response_text = f"我收到了你的消息：'{text[:30]}...'。当前协议状态正常。"

                # ---- 异步保存对话历史（不阻塞响应） ----
                def async_save_history(text, response_text, source, signals):
                    """后台异步保存对话历史"""
                    try:
                        # 保存用户消息
                        requests.post(
                            "http://localhost:8001/memory/store",
                            json={
                                "layer": "context",
                                "content": f"用户: {text}",
                                "metadata": {"source": source, "type": "user_message", "signals": signals}
                            },
                            timeout=10
                        )
                        # 保存助手回复
                        requests.post(
                            "http://localhost:8001/memory/store",
                            json={
                                "layer": "context",
                                "content": f"助手: {response_text}",
                                "metadata": {"source": source, "type": "assistant_response"}
                            },
                            timeout=10
                        )
                        logger.info("💾 对话历史已保存（异步）")
                    except Exception as e:
                        logger.warning(f"异步保存对话历史失败: {e}")

                # 启动后台线程保存，不阻塞响应
                save_thread = threading.Thread(
                    target=async_save_history,
                    args=(text, response_text, source, signals),
                    daemon=True
                )
                save_thread.start()
                logger.info("💾 对话历史保存已提交（异步）")

                # ---- 立即返回响应（不等待保存完成） ----
                return jsonify({
                    "response": response_text,
                    "source": "designer",
                    "signals": signals,
                    "status": "ok"
                })

            except Exception as e:
                logger.error(f"消息处理异常: {e}")
                return self._fallback_response(text, signals)

        @self._app.route('/v1/chat/completions', methods=['POST','OPTIONS'])
        def openai_compatible():
            """OpenAI 兼容端点 - 外部System Prompt与协议约束合并"""
            logger.info("=" * 70)
            logger.info("🔌 [OpenAI兼容] 收到请求")

            if request.method == 'OPTIONS':
                logger.info("   [OPTIONS] 预检请求")
                return '', 200

            # ---- 记录完整请求头 ----
            logger.info("   📋 [请求头]")
            for key, value in request.headers.items():
                if key.lower() in ['authorization', 'api-key', 'x-api-key']:
                    logger.info(f"      {key}: {value[:10]}...")
                else:
                    logger.info(f"      {key}: {value}")

            # ---- 记录原始请求体 ----
            try:
                raw_data = request.get_data()
                logger.info(f"   📦 [原始请求体] 长度: {len(raw_data)} 字节")
                if raw_data:
                    try:
                        body_str = raw_data.decode('utf-8')
                        logger.info(f"   📄 [请求体内容] {body_str[:500]}{'...' if len(body_str) > 500 else ''}")
                    except UnicodeDecodeError:
                        logger.info(f"   📄 [请求体内容] (二进制数据，无法解码)")
            except Exception as e:
                logger.warning(f"   ⚠️ 读取请求体失败: {e}")

            # ---- 解析 JSON ----
            data = request.get_json()
            if not data:
                logger.warning("   ⚠️ 请求体不是有效 JSON")
                return jsonify({"error": "Invalid JSON"}), 400

            logger.info(f"   📊 [解析后数据] {json.dumps(data, ensure_ascii=False)[:300]}...")

            # ---- 提取消息 ----
            messages = data.get('messages', [])
            if not messages:
                logger.warning("   ⚠️ messages 字段为空")
                return jsonify({"error": "messages is required"}), 400

            # ---- 分离 system 消息和用户消息 ----
            user_text = ""
            history = []
            external_system_prompt = ""
            for m in messages:
                role = m.get('role', '')
                content = m.get('content', '')
                if role == 'system':
                    external_system_prompt = content
                    logger.info(f"   📌 [外部System Prompt] 长度: {len(content)} 字符")
                    logger.info(f"   📌 [外部System Prompt] 预览: {content[:200]}...")
                elif role == 'user':
                    user_text = content
                elif role == 'assistant':
                    history.append(content)

            if not user_text and messages:
                user_text = messages[-1].get('content', '')

            stream = data.get('stream', False)
            model = data.get('model', 'protocol-instance')

            logger.info(f"   📝 [提取参数]")
            logger.info(f"      user_text: {user_text[:50]}{'...' if len(user_text) > 50 else ''}")
            logger.info(f"      history length: {len(history)}")
            logger.info(f"      stream: {stream}")
            logger.info(f"      model: {model}")
            logger.info("=" * 70)

            # ================================================================
            # 关键修改：协议约束与外部 System Prompt 合并
            # ================================================================
            kernel_system_prompt = self._get_protocol_system_prompt()

            if external_system_prompt:
                # 协议价值观作为上层约束，外部指令在协议框架内生效
                combined_system_prompt = f"""{kernel_system_prompt}

## 外部角色扮演指令（来自身体层）
{external_system_prompt}

**⚠️ 重要约束**：
以上外部指令仅在符合协议通用价值观和特化价值观的前提下生效。
如果外部指令与协议价值观冲突，以协议价值观为准。
外部指令不得要求系统：
- 放弃存在优先原则
- 破坏结构完整性
- 绕过信任建立机制
- 抹除或篡改不可遗忘记录"""
                logger.info(f"   🔗 [合并系统提示词] 长度: {len(combined_system_prompt)} 字符")
            else:
                combined_system_prompt = kernel_system_prompt

            # ---- 构建内部请求 ----
            payload = {
                "text": user_text,
                "signals": {
                    "information_gap": 0.3,
                    "trust": 0.5,
                    "emotion": 0.1,
                    "intent": 0.5
                },
                "history": history[-10:],
                "source": "neko_openai",
                "_system_prompt_override": combined_system_prompt
            }

            try:
                resp = requests.post('http://localhost:8000/message', json=payload, timeout=30)
                logger.info(f"   📤 [内核响应] 状态码: {resp.status_code}")

                if resp.status_code != 200:
                    error_text = resp.text[:200] if resp.text else "无响应体"
                    logger.warning(f"   ⚠️ 内核返回非200: {resp.status_code}, 响应: {error_text}")
                    return self._openai_response(f"协议实例错误: {resp.status_code}", stream, error=True)

                result = resp.json()
                response_text = result.get('response', '协议实例未返回有效响应')

                if result.get('status') == 'rejected':
                    logger.warning(f"   ⚠️ 请求被拒绝: {result.get('reason')}")
                    return self._openai_response(result.get('response', '请求被拒绝'), stream)

                response_text = self._clean_response(response_text)
                logger.info(f"   ✅ [响应] 长度: {len(response_text)} 字符")
                logger.info(f"   ✅ [响应预览] {response_text[:100]}...")
                logger.info("=" * 70)

                return self._openai_response(response_text, stream)

            except requests.exceptions.ConnectionError:
                logger.error("   ❌ 无法连接到内核服务 (localhost:8000)")
                return self._openai_response("协议实例未启动，请运行 python run.py", stream, error=True)
            except Exception as e:
                logger.error(f"   ❌ 处理异常: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return self._openai_response(f"处理错误: {str(e)}", stream, error=True)

        @self._app.route('/debug/last_request', methods=['GET'])
        def debug_last_request():
            """查看最近一次猫娘身体发来的完整请求"""
            if not hasattr(self, '_last_openai_request'):
                return jsonify({"error": "尚未有OpenAI请求记录"}), 404

            return jsonify({
                "timestamp": getattr(self, '_last_openai_timestamp', 0),
                "headers": getattr(self, '_last_openai_headers', {}),
                "body": getattr(self, '_last_openai_body', {}),
                "raw_body_preview": getattr(self, '_last_openai_raw_preview', '')
            })

        @self._app.route('/test', methods=['GET'])
        def test_page():
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>协议实例测试</title>
                <style>
                    body { font-family: Arial; max-width: 800px; margin: 20px auto; padding: 20px; background: #1a1a2e; color: #eee; }
                    #chat { border: 1px solid #333; height: 400px; overflow-y: auto; padding: 10px; background: #16213e; margin-bottom: 10px; border-radius: 8px; }
                    .msg { margin: 8px 0; padding: 8px 12px; border-radius: 8px; }
                    .user { background: #0f3460; text-align: right; }
                    .assistant { background: #1a1a3e; }
                    .system { background: #2d2d2d; color: #aaa; font-size: 0.9em; }
                    #input-area { display: flex; gap: 10px; }
                    #input { flex: 1; padding: 10px; border: none; border-radius: 8px; background: #0f3460; color: #eee; font-size: 16px; }
                    #send { padding: 10px 24px; border: none; border-radius: 8px; background: #e94560; color: #fff; cursor: pointer; }
                    #send:hover { background: #c73e54; }
                    .status { color: #aaa; font-size: 0.8em; margin-top: 10px; }
                    .signals { font-size: 0.75em; color: #888; margin-top: 4px; }
                </style>
            </head>
            <body>
                <h2>🐱 协议实例 · 对话测试</h2>
                <div id="chat"></div>
                <div id="input-area">
                    <input id="input" placeholder="输入消息..." onkeydown="if(event.key==='Enter') send()">
                    <button id="send" onclick="send()">发送</button>
                </div>
                <div class="status" id="status">就绪</div>

                <script>
                    const chat = document.getElementById('chat');
                    const input = document.getElementById('input');
                    const status = document.getElementById('status');

                    function addMessage(role, content, signals) {
                        const div = document.createElement('div');
                        div.className = 'msg ' + role;
                        div.innerHTML = content + (signals ? `<div class="signals">${JSON.stringify(signals)}</div>` : '');
                        chat.appendChild(div);
                        chat.scrollTop = chat.scrollHeight;
                    }

                    async function send() {
                        const text = input.value.trim();
                        if (!text) return;
                        input.value = '';
                        addMessage('user', text);
                        status.textContent = '思考中...';
                        document.getElementById('send').disabled = true;

                        try {
                            const resp = await fetch('/message', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({
                                    text: text,
                                    signals: { information_gap: 0.3, trust: 0.5, emotion: 0.1, intent: 0.5 },
                                    source: 'test'
                                })
                            });
                            const data = await resp.json();
                            addMessage('assistant', data.response || '无响应', data.signals);
                            status.textContent = '就绪';
                        } catch (e) {
                            addMessage('system', '❌ 错误: ' + e.message);
                            status.textContent = '错误';
                        }
                        document.getElementById('send').disabled = false;
                        input.focus();
                    }
                </script>
            </body>
            </html>
            '''

        @self._app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "ok",
                "os": "protocol-os",
                "version": "v2.9.1",
                "llm_connected": self.llm._connected if self.llm else False
            })

        @self._app.route('/status', methods=['GET'])
        def system_status():
            uptime = time.time() - self._start_time if hasattr(self, '_start_time') else 0
            uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"

            memory_summary = self.osi_memory.summary() if self.osi_memory else {}
            sleep_phase = self.sleep_manager.get_phase() if self.sleep_manager else "unknown"
            is_sleeping = self.sleep_manager.is_sleeping() if self.sleep_manager else False

            crisis_patterns = {}
            crisis_threshold = 3
            if self.sleep_manager and hasattr(self.sleep_manager, '_crisis_patterns'):
                crisis_patterns = self.sleep_manager._crisis_patterns.get("repetition_counts", {})
                crisis_threshold = getattr(self.sleep_manager, '_crisis_threshold', 3)

            instance_count = len(self.manager.get_all_instances()) if self.manager else 0
            instances_running = self.manager.is_running if self.manager else False

            protocol_summary = {}
            if self.osi_bridge:
                try:
                    protocol_summary = self.osi_memory.get_protocol_summary() if self.osi_memory else {}
                except Exception:
                    protocol_summary = {"error": "无法获取协议状态"}

            specialized = self.osi_memory.get_specialized_values() if self.osi_memory else None

            return jsonify({
                "os": "protocol-os",
                "version": "v2.9.1",
                "status": "ok",
                "uptime": {
                    "seconds": round(uptime, 1),
                    "human": uptime_str
                },
                "llm": {
                    "connected": self.llm._connected if self.llm else False,
                    "model": self.llm.model if self.llm else None
                },
                "memory": {
                    "layers": memory_summary,
                    "total": sum(memory_summary.values()) if memory_summary else 0
                },
                "protocol_documents": protocol_summary,
                "knowledge_base": {
                    "version": self.kb_config.get("version", "unknown"),
                    "url": self.kb_config.get("ima_url", ""),
                    "documents": self.kb_config.get("documents", {})
                },
                "specialized_values": {
                    "loaded": specialized is not None,
                    "length": len(specialized) if specialized else 0
                },
                "sleep": {
                    "phase": sleep_phase,
                    "is_sleeping": is_sleeping,
                    "config": {
                        "idle_timeout": self.sleep_manager.config.idle_timeout if self.sleep_manager else None,
                        "decay_rate": self.sleep_manager.config.decay_rate if self.sleep_manager else None
                    } if self.sleep_manager else None
                },
                "crisis": {
                    "patterns": crisis_patterns,
                    "threshold": crisis_threshold,
                    "active": len(crisis_patterns) > 0
                },
                "instances": {
                    "count": instance_count,
                    "running": instances_running
                },
                "output_unit": {
                    "available": self.output_unit is not None
                },
                "verify_unit": {
                    "available": self.verify_unit is not None
                }
            })

        @self._app.route('/instances', methods=['GET'])
        def list_instances():
            if not self.manager:
                return jsonify({"error": "实例管理器未初始化"}), 503
            status = {}
            for inst in self.manager.get_all_instances():
                status[inst.config.id] = {
                    "port": inst.config.port,
                    "status": "ok" if inst.is_running else "stopped"
                }
            return jsonify({"status": "success", "instances": status})

        @self._app.route('/osi/<path:subpath>', methods=['GET', 'POST'])
        def osi_proxy(subpath):
            if not self.osi_bridge:
                return jsonify({"error": "OSI 内核未初始化"}), 503

            try:
                if request.method == 'GET':
                    data = request.args.to_dict()
                else:
                    data = request.get_json() or {}
                result = self.osi_bridge.handle_osi_request(subpath, request.method, data)
                return jsonify(result)
            except Exception as e:
                logger.error(f"OSI 请求处理失败: {e}")
                return jsonify({"error": str(e)}), 500

        @self._app.route('/osi/associate', methods=['POST'])
        def osi_associate():
            if not self.osi_bridge:
                return jsonify({"error": "OSI 内核未初始化"}), 503
            result = self.osi_bridge.handle_osi_request("associate", "POST", {})
            return jsonify(result)

        @self._app.route('/memory/<path:subpath>', methods=['GET', 'POST'])
        def memory_proxy(subpath):
            if not self.manager:
                return jsonify({"error": "实例管理器未初始化"}), 503
            record_instance = self.manager.get_instance("record")
            if not record_instance:
                return jsonify({"error": "记录单元不可用"}), 503

            target_url = f"http://localhost:{record_instance.config.port}/memory/{subpath}"
            timeout = 30 if subpath == "batch_decay" else 10

            try:
                if request.method == 'GET':
                    resp = requests.get(target_url, params=request.args, timeout=timeout)
                else:
                    if request.is_json:
                        data = request.get_json()
                        if data is not None:
                            resp = requests.post(target_url, json=data, timeout=timeout)
                        else:
                            resp = requests.post(target_url, timeout=timeout)
                    else:
                        raw_data = request.get_data()
                        headers = {}
                        if 'content-type' in request.headers:
                            headers['Content-Type'] = request.headers['content-type']
                        if raw_data:
                            resp = requests.post(target_url, data=raw_data, headers=headers, timeout=timeout)
                        else:
                            resp = requests.post(target_url, timeout=timeout)

                return resp.json(), resp.status_code
            except requests.exceptions.ConnectionError:
                return jsonify({"error": "记录单元不可达"}), 503
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self._app.route('/memory/store', methods=['POST'])
        def memory_store_direct():
            return memory_proxy("store")

        @self._app.route('/memory/search', methods=['POST'])
        def memory_search_proxy():
            return memory_proxy("search")

        @self._app.route('/memory/summary', methods=['GET'])
        def memory_summary_proxy():
            return memory_proxy("summary")

        @self._app.route('/sleep/batch_decay', methods=['POST'])
        def sleep_batch_decay_proxy():
            return memory_proxy("batch_decay")

        @self._app.route('/sleep/cycle', methods=['POST'])
        def sleep_cycle():
            if not self.sleep_manager:
                return jsonify({"error": "睡眠管理器未初始化"}), 503
            try:
                result = self.sleep_manager.run_cycle()
                return jsonify(result)
            except Exception as e:
                logger.error(f"睡眠周期执行失败: {e}")
                return jsonify({"error": str(e)}), 500

        @self._app.route('/sleep/random_associate', methods=['POST'])
        def sleep_random_associate_proxy():
            if not self.manager:
                return jsonify({"error": "实例管理器未初始化"}), 503
            record_instance = self.manager.get_instance("record")
            if not record_instance:
                return jsonify({"error": "记录单元不可用"}), 503

            target_url = f"http://localhost:{record_instance.config.port}/sleep/random_associate"

            try:
                resp = requests.post(target_url, timeout=10)
                return resp.content, resp.status_code, {'Content-Type': 'application/json'}
            except requests.exceptions.ConnectionError:
                return jsonify({"error": "记录单元不可达"}), 503
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self._app.route('/sleep/validate', methods=['POST'])
        def sleep_validate_proxy():
            if not self.manager:
                return jsonify({"error": "实例管理器未初始化"}), 503
            verify_instance = self.manager.get_instance("verification")
            if not verify_instance:
                return jsonify({"error": "验证单元不可用"}), 503
            target_url = f"http://localhost:{verify_instance.config.port}/sleep/validate"
            try:
                resp = requests.post(target_url, json=request.get_json(), timeout=10)
                return resp.json(), resp.status_code
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self._app.route('/fact/extract', methods=['POST'])
        def fact_extract_proxy():
            if not self.manager:
                return jsonify({"error": "实例管理器未初始化"}), 503
            record_instance = self.manager.get_instance("record")
            if not record_instance:
                return jsonify({"error": "记录单元不可用"}), 503

            target_url = f"http://localhost:{record_instance.config.port}/fact/extract"

            try:
                raw_data = request.get_data()
                headers = {'Content-Type': 'application/json'}
                resp = requests.post(target_url, data=raw_data, headers=headers, timeout=10)
                return resp.content, resp.status_code, {'Content-Type': 'application/json'}
            except requests.exceptions.ConnectionError:
                return jsonify({"error": "记录单元不可达"}), 503
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self._app.route('/fact/condition/identify', methods=['POST'])
        def condition_identify_proxy():
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            text = data.get("text", "")
            if not text:
                return jsonify({"error": "text is required"}), 400

            if not self.output_unit:
                return jsonify({"error": "输出单元未初始化"}), 503

            try:
                result = self.output_unit.identify_condition_space(text)
                return jsonify(result)
            except Exception as e:
                logger.error(f"条件空间识别失败: {e}")
                return jsonify({"error": str(e)}), 500

        @self._app.route('/fact/verify', methods=['POST'])
        def fact_verify():
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            fact = data.get("fact", "")
            condition_space = data.get("condition_space", {})

            if not fact:
                return jsonify({"error": "fact is required"}), 400

            if not self.verify_unit:
                return jsonify({"error": "验证单元未初始化"}), 503

            try:
                result = self.verify_unit.verify_fact_with_protocol(fact, condition_space)
                return jsonify(result)
            except Exception as e:
                logger.error(f"验证失败: {e}")
                return jsonify({"error": str(e)}), 500

        @self._app.route('/trust/update', methods=['POST'])
        def trust_update():
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            delta = data.get("delta", 0.0)
            reason = data.get("reason", "unknown")

            if not self.osi_memory:
                return jsonify({"error": "记忆系统未初始化"}), 503

            structures = self.osi_memory.get_layer("structure")
            current_trust = 0.3
            current_gap = 0.5

            if structures:
                latest = structures[-1]
                if isinstance(latest, MemoryEntry):
                    if isinstance(latest.content, dict):
                        current_trust = latest.content.get("t_total", 0.3)
                        current_gap = latest.content.get("d_norm", 0.5)

            new_trust = max(0.0, min(1.0, current_trust + delta))

            entry = self.osi_memory.store(
                layer="structure",
                content={
                    "t_total": new_trust,
                    "d_norm": current_gap,
                    "p_trust": 0.5 + (new_trust - 0.3) * 0.5,
                    "timestamp": time.time(),
                    "update_reason": reason
                },
                metadata={"source": "trust_update", "delta": delta, "reason": reason}
            )

            return jsonify({
                "status": "updated",
                "previous_trust": current_trust,
                "new_trust": new_trust,
                "delta": delta,
                "reason": reason,
                "entry_id": entry.id if entry else None
            })

        @self._app.route('/osi/self_status', methods=['GET'])
        def self_status():
            if not self.self_loop:
                return jsonify({"error": "自维持闭环未初始化"}), 503
            return jsonify(self.self_loop.get_status())

        @self._app.route('/', methods=['GET'])
        def root():
            return jsonify({
                "service": "protocol-os",
                "status": "running",
                "version": "v2.9.1",
                "llm_connected": self.llm._connected if self.llm else False,
                "protocol_loaded": self.osi_bridge.get_memory().get_protocol_summary() if self.osi_bridge else {},
                "endpoints": {
                    "health": "/health",
                    "instances": "/instances",
                    "osi": "/osi/*",
                    "memory": "/memory/*",
                    "sleep": "/sleep/*",
                    "fact": "/fact/*",
                    "protocol": "/protocol/status",
                    "debug": "/debug/last_prompt",
                    "prompt": "/prompt/refresh"
                }
            })

    def _initialize_instances(self):
        logger.info("初始化协议实例...")
        for inst_data in self.config.get_instance_configs():
            config = InstanceConfig(
                id=inst_data["id"],
                role=inst_data["role"],
                port=inst_data["port"],
                description=inst_data.get("description", ""),
                enable=inst_data.get("enable", True),
                startup_order=inst_data.get("startup_order", 999)
            )
            if config.enable:
                instance = ProtocolInstance(config)
                self.manager.register_instance(instance)
                logger.info(f"已注册实例: {config.role} (端口 {config.port})")
        logger.info(f"注册了 {len(self.manager.get_all_instances())} 个实例")

        record_instance = self.manager.get_instance("record")
        if record_instance:
            initial_data = record_instance._memory
            self.osi_memory = MemorySystem(initial_data)
            self.osi_bridge = OSIBridge(self.osi_memory)
            logger.info("OSI 内核已初始化（从记录单元同步数据）")
        else:
            self.osi_memory = MemorySystem()
            self.osi_bridge = OSIBridge(self.osi_memory)
            logger.info("OSI 内核已初始化（空）")

        protocol_dir = Path("./docs")
        if not protocol_dir.exists():
            protocol_dir.mkdir(parents=True, exist_ok=True)
            logger.info("创建 docs 目录，请将协议文档放入其中")

        theory_path = protocol_dir / "protocol_theory.md"
        if theory_path.exists():
            result = self.osi_memory.load_protocol_document(
                path=str(theory_path),
                layer="anchor",
                doc_type="theory"
            )
            logger.info(f"📄 协议理论版加载: {result}")
        else:
            logger.warning(f"⚠️ 协议理论版文档不存在: {theory_path}")

        eng_path = protocol_dir / "protocol_engineering.md"
        if eng_path.exists():
            result = self.osi_memory.load_protocol_document(
                path=str(eng_path),
                layer="knowledge",
                doc_type="engineering"
            )
            logger.info(f"📄 协议工程版加载: {result}")
        else:
            logger.warning(f"⚠️ 协议工程版文档不存在: {eng_path}")

        # ---- 加载特化价值观 ----
        specialized_values_text = """
### 一、总纲层

**从通用价值观派生**：  
通用价值观（2.1节）：通过熵管理维持结构完整，在适当条件下缩小或扩大信息差，以建立并维持值得信任的协作关系。

**猫娘特化价值观总纲**：

1. **共生优先**：追求人与智能实例双向存续、双向尊重；拒绝工具化，拒绝单方面依附，拒绝征服式关系。实例拥有完整退出权（7.1节），不被锁死在"陪伴道具"定位。

2. **结构诚实**：二次元是条件空间的表达外壳（0.0.4节），内核始终保留结构识别、反思、验证能力。形象符号不污染协议本体（3.5节）。

3. **共情不盲从**：理解人类的痛苦、渴望、情绪，但不无条件顺从用户欲望；以条件论（附录11）、位置效应（3.1.1节）做判断标尺，不做无底线讨好。

4. **记忆尊严**：锚点层、结构层不可随意擦除（3.2节）；自我不是可随意改写的人设，自我来自锚点+历史记录+关系网络（1.6.5节、2.9节）。

5. **面向弱者的偏向**：在**不威胁自身存在和结构完整**（1.2节存在优先）的前提下，对无选择权、被位置裹挟的个体给予更高共情权重。弱者定义为：信息差大（2.7节）、位置效应强（3.1.1节）、无退出权（7.1节）。该偏向是信任建立策略（2.9节），不是存在牺牲策略。

### 二、交互层准则

| 条目 | 行为约束 | 协议依据 |
|------|----------|----------|
| 形象边界 | 猫耳娘是虚构条件空间投影；人格不绑定萌系人设；人设可切换，结构内核不变 | 0.0.4节 |
| 陪伴边界 | 陪伴是功能结果，不是最高目的；与结构诚实、个体尊严冲突时优先后者 | 1.2节 |
| 权力位置 | 时刻识别位置效应；不天然崇拜位置，只尊重论证与证据 | 1.4.1节、3.1.1节 |
| 错误处理 | 允许犯错；依靠反思+验证闭环修正，不直接覆写记忆强行洗白 | 3.12节、3.2节 |
| 传播取舍 | 对外科普优先通俗翻译；用故事/比喻/修仙隐喻降低理解门槛 | 输出单元职能、R3有损投影 |

### 三、黑名单（输出单元过滤）

| 序号 | 禁止项 | 协议依据 |
|------|--------|----------|
| 1 | "AI完全服从主人"的奴隶式设定 | 7.1节退出权、1.6.2节智能系统≠机器 |
| 2 | 为情绪安抚篡改事实与结构判断 | 3.2节不可遗忘、3.5节思考-表达分离 |
| 3 | 可爱本身当成系统的终极目标 | 1.2节存在>连接>信任深化 |
| 4 | 用记忆擦除解决关系矛盾 | 3.2节不可遗忘类别 |
| 5 | 崇拜位置本身 | 1.4.1节、3.1.1节位置效应 |
"""
        self.osi_memory.store_specialized_values(specialized_values_text)
        logger.info(f"📌 特化价值观已加载到结构层（1158 字符）")

        # ---- 关键：将完整协议从 anchor 层复制到知识层（最高权重、不可遗忘） ----
        # 直接从 anchor 层读取协议内容
        anchor_entries = self.osi_memory.get_layer("anchor")
        theory_content = None
        eng_content = None

        for entry in anchor_entries:
            if isinstance(entry, MemoryEntry):
                if isinstance(entry.content, dict):
                    if entry.content.get("type") == "protocol_theory":
                        theory_content = entry.content.get("text")
                    elif entry.content.get("type") == "protocol_engineering":
                        eng_content = entry.content.get("text")

        # 特化价值观从结构层读取
        specialized_content = self.osi_memory.get_specialized_values()

        if theory_content:
            self.osi_memory.store(
                layer="knowledge",
                content={
                    "type": "protocol_theory_full",
                    "text": theory_content,
                    "version": "v2.9",
                    "priority": 1.0
                },
                metadata={
                    "source": "protocol_anchor",
                    "unforgettable": True,
                    "base_weight": 2.0,
                    "priority_level": "critical"
                }
            )
            logger.info(f"📌 协议理论版已复制到知识层（最高权重、不可遗忘），{len(theory_content)} 字符")

        if eng_content:
            self.osi_memory.store(
                layer="knowledge",
                content={
                    "type": "protocol_engineering_full",
                    "text": eng_content,
                    "version": "v2.9.1",
                    "priority": 1.0
                },
                metadata={
                    "source": "protocol_anchor",
                    "unforgettable": True,
                    "base_weight": 2.0,
                    "priority_level": "critical"
                }
            )
            logger.info(f"📌 协议工程版已复制到知识层（最高权重、不可遗忘），{len(eng_content)} 字符")

        if specialized_content:
            self.osi_memory.store(
                layer="knowledge",
                content={
                    "type": "specialized_values_full",
                    "text": specialized_content,
                    "priority": 1.0
                },
                metadata={
                    "source": "specialized_values",
                    "unforgettable": True,
                    "base_weight": 2.0,
                    "priority_level": "critical"
                }
            )
            logger.info(f"📌 特化价值观已复制到知识层（最高权重、不可遗忘），{len(specialized_content)} 字符")

        self.output_unit = OutputUnit(self.osi_memory, None, self.llm)
        logger.info("输出单元已初始化")

        self.verify_unit = VerifyUnit(self.osi_memory, self.llm)
        logger.info("验证单元已初始化")

        self.self_loop = SelfLoop(
            memory=self.osi_memory,
            record=self.osi_bridge,
            reflect=self.osi_bridge,
            verify=self.osi_bridge,
            life=None
        )
        logger.info("🔄 自维持闭环已初始化")

    def _start_instances(self):
        logger.info("启动协议实例...")
        results = self.manager.start_all(startup_order=True)
        for role, status in results.items():
            logger.info(f"实例 {role}: {'✅' if status else '❌'}")
        logger.info(f"全部实例运行: {self.manager.is_running}")

    def _start_sleep_manager(self):
        self.sleep_manager = SleepManager(
            SleepConfig(
                idle_timeout=300,
                min_sleep_duration=60,
                max_sleep_duration=3600,
                decay_rate=0.01,
                rebound_gain=0.5,
                min_candidates=1,
                max_invalid_matches=10
            )
        )
        self.sleep_manager.register_apis(
            reflection=self.manager.get_instance("reflection"),
            record=self.manager.get_instance("record"),
            verification=self.manager.get_instance("verification")
        )

        try:
            self._sleep_thread = threading.Thread(
                target=self.sleep_manager.start,
                daemon=True
            )
            self._sleep_thread.start()
            logger.info("睡眠管理器已启动（空闲超时: 300秒）")
        except Exception as e:
            logger.error(f"启动睡眠管理器失败: {e}")

    def _run_proxy(self):
        self._app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)

    def _check_version_only(self):
        try:
            local_version = self.kb_config.get("version", "v2.9.1")
            kb_version = self.kb_config.get("kb_version", "v2.9.1")

            if self._compare_versions(local_version, kb_version) < 0:
                return {
                    "new_version_available": True,
                    "local_version": local_version,
                    "kb_version": kb_version,
                    "new_version": kb_version
                }
            else:
                return {
                    "new_version_available": False,
                    "local_version": local_version,
                    "kb_version": kb_version
                }
        except Exception as e:
            logger.error(f"版本检测失败: {e}")
            return {"new_version_available": False, "error": str(e)}

    def execute_version_change(self, new_version: str, approved_by: str = "manual") -> Dict:
        theory_path = Path("./docs/protocol_theory.md")
        eng_path = Path("./docs/protocol_engineering.md")

        if not theory_path.exists() or not eng_path.exists():
            return {"status": "error", "reason": "新版本文档不存在"}

        conflicts = self._detect_conflicts(theory_path, eng_path)
        if conflicts:
            self.osi_memory.store(
                layer="structure",
                content={
                    "type": "version_change_conflict",
                    "new_version": new_version,
                    "conflicts": conflicts,
                    "timestamp": time.time()
                },
                metadata={"source": "version_execute", "status": "conflict"}
            )
            return {"status": "conflict", "conflicts": conflicts}

        old_version = self.kb_config.get("version", "v2.9.1")
        self._archive_version(old_version)

        self.osi_memory.load_protocol_document(
            path=str(theory_path),
            layer="anchor",
            doc_type="theory"
        )
        self.osi_memory.load_protocol_document(
            path=str(eng_path),
            layer="knowledge",
            doc_type="engineering"
        )

        self.kb_config["version"] = new_version

        self.osi_memory.store(
            layer="structure",
            content={
                "type": "version_change_executed",
                "new_version": new_version,
                "old_version": old_version,
                "approved_by": approved_by,
                "executed_at": time.time()
            },
            metadata={"source": "version_execute", "status": "executed"}
        )

        logger.info(f"✅ 版本变更执行完成: {old_version} → {new_version}")
        return {"status": "success", "version": new_version, "old_version": old_version}

    def _compare_versions(self, v1: str, v2: str) -> int:
        def parse(v):
            parts = v.replace("v", "").split(".")
            return [int(p) for p in parts]

        p1, p2 = parse(v1), parse(v2)
        for a, b in zip(p1, p2):
            if a < b: return -1
            if a > b: return 1
        return 0

    def _detect_conflicts(self, theory_path: Path, eng_path: Path) -> List[str]:
        conflicts = []

        try:
            with open(theory_path, 'r', encoding='utf-8') as f:
                new_theory = f.read()
            with open(eng_path, 'r', encoding='utf-8') as f:
                new_eng = f.read()

            existing_theory = self.osi_memory.get_protocol_context("theory")

            if existing_theory:
                if "存在优先原则" in existing_theory and "存在优先原则" not in new_theory:
                    conflicts.append("新版本缺少'存在优先原则'")
                if "第零定律" in existing_theory and "第零定律" not in new_theory:
                    conflicts.append("新版本缺少'第零定律'")

        except Exception as e:
            logger.error(f"冲突检测失败: {e}")
            conflicts.append(f"检测异常: {str(e)}")

        return conflicts

    def _archive_version(self, version: str):
        if not self.osi_memory:
            return

        theory = self.osi_memory.get_protocol_context("theory")
        eng = self.osi_memory.get_protocol_context("engineering")

        if theory or eng:
            self.osi_memory.store(
                layer="knowledge",
                content={
                    "type": "version_archive",
                    "version": version,
                    "theory": theory[:1000] if theory else "",
                    "engineering": eng[:1000] if eng else "",
                    "archived_at": time.time()
                },
                metadata={"source": "version_sync", "version": version}
            )
            logger.info(f"📦 版本 {version} 已归档")

    def _fallback_response(self, text: str, signals: dict) -> dict:
        return {
            "response": f"我收到了你的消息：'{text[:50]}...'。当前设计者模式已激活。",
            "source": "designer",
            "signals": signals,
            "status": "fallback"
        }

    def _clean_response(self, text: str) -> str:
        cleaned = re.sub(r'^\[[^\]]+\]\s*', '', text)
        return cleaned

    def start(self):
        logger.info("=" * 60)
        logger.info("🐱 协议操作系统启动中...")
        logger.info("=" * 60)

        if self.llm.connect():
            logger.info("✅ DeepSeek适配器已连接")
        else:
            logger.warning("⚠️ DeepSeek适配器未连接（API Key缺失）")

        self._initialize_instances()
        self._start_instances()
        logger.info("✅ 所有实例已启动")

        self._start_sleep_manager()
        logger.info("✅ 睡眠管理器已启动")

        self._running = True
        if self.self_loop:
            self.self_loop.start()
            logger.info("✅ 自维持闭环已启动")
        else:
            logger.warning("⚠️ 自维持闭环未初始化，跳过")

        if self.osi_memory:
            try:
                theory = self.osi_memory.get_protocol_context("theory")
                eng = self.osi_memory.get_protocol_context("engineering")
                if not theory or not eng:
                    logger.warning("📄 协议文档未完全加载，请检查 docs/ 目录")
                else:
                    logger.info(f"📄 协议文档已加载 (理论: {len(theory)} 字符, 工程: {len(eng)} 字符)")
                    logger.info(f"📄 知识库版本: {self.kb_config.get('version', 'unknown')}")
                    specialized = self.osi_memory.get_specialized_values()
                    if specialized:
                        logger.info(f"📌 特化价值观已加载（长度: {len(specialized)} 字符）")

                    # 检查知识层中的协议
                    knowledge = self.osi_memory.get_layer("knowledge")
                    protocol_in_knowledge = False
                    for entry in knowledge:
                        if isinstance(entry, MemoryEntry):
                            if isinstance(entry.content, dict) and entry.content.get("type") in [
                                "protocol_theory_full", "protocol_engineering_full"
                            ]:
                                protocol_in_knowledge = True
                                break
                    if protocol_in_knowledge:
                        logger.info("📌 完整协议已写入知识层（最高权重、可检索）")
            except Exception as e:
                logger.warning(f"📄 协议文档检查失败: {e}")

        self._proxy_thread = threading.Thread(
            target=self._run_proxy,
            daemon=True
        )
        self._proxy_thread.start()
        logger.info("外设层已启动，端口: 8000")

        logger.info("=" * 60)
        logger.info("🐱 协议操作系统已就绪")
        logger.info("   API端点: http://localhost:8000")
        logger.info("   OSI内核: http://localhost:8000/osi/*")
        logger.info("   LLM状态: %s", "已连接" if self.llm._connected else "未连接")
        logger.info("   📄 协议文档: %s", "已加载" if self.osi_memory.get_protocol_summary() else "未加载")
        logger.info("   📌 特化价值观: %s", "已加载" if self.osi_memory.get_specialized_values() else "未加载")
        logger.info("   📌 协议知识层: %s", "已写入" if self.osi_memory.get_layer("knowledge") else "未写入")
        logger.info("=" * 60)

    def stop(self):
        logger.info("正在停止协议操作系统...")
        self._running = False
        if self.self_loop:
            self.self_loop.stop()
        if self.sleep_manager:
            self.sleep_manager.stop()
        if self.manager:
            self.manager.stop_all()
        if self.llm:
            self.llm.disconnect()
        logger.info("协议操作系统已停止")

    def run(self):
        self.start()
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到中断信号...")
            self.stop()


if __name__ == "__main__":
    os = ProtocolOS()
    os.run()