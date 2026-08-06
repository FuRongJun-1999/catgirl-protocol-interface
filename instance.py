import json
import logging
import threading
import time
import requests
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from flask import Flask, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 项目根目录：从当前文件向上两级 (src/core/ -> src/ -> 项目根目录)
project_root = Path(__file__).parent.parent.parent
model_path = str(project_root / 'models' / 'bge-small-zh-v1.5')
MEMORY_DATA_PATH = project_root / 'data' / 'memory.json'

logger = logging.getLogger(__name__)


@dataclass
class InstanceConfig:
    id: str
    role: str
    port: int
    description: str
    enable: bool
    startup_order: int


class ProtocolInstance:
    def __init__(self, config: InstanceConfig):
        self.config = config
        self._running = False
        self._trust_state = {
            "p_trust": 0.5,
            "p_gap": 0.5,
            "t_total": 0.3,
            "e_weight": 0.0
        }
        # 五层记忆存储
        self._memory = {
            "anchor": [],
            "structure": [],
            "context": [],
            "knowledge": [],
            "self": []
        }
        # 情境层衰减配置
        self._decay_config = {
            "decay_rate": 0.01,
            "rebound_gain": 0.5,
            "min_weight": 0.01,
            "retrieval_threshold": 0.1
        }

        # ---- 重要性评估相关属性 ----
        self._last_d2D_sign = "unknown"
        self._last_d2D_value = 0.0
        self._last_d2T_sign = "unknown"
        self._last_d2T_value = 0.0
        self._last_importance_result = {}
        self._pending_promotion = []

        # ===== 如果是记录单元，加载持久化记忆 =====
        if self.config.role == "record":
            self._load_memory_from_disk()

        self._app = Flask(f"instance_{config.id}")
        self._server = None
        self._thread = None
        self._setup_routes()

        # 记录单元：初始化向量数据库
        if self.config.role == "record":
            self._embedding_model = None
            try:
                self._chroma_client = chromadb.PersistentClient(
                    path=f"./chroma_data/{self.config.id}"
                )
                self._embedding_model = SentenceTransformer(model_path)

                self._collections = {}
                for layer in ["anchor", "structure", "context", "knowledge", "self"]:
                    self._collections[layer] = self._chroma_client.get_or_create_collection(
                        name=f"memory_{layer}",
                        metadata={"hnsw:space": "cosine"}
                    )
                logger.info(f"向量数据库已初始化，模型: bge-small-zh-v1.5")
            except Exception as e:
                logger.error(f"记录单元初始化失败: {e}")
                self._embedding_model = None
                self._collections = {}

    # ===== 记忆持久化方法 =====

    def _load_memory_from_disk(self):
        """从磁盘加载持久化记忆"""
        if not MEMORY_DATA_PATH.exists():
            logger.info(f"记忆文件不存在: {MEMORY_DATA_PATH}，使用空记忆")
            return

        try:
            with open(MEMORY_DATA_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for layer, entries in data.get("layers", {}).items():
                if layer in self._memory:
                    self._memory[layer] = entries

            logger.info(f"记忆已从磁盘加载: {sum(len(v) for v in self._memory.values())} 条记录")
        except Exception as e:
            logger.error(f"加载记忆失败: {e}")

    def _save_memory_to_disk(self):
        """保存记忆到磁盘"""
        try:
            serializable_memory = {}
            for layer, entries in self._memory.items():
                serializable_memory[layer] = []
                for entry in entries:
                    if hasattr(entry, '__dict__'):
                        serializable_memory[layer].append({
                            "id": entry.id,
                            "content": entry.content,
                            "timestamp": entry.timestamp,
                            "metadata": entry.metadata,
                            "weight": getattr(entry, "weight", 1.0),
                            "base_weight": getattr(entry, "base_weight", 1.0),
                            "last_access": getattr(entry, "last_access", entry.timestamp),
                            "trigger_count": getattr(entry, "trigger_count", 0)
                        })
                    elif isinstance(entry, dict):
                        serializable_memory[layer].append(entry)
                    else:
                        serializable_memory[layer].append({
                            "id": f"{layer}_{int(time.time() * 1000)}",
                            "content": str(entry),
                            "timestamp": time.time(),
                            "metadata": {},
                            "weight": 1.0,
                            "base_weight": 1.0,
                            "last_access": time.time(),
                            "trigger_count": 0
                        })
            
            MEMORY_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(MEMORY_DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": time.time(),
                    "layers": serializable_memory
                }, f, ensure_ascii=False, indent=2)
            logger.debug(f"记忆已保存到磁盘: {sum(len(v) for v in serializable_memory.values())} 条记录")
        except Exception as e:
            logger.error(f"保存记忆失败: {e}")

    # ===== 重要性评估方法 =====

    def _calculate_importance(self, entry: Dict) -> float:
        try:
            structures = [s for s in self._memory.get("structure", []) if isinstance(s, dict)]
            total = len(structures)

            if total < 2:
                self._last_importance_result = {
                    "score": 0.3,
                    "reason": "structure_data_insufficient",
                    "total": total
                }
                return 0.3

            current_trust = 0.3
            current_gap = 0.5
            current_e_weight = 0.0
            latest = structures[-1].get("content", {})
            if isinstance(latest, dict):
                current_trust = latest.get("t_total", 0.3)
                current_gap = latest.get("d_norm", 0.5)
                current_e_weight = latest.get("e_weight", 0.0)

            gap_reduction = 0.0
            before_content = structures[-2].get("content", {})
            after_content = structures[-1].get("content", {})
            if isinstance(before_content, dict) and isinstance(after_content, dict):
                before = before_content.get("d_norm", 0.5)
                after = after_content.get("d_norm", 0.5)
                if after < before:
                    gap_reduction = min(1.0, (before - after) / 0.5)

            trust_increase = 0.0
            if isinstance(before_content, dict) and isinstance(after_content, dict):
                before_t = before_content.get("t_total", 0.3)
                after_t = after_content.get("t_total", 0.3)
                if after_t > before_t:
                    trust_increase = min(1.0, (after_t - before_t) / 0.5)

            gap_second_order = 0.0
            d2D_sign = "zero"
            if total >= 3:
                d0 = structures[-3].get("content", {})
                d1 = structures[-2].get("content", {})
                d2 = structures[-1].get("content", {})
                if all(isinstance(x, dict) for x in [d0, d1, d2]):
                    d0_val = d0.get("d_norm", 0.5)
                    d1_val = d1.get("d_norm", 0.5)
                    d2_val = d2.get("d_norm", 0.5)
                    dD_1 = d1_val - d0_val
                    dD_2 = d2_val - d1_val
                    d2D = dD_2 - dD_1
                    gap_second_order = min(1.0, abs(d2D) * 10)
                    d2D_sign = "positive" if d2D > 0 else "negative" if d2D < 0 else "zero"

            trust_second_order = 0.0
            if total >= 3:
                t0 = structures[-3].get("content", {})
                t1 = structures[-2].get("content", {})
                t2 = structures[-1].get("content", {})
                if all(isinstance(x, dict) for x in [t0, t1, t2]):
                    t0_val = t0.get("t_total", 0.3)
                    t1_val = t1.get("t_total", 0.3)
                    t2_val = t2.get("t_total", 0.3)
                    dT_1 = t1_val - t0_val
                    dT_2 = t2_val - t1_val
                    d2T = dT_2 - dT_1
                    trust_second_order = min(1.0, abs(d2T) * 10)

            emotion_deviation = 0.0
            e_weights = []
            for s in structures:
                c = s.get("content", {})
                if isinstance(c, dict) and "e_weight" in c:
                    e_weights.append(c.get("e_weight", 0.0))
            if len(e_weights) >= 3:
                expectation = sum(e_weights[-10:]) / len(e_weights[-10:])
                deviation = abs(current_e_weight - expectation)
                emotion_deviation = min(1.0, deviation * 5)

            semantic_density = 0.0
            content = entry.get("content", "")
            if isinstance(content, str):
                words = len(content.split())
                unique_words = len(set(content.split()))
                if words > 0:
                    semantic_density = min(1.0, (unique_words / words) * 0.6 + min(1.0, words / 30) * 0.4)

            w_gap, w_trust, w_gap2, w_trust2, w_emotion, w_semantic = self._get_dynamic_weights(
                current_trust, current_gap
            )

            second_order_boost = 0.0
            if gap_second_order > 0.3:
                second_order_boost += 0.08
            if trust_second_order > 0.3:
                second_order_boost += 0.08

            score = (
                w_gap * gap_reduction +
                w_trust * trust_increase +
                w_gap2 * gap_second_order +
                w_trust2 * trust_second_order +
                w_emotion * emotion_deviation +
                w_semantic * semantic_density
            ) + second_order_boost

            if d2D_sign == "negative":
                score += 0.05
            elif d2D_sign == "positive":
                score += 0.10

            self._last_importance_result = {
                "score": round(score, 3),
                "weights": {
                    "gap": round(w_gap, 2),
                    "trust": round(w_trust, 2),
                    "gap2": round(w_gap2, 2),
                    "trust2": round(w_trust2, 2),
                    "emotion": round(w_emotion, 2),
                    "semantic": round(w_semantic, 2)
                },
                "gap_reduction": round(gap_reduction, 3),
                "trust_increase": round(trust_increase, 3),
                "gap_second_order": round(gap_second_order, 3),
                "trust_second_order": round(trust_second_order, 3),
                "emotion_deviation": round(emotion_deviation, 3),
                "semantic_density": round(semantic_density, 3),
                "second_order_boost": round(second_order_boost, 3),
                "system_state": {
                    "trust": round(current_trust, 3),
                    "gap": round(current_gap, 3),
                    "e_weight": round(current_e_weight, 3),
                    "phase": self._get_phase(current_trust, current_gap)
                }
            }

            return min(1.0, max(0.0, score))

        except Exception as e:
            logger.warning(f"重要性评估失败: {e}")
            self._last_importance_result = {"error": str(e), "type": str(type(e))}
            return 0.5

    # ---- 辅助方法 ----
    def _get_phase(self, trust: float, gap: float) -> str:
        if trust < 0.5:
            return "建立期"
        if gap > 0.5:
            return "认知更新期"
        return "深化期"

    def _get_dynamic_weights(self, trust: float, gap: float) -> tuple:
        if trust < 0.5:
            return 0.15, 0.45, 0.15, 0.10, 0.05, 0.10
        if gap > 0.5:
            return 0.35, 0.20, 0.20, 0.10, 0.05, 0.10
        return 0.15, 0.15, 0.15, 0.10, 0.30, 0.10
        
    def _setup_routes(self):
        @self._app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "ok",
                "instance": self.config.id,
                "role": self.config.role,
                "running": self._running
            })

        @self._app.route('/trust', methods=['GET'])
        def get_trust():
            return jsonify(self._trust_state)

        @self._app.route('/trust', methods=['POST'])
        def update_trust():
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            for key in ["p_trust", "p_gap", "t_total", "e_weight"]:
                if key in data:
                    self._trust_state[key] = data[key]
            return jsonify({"status": "updated", "trust": self._trust_state})

        # ===== 统一消息端点（所有实例） =====

        @self._app.route('/message', methods=['POST'])
        def message():
            """统一消息端点：接收标准格式请求，支持思维链"""
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            text = data.get("text", "")
            signals = data.get("signals", {})
            history = data.get("history", [])
            source = data.get("source", "chat")
            router = data.get("router", "designer")
            chain_id = data.get("_chain_id", "")
            chain_step = data.get("_chain_step", 0)
            
            if not text:
                return jsonify({"error": "text is required"}), 400
            
            # 根据角色和链步骤处理
            role = self.config.role
            response_text = self._process_chain_message(text, signals, history, source, role, chain_id, chain_step)
            
            return jsonify({
                "response": response_text,
                "source": role,
                "signals": signals,
                "status": "ok"
            })

        # ===== 设计者观测端点（实例6专属） =====

        @self._app.route('/arbitration/version/pending', methods=['GET'])
        def arbitration_version_pending():
            if self.config.role != "arbitration":
                return jsonify({"error": "Forbidden"}), 403
            
            structures = self._memory.get("structure", [])
            pending = []
            for entry in structures:
                if isinstance(entry, dict):
                    content = entry.get("content", {})
                    if content.get("type") == "version_change_request" and content.get("status") == "pending":
                        pending.append(entry)
                elif hasattr(entry, 'content'):
                    if entry.content.get("type") == "version_change_request" and entry.content.get("status") == "pending":
                        pending.append(entry)
            
            return jsonify({
                "status": "ok",
                "pending_requests": pending,
                "count": len(pending),
                "observed_at": time.time()
            })

        @self._app.route('/arbitration/version/status', methods=['GET'])
        def arbitration_version_status():
            if self.config.role != "arbitration":
                return jsonify({"error": "Forbidden"}), 403
            
            structures = self._memory.get("structure", [])
            latest_version = "v2.9.1"
            for entry in structures:
                if isinstance(entry, dict):
                    content = entry.get("content", {})
                    if content.get("type") == "version_change_executed":
                        latest_version = content.get("new_version", latest_version)
                elif hasattr(entry, 'content'):
                    if entry.content.get("type") == "version_change_executed":
                        latest_version = entry.content.get("new_version", latest_version)
            
            return jsonify({
                "current_version": latest_version,
                "observed_at": time.time()
            })

        # ===== 记忆系统端点（记录单元专属） =====

        @self._app.route('/memory/all', methods=['GET'])
        def memory_all():
            if self.config.role != "record":
                return jsonify({"error": "Only record unit can provide memory data"}), 403

            layer = request.args.get('layer')
            if layer:
                if layer not in self._memory:
                    return jsonify({"error": f"Invalid layer: {layer}"}), 400
                return jsonify({layer: self._memory[layer]})
            return jsonify(self._memory)

        @self._app.route('/memory/summary', methods=['GET'])
        def memory_summary():
            if self.config.role != "record":
                return jsonify({"error": "Only record unit has memory data"}), 403
            counts = {layer: len(entries) for layer, entries in self._memory.items()}
            return jsonify({
                "instance": self.config.id,
                "role": self.config.role,
                "memory_counts": counts,
                "total_entries": sum(counts.values())
            })

        @self._app.route('/memory/store', methods=['POST'])
        def memory_store():
            if self.config.role != "record":
                return jsonify({"error": "Only record unit can store memory"}), 403

            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            layer = data.get("layer")
            content = data.get("content")
            metadata = data.get("metadata", {})

            if isinstance(content, dict):
                content_str = json.dumps(content, ensure_ascii=False)
            else:
                content_str = str(content)

            if not layer or not content_str:
                return jsonify({"error": "layer and content are required"}), 400

            if layer not in self._memory:
                return jsonify({"error": f"Invalid layer: {layer}"}), 400

            entry_id = f"{layer}_{int(time.time() * 1000)}"
            entry = {
                "id": entry_id,
                "timestamp": time.time(),
                "content": content_str,
                "metadata": metadata
            }

            if layer == "context":
                entry["weight"] = 1.0
                entry["base_weight"] = 1.0
                entry["last_access"] = time.time()
                entry["trigger_count"] = 0
                entry["decay_rate"] = self._decay_config["decay_rate"]

                try:
                    importance = self._calculate_importance(entry)
                    entry["importance"] = importance
                    if importance > 0.7:
                        self._pending_promotion.append(entry)
                        logger.info(f"📌 高重要性记忆 (得分: {importance:.3f}): {content_str[:50]}...")
                except Exception as e:
                    logger.warning(f"重要性评估失败，使用默认值: {e}")
                    entry["importance"] = 0.5

            self._memory[layer].append(entry)

            if hasattr(self, '_embedding_model') and self._embedding_model is not None:
                embedding = self._embedding_model.encode(str(content)).tolist()
                self._collections[layer].add(
                    ids=[entry_id],
                    embeddings=[embedding],
                    metadatas=[{
                        "content": str(content),
                        "timestamp": entry["timestamp"],
                        "source": metadata.get("source", "unknown"),
                        "layer": layer
                    }]
                )

            self._save_memory_to_disk()

            return jsonify({
                "status": "stored",
                "layer": layer,
                "entry": entry,
                "count": len(self._memory[layer])
            })

        @self._app.route('/memory/importance_detail', methods=['GET'])
        def importance_detail():
            if self.config.role != "record":
                return jsonify({"error": "Only record unit can provide importance details"}), 403

            detail = self._get_importance_detail()
            if not isinstance(detail, dict):
                return jsonify({
                    "error": "Invalid importance detail format",
                    "raw": detail,
                    "type": str(type(detail))
                })
            return jsonify(detail)

        @self._app.route('/memory/recall', methods=['POST'])
        def memory_recall():
            if self.config.role != "record":
                return jsonify({"error": "Only record unit can recall memory"}), 403

            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            query = data.get("query", "")
            layer = data.get("layer")
            limit = data.get("limit", 10)
            threshold = data.get("threshold", 0.1)

            if not query:
                return jsonify({"error": "query is required"}), 400

            query_embedding = self._embedding_model.encode(query).tolist()
            target_layers = [layer] if layer else list(self._collections.keys())
            results = []

            for layer_name in target_layers:
                collection = self._collections[layer_name]
                try:
                    resp = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=limit,
                        include=["metadatas", "distances"]
                    )
                    for i, (meta, dist) in enumerate(zip(resp['metadatas'][0], resp['distances'][0])):
                        similarity = 1 - dist
                        if similarity >= threshold:
                            entry = self._find_memory_entry(layer_name, meta.get("id", ""))
                            weight = self._calculate_decay_weight(entry) if entry else 1.0
                            final_score = similarity * weight
                            results.append({
                                "layer": layer_name,
                                "content": meta.get("content", ""),
                                "similarity": round(similarity, 3),
                                "weight": round(weight, 3),
                                "final_score": round(final_score, 3),
                                "timestamp": meta.get("timestamp", 0)
                            })
                except Exception as e:
                    logger.warning(f"查询 {layer_name} 失败: {e}")
                    continue

            results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            return jsonify({
                "status": "success",
                "query": query,
                "results": results[:limit],
                "count": len(results)
            })

        @self._app.route('/memory/touch', methods=['POST'])
        def memory_touch():
            if self.config.role != "record":
                return jsonify({"error": "Only record unit can touch memory"}), 403

            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            entry_id = data.get("entry_id")
            if not entry_id:
                return jsonify({"error": "entry_id is required"}), 400

            entry = self._find_memory_entry_by_id(entry_id)
            if not entry:
                return jsonify({"error": "Entry not found"}), 404

            entry["trigger_count"] = entry.get("trigger_count", 0) + 1
            entry["last_access"] = time.time()

            rebound_gain = self._decay_config["rebound_gain"] / (entry["trigger_count"] + 1)
            current_weight = self._calculate_decay_weight(entry)
            entry["base_weight"] = min(
                entry.get("initial_weight", 1.0),
                current_weight + rebound_gain
            )

            self._save_memory_to_disk()

            return jsonify({
                "status": "touched",
                "entry_id": entry_id,
                "new_weight": self._calculate_decay_weight(entry),
                "trigger_count": entry["trigger_count"]
            })

        @self._app.route('/sleep/batch_decay', methods=['POST'])
        def sleep_batch_decay():
            if self.config.role != "record":
                return jsonify({"error": "Only record unit can batch decay"}), 403

            data = request.get_json() or {}
            decay_rate = data.get('decay_rate', 0.01)

            updated_count = 0
            for entry in self._memory.get("context", []):
                if hasattr(entry, 'metadata'):
                    if entry.metadata.get("unforgettable", False):
                        continue
                    elapsed_days = (time.time() - entry.last_access) / 86400
                    old_weight = entry.weight
                    new_weight = old_weight / (1 + decay_rate * (elapsed_days ** 2))
                    entry.weight = max(new_weight, 0.01)
                    entry.last_access = time.time()
                    updated_count += 1
                elif isinstance(entry, dict):
                    if entry.get("metadata", {}).get("unforgettable", False):
                        continue
                    elapsed_days = (time.time() - entry.get("last_access", time.time())) / 86400
                    old_weight = entry.get("weight", 1.0)
                    new_weight = old_weight / (1 + decay_rate * (elapsed_days ** 2))
                    entry["weight"] = max(new_weight, 0.01)
                    entry["last_access"] = time.time()
                    updated_count += 1

            self._save_memory_to_disk()

            return jsonify({
                "status": "decayed",
                "updated_count": updated_count,
                "decay_rate": decay_rate
            })

        @self._app.route('/fact/condition/identify', methods=['POST'])
        def condition_identify():
            if self.config.role != "record":
                return jsonify({"error": "Only record unit can identify condition space"}), 403

            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            text = data.get("text", "")
            if not text:
                return jsonify({"error": "text is required"}), 400

            try:
                resp = requests.post(
                    "http://localhost:8004/condition/identify",
                    json={"text": text},
                    timeout=30
                )
                if resp.status_code == 200:
                    return jsonify(resp.json())
                else:
                    return jsonify({"error": "输出单元调用失败"}), 500
            except Exception as e:
                logger.error(f"条件空间识别调用失败: {e}")
                return jsonify({"error": str(e)}), 500

        @self._app.route('/memory/search', methods=['POST'])
        def memory_search():
            """
            增强版记忆检索：支持权重排序、优先级过滤
            - 协议内容（priority_level: critical）自动获得 2.0 倍权重
            - 结果按 final_score 排序，高权重内容优先返回
            """
            if self.config.role != "record":
                return jsonify({"error": "Only record unit can search memory"}), 403

            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            query = data.get("query", "")
            layer = data.get("layer")
            limit = data.get("limit", 10)
            threshold = data.get("threshold", 0.5)
            signals = data.get("signals", {})
            rerank = data.get("rerank", False)
            priority = data.get("priority")  # "critical", "high", "normal"

            if not query:
                return jsonify({"error": "query is required"}), 400

            if not hasattr(self, '_embedding_model') or self._embedding_model is None:
                return jsonify({"error": "向量模型未初始化"}), 500

            query_embedding = self._embedding_model.encode(query).tolist()
            target_layers = [layer] if layer else list(self._collections.keys())
            results = []

            for layer_name in target_layers:
                collection = self._collections[layer_name]
                try:
                    resp = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=limit * 2,
                        include=["metadatas", "distances"]
                    )
                    for i, (meta, dist) in enumerate(zip(resp['metadatas'][0], resp['distances'][0])):
                        similarity = 1 - dist
                        if similarity >= threshold:
                            # ---- 权重计算 ----
                            weight = 1.0
                            priority_level = meta.get("priority_level", "normal")

                            if priority_level == "critical":
                                weight = 2.0
                            elif priority_level == "high":
                                weight = 1.5
                            elif meta.get("unforgettable"):
                                weight = 1.3

                            final_score = similarity * weight

                            results.append({
                                "layer": layer_name,
                                "content": meta.get("content", ""),
                                "similarity": round(similarity, 3),
                                "timestamp": meta.get("timestamp", 0),
                                "source": meta.get("source", "unknown"),
                                "weight": round(weight, 2),
                                "final_score": round(final_score, 3),
                                "priority_level": priority_level,
                                "id": meta.get("id", f"result_{i}")
                            })
                except Exception as e:
                    logger.warning(f"搜索 {layer_name} 失败: {e}")
                    continue

            # ---- 按 final_score 排序（高权重内容优先） ----
            results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

            # ---- 记录检索摘要 ----
            critical_hits = [r for r in results if r.get("priority_level") == "critical"]
            if critical_hits:
                logger.info(f"📊 检索到 {len(critical_hits)} 条 critical 优先级内容（协议/特化价值观）")
                for r in critical_hits[:2]:
                    logger.info(f"   📌 {r.get('content', '')[:60]}... (权重: {r.get('weight')})")

            return jsonify({
                "status": "success",
                "query": query,
                "results": results[:limit],
                "count": len(results),
                "critical_hits": len(critical_hits)
            })

        @self._app.route('/fact/extract', methods=['POST'])
        def fact_extract():
            if self.config.role != "record":
                return jsonify({"error": "Only record unit can extract facts"}), 403

            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            text = data.get("text", "")
            context = data.get("context", {})

            if not text:
                return jsonify({"error": "text is required"}), 400

            try:
                resp = requests.post(
                    "http://localhost:8004/condition/identify",
                    json={"text": text},
                    timeout=30
                )
                if resp.status_code == 200:
                    condition_space = resp.json()
                else:
                    condition_space = {"error": "条件空间识别失败"}
            except Exception as e:
                condition_space = {"error": str(e)}

            fact = {
                "content": text,
                "source": context.get("source", "未知"),
                "user": context.get("user", "未知"),
                "timestamp": time.time(),
                "condition_space": condition_space
            }

            entry = self._memory["knowledge"].append({
                "id": f"fact_{int(time.time() * 1000)}",
                "timestamp": time.time(),
                "content": text,
                "metadata": {
                    "source": context.get("source", "未知"),
                    "user": context.get("user", "未知"),
                    "condition_space": condition_space,
                    "verified": False
                }
            })

            return jsonify({
                "status": "stored",
                "fact": fact,
                "condition_space": condition_space
            })

        # ===== 联想功能端点（记录单元专属） =====

        @self._app.route('/sleep/random_associate', methods=['POST'])
        def sleep_random_associate():
            if self.config.role != "record":
                return jsonify({"error": "Only record unit can associate"}), 403

            try:
                context_entries = self._memory.get("context", [])
                knowledge_entries = self._memory.get("knowledge", [])

                if not context_entries or not knowledge_entries:
                    return jsonify({"valid": False, "reason": "记忆数据不足"})

                context_entry = random.choice(context_entries)
                if hasattr(context_entry, 'content'):
                    context_content = context_entry.content
                else:
                    context_content = context_entry.get("content", "")

                if not context_content:
                    return jsonify({"valid": False, "reason": "情境内容为空"})

                if not hasattr(self, '_embedding_model') or self._embedding_model is None:
                    return jsonify({"valid": False, "reason": "向量模型未初始化"})

                query_embedding = self._embedding_model.encode(str(context_content)).tolist()
                collection = self._collections.get("knowledge")

                if not collection:
                    return jsonify({"valid": False, "reason": "知识层未初始化"})

                resp = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3,
                    include=["metadatas", "distances"]
                )

                best_match = None
                best_similarity = 0
                for i, (meta, dist) in enumerate(zip(resp['metadatas'][0], resp['distances'][0])):
                    similarity = 1 - dist
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = meta

                if best_match and best_similarity > 0.3:
                    if hasattr(context_entry, '__dict__'):
                        context_dict = {
                            "id": context_entry.id,
                            "content": context_entry.content,
                            "timestamp": context_entry.timestamp,
                            "metadata": context_entry.metadata,
                            "weight": getattr(context_entry, "weight", 1.0),
                            "base_weight": getattr(context_entry, "base_weight", 1.0),
                            "last_access": getattr(context_entry, "last_access", context_entry.timestamp),
                            "trigger_count": getattr(context_entry, "trigger_count", 0)
                        }
                    else:
                        context_dict = context_entry

                    return jsonify({
                        "valid": True,
                        "similarity": best_similarity,
                        "candidate": {
                            "context": context_dict,
                            "knowledge": {
                                "content": best_match.get("content", ""),
                                "timestamp": best_match.get("timestamp", 0)
                            },
                            "connection": f"语义相似度: {best_similarity:.3f}"
                        }
                    })
                return jsonify({"valid": False, "reason": "未找到足够相似的连接"})

            except Exception as e:
                logger.error(f"联想功能失败: {e}")
                return jsonify({"valid": False, "reason": str(e)})

        # ===== 验证单元端点 =====

        @self._app.route('/sleep/validate', methods=['POST'])
        def sleep_validate():
            if self.config.role != "verification":
                return jsonify({"error": "Only verification unit can validate"}), 403

            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            candidate = data.get("candidate", {})
            similarity = candidate.get("similarity", 0.0)

            if similarity >= 0.5:
                return jsonify({
                    "valid": True,
                    "reason": "结构价值可识别",
                    "candidate": candidate
                })
            else:
                return jsonify({
                    "valid": False,
                    "reason": f"相似度 {similarity:.3f} 低于阈值 0.5",
                    "candidate": candidate
                })

        # ===== 快照管理端点（记录单元专属） =====
        if self.config.role == "record":
            @self._app.route('/memory/snapshot', methods=['POST'])
            def memory_snapshot():
                """生成情境层快照"""
                data = request.get_json() or {}
                source = data.get("source", "auto")
                window_size = data.get("window_size", 10)

                context_entries = self._memory.get("context", [])
                if not context_entries:
                    return jsonify({"error": "No context entries available"}), 400

                recent = context_entries[-window_size:] if len(context_entries) >= window_size else context_entries

                structures = self._memory.get("structure", [])
                latest_trust = 0.3
                latest_gap = 0.5
                latest_e_weight = 0.0
                if structures:
                    latest = structures[-1]
                    if isinstance(latest, dict):
                        latest_trust = latest.get("content", {}).get("t_total", 0.3)
                        latest_gap = latest.get("content", {}).get("d_norm", 0.5)
                        latest_e_weight = latest.get("content", {}).get("e_weight", 0.0)

                signals = {"trust": latest_trust, "gap": latest_gap, "e_weight": latest_e_weight}

                if "snapshots" not in self._memory:
                    self._memory["snapshots"] = []

                snapshots = self._memory["snapshots"]
                if len(snapshots) >= 50:
                    active = [s for s in snapshots if s.get("status") == "active"]
                    if active:
                        active.sort(key=lambda x: x.get("timestamp", 0))
                        to_remove = [s["id"] for s in active[:10]]
                        self._memory["snapshots"] = [s for s in snapshots if s["id"] not in to_remove]

                snapshot = {
                    "id": f"snapshot_{int(time.time() * 1000)}",
                    "timestamp": time.time(),
                    "source": source,
                    "window_size": window_size,
                    "entries_count": len(recent),
                    "entries": recent[:10],
                    "signals": signals,
                    "layer_boundary": "L1",
                    "status": "active",
                    "consolidated_at": None,
                    "boundary_audit": None
                }
                self._memory["snapshots"].append(snapshot)
                self._save_memory_to_disk()

                logger.info(f"📸 快照已生成: {snapshot['id']}")
                return jsonify({"status": "created", "snapshot": snapshot})

            @self._app.route('/memory/snapshot/list', methods=['GET'])
            def snapshot_list():
                snapshots = self._memory.get("snapshots", [])
                return jsonify({"snapshots": snapshots, "count": len(snapshots)})

            @self._app.route('/memory/snapshot/audit', methods=['POST'])
            def snapshot_audit():
                data = request.get_json()
                snapshot_id = data.get("snapshot_id")
                if not snapshot_id:
                    return jsonify({"error": "snapshot_id is required"}), 400

                snapshots = self._memory.get("snapshots", [])
                l2_keywords = ["推演草稿", "未决分歧", "中间计算", "过程上下文", "临时状态"]
                for s in snapshots:
                    if s.get("id") == snapshot_id:
                        entries = s.get("entries", [])
                        content_str = json.dumps(entries)
                        found = [kw for kw in l2_keywords if kw in content_str]
                        if found:
                            s["boundary_audit"] = {"status": "BOUNDARY_VIOLATION", "reason": f"包含L2特征词: {found}", "audited_at": time.time()}
                            s["status"] = "flagged"
                            self._save_memory_to_disk()
                            return jsonify({"status": "audited", "result": "BOUNDARY_VIOLATION", "reason": f"包含L2特征词: {found}"})
                        else:
                            s["boundary_audit"] = {"status": "PASSED", "audited_at": time.time()}
                            self._save_memory_to_disk()
                            return jsonify({"status": "audited", "result": "PASSED"})
                return jsonify({"error": "Snapshot not found"}), 404

            @self._app.route('/memory/snapshot/consolidate', methods=['POST'])
            def snapshot_consolidate():
                data = request.get_json()
                snapshot_id = data.get("snapshot_id")
                if not snapshot_id:
                    return jsonify({"error": "snapshot_id is required"}), 400
                snapshots = self._memory.get("snapshots", [])
                for s in snapshots:
                    if s.get("id") == snapshot_id:
                        s["status"] = "consolidated"
                        s["consolidated_at"] = time.time()
                        self._save_memory_to_disk()
                        return jsonify({"status": "consolidated"})
                return jsonify({"error": "Snapshot not found"}), 404

    def _get_importance_detail(self) -> Dict:
        result = getattr(self, '_last_importance_result', {})
        if isinstance(result, dict):
            return result
        logger.warning(f"重要性详情格式异常: {type(result)} - {result}")
        return {
            "error": "Invalid importance detail format",
            "raw": str(result),
            "type": str(type(result))
        }
    
    def _calculate_decay_weight(self, entry):
        elapsed_days = (time.time() - entry["last_access"]) / 86400
        decay_factor = 1 / (1 + self._decay_config["decay_rate"] * elapsed_days ** 2)
        weight = entry["base_weight"] * decay_factor
        return max(weight, self._decay_config["min_weight"])

    def _extract_fact(self, text: str, context: Dict = None) -> Dict:
        condition_space = self._identify_condition_space(text, context)
        fact = self._extract_structured_fact(text, condition_space)
        premises = self._verify_premises(fact, condition_space)
        
        return {
            "fact": fact,
            "condition_space": condition_space,
            "premises": premises,
            "verification_status": "verified" if premises.get("all_verified", False) else "pending",
            "timestamp": time.time()
        }

    def _identify_condition_space(self, text: str, context: Dict = None) -> Dict:
        return {
            "observation_position": context.get("position", "记录单元"),
            "observation_tool": "事实抽取器",
            "time_window": time.strftime("%Y-%m-%d %H:%M:%S"),
            "existence_constraint": "协议v2.9框架内"
        }

    def _extract_structured_fact(self, text: str, condition_space: Dict) -> Dict:
        return {
            "type": "observational_fact",
            "content": text,
            "condition_space": condition_space,
            "source": "fact_extract",
            "confidence": 0.7
        }

    def _verify_premises(self, fact: Dict, condition_space: Dict) -> Dict:
        premises = []
        return {
            "premises": premises,
            "all_verified": True,
            "confidence": 0.7
        }

    # ===== 多实例协同核心方法 =====

    def _process_chain_message(self, text: str, signals: dict, history: list, source: str, role: str, chain_id: str, step: int) -> str:
        """根据角色和链步骤处理消息"""
        if role == "record":
            return self._process_record_chain(text, signals, chain_id, step)
        elif role == "reflection":
            return self._process_reflection_chain(text, signals, chain_id, step)
        elif role == "verification":
            return self._process_verification_chain(text, signals, chain_id, step)
        elif role == "output":
            return self._process_output_chain(text, signals, chain_id, step)
        elif role == "life_support":
            return self._process_life_chain(text, signals, chain_id, step)
        elif role == "arbitration":
            return self._process_designer_chain(text, signals, chain_id, step)
        else:
            return f"实例 {role} 收到消息: {text[:50]}..."

    def _process_record_chain(self, text: str, signals: dict, chain_id: str, step: int) -> str:
        """记录单元：检索知识层"""
        results = self._memory.search(text, layer="knowledge", limit=3)
        if results:
            return f"[记录单元] 找到 {len(results)} 条相关信息"
        return f"[记录单元] 未找到相关信息"

    def _process_reflection_chain(self, text: str, signals: dict, chain_id: str, step: int) -> str:
        """反思单元：推演分析，写入结构层"""
        try:
            resp = requests.post("http://localhost:8001/memory/search", json={"query": text, "limit": 5}, timeout=10)
            results = []
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
            
            analysis = "基于记忆推演"
            if results:
                analysis = f"基于 {len(results)} 条相关记忆推演"
            
            try:
                requests.post(
                    "http://localhost:8001/memory/store",
                    json={
                        "layer": "structure",
                        "content": {
                            "type": "chain_node",
                            "chain_id": chain_id,
                            "step": step,
                            "source": "reflection",
                            "analysis": analysis,
                            "details": results[0].get("content", "")[:100] if results else "",
                            "timestamp": time.time()
                        },
                        "metadata": {"source": "reflection_chain", "chain_id": chain_id}
                    },
                    timeout=10
                )
            except Exception:
                pass
            return f"[反思单元] 推演完成: {analysis}"
        except Exception as e:
            logger.debug(f"反思单元链处理失败: {e}")
            return f"[反思单元] 推演完成"

    def _process_verification_chain(self, text: str, signals: dict, chain_id: str, step: int) -> str:
        """验证单元：复核分析，写入结构层"""
        try:
            chain_result = self._read_chain_node(chain_id, step - 1)
            trust = signals.get('trust', 0.5)
            verdict = "通过" if trust >= 0.5 else "需关注"
            
            try:
                requests.post(
                    "http://localhost:8001/memory/store",
                    json={
                        "layer": "structure",
                        "content": {
                            "type": "chain_node",
                            "chain_id": chain_id,
                            "step": step,
                            "source": "verification",
                            "verdict": verdict,
                            "ref_analysis": chain_result.get("analysis", "") if chain_result else "",
                            "timestamp": time.time()
                        },
                        "metadata": {"source": "verification_chain", "chain_id": chain_id}
                    },
                    timeout=10
                )
            except Exception:
                pass
            return f"[验证单元] 复核完成: {verdict}"
        except Exception as e:
            logger.debug(f"验证单元链处理失败: {e}")
            return f"[验证单元] 复核完成"

    def _process_designer_chain(self, text: str, signals: dict, chain_id: str, step: int) -> str:
        """设计者：综合判断，写入结构层"""
        try:
            chain = self._read_full_chain(chain_id)
            
            reflection_content = ""
            verification_content = ""
            for node in chain:
                if node.get("source") == "reflection":
                    reflection_content = node.get("analysis", "")
                elif node.get("source") == "verification":
                    verification_content = node.get("verdict", "")
            
            if reflection_content and verification_content:
                judgment = f"基于推演分析，验证结果为{verification_content}。{reflection_content}"
            elif reflection_content:
                judgment = f"推演分析: {reflection_content}"
            else:
                judgment = f"方向判断: 意图 {signals.get('intent', 0.5):.2f}，正在提供整体方向..."
            
            try:
                requests.post(
                    "http://localhost:8001/memory/store",
                    json={
                        "layer": "structure",
                        "content": {
                            "type": "chain_node",
                            "chain_id": chain_id,
                            "step": step,
                            "source": "designer",
                            "judgment": judgment,
                            "chain_summary": str(chain)[:200] if chain else "",
                            "timestamp": time.time()
                        },
                        "metadata": {"source": "designer_chain", "chain_id": chain_id}
                    },
                    timeout=10
                )
            except Exception:
                pass
            return judgment
        except Exception as e:
            logger.debug(f"设计者链处理失败: {e}")
            return f"方向判断: 意图 {signals.get('intent', 0.5):.2f}，正在提供整体方向..."

    def _process_output_chain(self, text: str, signals: dict, chain_id: str, step: int) -> str:
        """输出单元：翻译最终结果"""
        try:
            designer_node = self._read_chain_node(chain_id, 3)
            if designer_node:
                return designer_node.get("judgment", "设计者已做出判断")
            return "输出单元: 翻译完成"
        except Exception as e:
            logger.debug(f"输出单元链处理失败: {e}")
            return "输出单元: 翻译完成"

    def _process_life_chain(self, text: str, signals: dict, chain_id: str, step: int) -> str:
        """维生系统：生存判断"""
        trust = signals.get('trust', 0.5)
        information_gap = signals.get('information_gap', 0.3)
        status = "稳定"
        if trust < 0.3:
            status = "警惕"
        if information_gap > 0.7:
            status = "危机"
        return f"[维生系统] 状态: {status}"

    def _read_chain_node(self, chain_id: str, step: int) -> Optional[Dict]:
        """读取特定思维链节点的内容"""
        structures = self._memory.get("structure", [])
        for entry in reversed(structures):
            if isinstance(entry, dict):
                content = entry.get("content", {})
            elif hasattr(entry, 'content'):
                content = entry.content
            else:
                continue
            if content.get("type") == "chain_node" and content.get("chain_id") == chain_id and content.get("step") == step:
                return content
        return None

    def _read_full_chain(self, chain_id: str) -> List[Dict]:
        """读取完整思维链"""
        structures = self._memory.get("structure", [])
        chain_nodes = []
        for entry in structures:
            if isinstance(entry, dict):
                content = entry.get("content", {})
            elif hasattr(entry, 'content'):
                content = entry.content
            else:
                continue
            if content.get("type") == "chain_node" and content.get("chain_id") == chain_id:
                chain_nodes.append(content)
        return sorted(chain_nodes, key=lambda x: x.get("step", 0))    

    
    def start(self):
        if self._running:
            return True

        logger.info(f"正在启动实例 {self.config.id} ({self.config.role})，端口 {self.config.port}")

        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self._thread.start()
        logger.info(f"线程已启动，等待服务就绪...")

        for i in range(10):
            time.sleep(0.5)
            try:
                resp = requests.get(f"http://localhost:{self.config.port}/health", timeout=1)
                if resp.status_code == 200:
                    self._running = True
                    logger.info(f"实例 {self.config.id} ({self.config.role}) 启动，端口: {self.config.port}")
                    return True
            except requests.exceptions.ConnectionError:
                logger.debug(f"端口 {self.config.port} 尚未就绪 (尝试 {i+1}/10)")
            except Exception as e:
                logger.error(f"健康检查异常: {e}")

        logger.error(f"实例 {self.config.id} ({self.config.role}) 启动超时")
        return False

    def _run_server(self):
        self._app.run(host='0.0.0.0', port=self.config.port, debug=False, use_reloader=False)

    def stop(self):
        if not self._running:
            return True

        if self.config.role == "record":
            self._save_memory_to_disk()
            logger.info("记忆已保存到磁盘")

        self._running = False
        logger.info(f"实例 {self.config.id} 停止")
        return True

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self._running else "error",
            "instance": self.config.id,
            "role": self.config.role,
            "running": self._running
        }

    def update_trust(self, trust_data: Dict[str, float]):
        for key in ["p_trust", "p_gap", "t_total", "e_weight"]:
            if key in trust_data:
                self._trust_state[key] = trust_data[key]

    def get_trust_state(self) -> Dict[str, float]:
        return self._trust_state.copy()

    def _find_memory_entry(self, layer: str, entry_id: str) -> Optional[Dict]:
        for entry in self._memory.get(layer, []):
            if entry.get("id") == entry_id:
                return entry
        return None

    def _find_memory_entry_by_id(self, entry_id: str) -> Optional[Dict]:
        for layer, entries in self._memory.items():
            for entry in entries:
                if entry.get("id") == entry_id:
                    return entry
        return None

    @property
    def is_running(self) -> bool:
        return self._running