"""
记忆系统 · 五层记忆（最小内核）

当前状态：与现有 instance.py 中的 _memory 保持一致
后续扩展：可替换为独立的持久化存储
"""

import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    id: str
    layer: str
    content: Any
    timestamp: float
    metadata: Dict[str, Any]
    weight: float = 1.0
    base_weight: float = 1.0
    last_access: float = None
    trigger_count: int = 0


class MemorySystem:
    """五层记忆系统（最小内核）"""

    def __init__(self, initial_data: Dict[str, List] = None):
        self._layers = initial_data or {
            "anchor": [],
            "structure": [],
            "context": [],
            "knowledge": [],
            "self": [],
            "snapshots": []
        }
        self._decay_config = {
            "rate": 0.01,
            "rebound_gain": 0.5,
            "min_weight": 0.01
        }
        self._normalize_layers()

    def _normalize_layers(self):
        """统一 _layers 中的所有条目为 MemoryEntry 对象"""
        for layer, entries in self._layers.items():
            if layer == "snapshots":
                continue
            normalized = []
            for entry in entries:
                if isinstance(entry, MemoryEntry):
                    normalized.append(entry)
                elif isinstance(entry, dict):
                    normalized.append(MemoryEntry(
                        id=entry.get("id", f"{layer}_{int(time.time() * 1000)}"),
                        layer=entry.get("layer", layer),
                        content=entry.get("content", ""),
                        timestamp=entry.get("timestamp", time.time()),
                        metadata=entry.get("metadata", {}),
                        weight=entry.get("weight", 1.0),
                        base_weight=entry.get("base_weight", 1.0),
                        last_access=entry.get("last_access", time.time()),
                        trigger_count=entry.get("trigger_count", 0)
                    ))
                else:
                    normalized.append(MemoryEntry(
                        id=f"{layer}_{int(time.time() * 1000)}",
                        layer=layer,
                        content=entry,
                        timestamp=time.time(),
                        metadata={},
                        weight=1.0,
                        base_weight=1.0,
                        last_access=time.time(),
                        trigger_count=0
                    ))
            self._layers[layer] = normalized

    def _get_entry_content_type(self, entry) -> Optional[str]:
        try:
            if isinstance(entry, MemoryEntry):
                if isinstance(entry.content, dict):
                    return entry.content.get("type")
                return None
            elif isinstance(entry, dict):
                content = entry.get("content")
                if isinstance(content, dict):
                    return content.get("type")
                return None
        except Exception:
            return None
        return None

    def store(self, layer: str, content: Any, metadata: Dict = None, similarity_threshold: float = 0.85) -> Optional[MemoryEntry]:
        if layer not in self._layers:
            return None

        if layer in ["knowledge", "context"] and isinstance(content, str):
            existing_entries = self._layers.get(layer, [])
            for existing in existing_entries:
                existing_content = existing.content if hasattr(existing, 'content') else existing.get("content", "")
                if not isinstance(existing_content, str):
                    continue
                similarity = self._calculate_similarity(content, existing_content)
                if similarity >= similarity_threshold:
                    if hasattr(existing, 'weight'):
                        existing.weight = min(2.0, existing.weight + 0.1)
                        existing.metadata["duplicate_count"] = existing.metadata.get("duplicate_count", 0) + 1
                        existing.metadata["last_duplicate_at"] = time.time()
                    else:
                        existing["weight"] = min(2.0, existing.get("weight", 1.0) + 0.1)
                        existing["metadata"] = existing.get("metadata", {})
                        existing["metadata"]["duplicate_count"] = existing["metadata"].get("duplicate_count", 0) + 1
                        existing["metadata"]["last_duplicate_at"] = time.time()
                    logger.info(f"🔄 去重: 检测到相似内容 (相似度 {similarity:.3f})，提升权重，不写入新条目")
                    return existing

        entry_id = f"{layer}_{int(time.time() * 1000)}"
        entry = MemoryEntry(
            id=entry_id,
            layer=layer,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
            last_access=time.time()
        )

        if layer == "context":
            entry.weight = 1.0
            entry.base_weight = 1.0
            entry.last_access = time.time()
            entry.trigger_count = 0

        self._layers[layer].append(entry)
        logger.debug(f"📝 新记忆写入: {layer} - {str(content)[:50]}...")
        return entry

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        if text1 in text2 or text2 in text1:
            return 0.95
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def search(self, query: str, layer: str = None, limit: int = 10) -> List[MemoryEntry]:
        """搜索记忆，返回带权重信息的结果"""
        if not query:
            return []
        try:
            import requests
            resp = requests.post(
                "http://localhost:8001/memory/search",
                json={"query": query, "layer": layer, "limit": limit},
                timeout=10
            )
            if resp.status_code != 200:
                logger.warning(f"搜索请求失败: {resp.status_code}")
                return []
            data = resp.json()
            results = data.get("results", [])
            entries = []
            for r in results:
                entry = MemoryEntry(
                    id=r.get("id", "unknown"),
                    layer=r.get("layer", "unknown"),
                    content=r.get("content", ""),
                    timestamp=r.get("timestamp", 0.0),
                    metadata={
                        "source": r.get("source", "unknown"),
                        "weight": r.get("weight", 1.0),
                        "final_score": r.get("final_score", 0.0),
                        "priority_level": r.get("priority_level", "normal")
                    },
                    weight=r.get("weight", 1.0),
                    base_weight=r.get("weight", 1.0),
                    last_access=time.time(),
                    trigger_count=0
                )
                entries.append(entry)
            return entries
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_layer(self, layer: str) -> List[MemoryEntry]:
        return self._layers.get(layer, [])

    def batch_decay(self, decay_rate: float = None) -> int:
        decay_rate = decay_rate or self._decay_config.get("rate", 0.01)
        updated_count = 0
        for entry in self._layers.get("context", []):
            elapsed_days = (time.time() - entry.last_access) / 86400
            old_weight = entry.weight
            new_weight = old_weight / (1 + decay_rate * (elapsed_days ** 2))
            entry.weight = max(new_weight, self._decay_config.get("min_weight", 0.01))
            updated_count += 1
        return updated_count

    def summary(self) -> Dict[str, int]:
        return {layer: len(entries) for layer, entries in self._layers.items()}

    def to_dict(self) -> Dict[str, List[Dict]]:
        return {
            layer: [
                {
                    "id": e.id,
                    "content": e.content,
                    "timestamp": e.timestamp,
                    "metadata": e.metadata,
                    "weight": e.weight,
                    "base_weight": e.base_weight,
                    "last_access": e.last_access,
                    "trigger_count": e.trigger_count
                }
                for e in entries
            ]
            for layer, entries in self._layers.items()
            if layer != "snapshots"
        }

    # ===== 协议文档加载 =====

    def load_protocol_document(self, path: str, layer: str = "anchor", doc_type: str = "theory") -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            logger.warning(f"协议文档不存在: {path}")
            return {"status": "error", "reason": "file_not_found"}
        except Exception as e:
            logger.error(f"读取协议文档失败: {e}")
            return {"status": "error", "reason": str(e)}

        existing = self._layers.get(layer, [])
        target_type = f"protocol_{doc_type}"
        for entry in existing:
            entry_type = self._get_entry_content_type(entry)
            if entry_type == target_type:
                logger.info(f"协议{doc_type}版已加载，跳过")
                return {"status": "skipped", "reason": "already_loaded"}

        entry = self.store(
            layer=layer,
            content={
                "type": target_type,
                "version": "v2.9" if doc_type == "theory" else "v2.9.1",
                "text": content,
                "full_path": str(path)
            },
            metadata={
                "source": "protocol_document",
                "doc_type": doc_type,
                "loaded_at": time.time()
            }
        )
        if entry:
            logger.info(f"✅ 协议{doc_type}版已加载到{layer}层，共{len(content)}字符")
            return {"status": "loaded", "entry_id": entry.id, "char_count": len(content)}
        return {"status": "error", "reason": "store_failed"}

    # ===== 扩展接口 =====

    def get_protocol_context(self, doc_type: str = "theory") -> Optional[str]:
        layer = "anchor" if doc_type == "theory" else "knowledge"
        for entry in self._layers.get(layer, []):
            if not isinstance(entry, MemoryEntry):
                continue
            if isinstance(entry.content, dict) and entry.content.get("type") == f"protocol_{doc_type}":
                return entry.content.get("text", "")
        return None

    def load_protocol_context(self, text: str) -> Optional[MemoryEntry]:
        return self.store(
            layer="anchor",
            content={"type": "protocol_context", "version": "v2.9", "text": text},
            metadata={"source": "protocol_document", "loaded_at": time.time()}
        )

    def get_protocol_summary(self) -> Dict[str, Any]:
        theory = self.get_protocol_context("theory")
        eng = self.get_protocol_context("engineering")
        return {
            "theory_loaded": theory is not None,
            "theory_length": len(theory) if theory else 0,
            "engineering_loaded": eng is not None,
            "engineering_length": len(eng) if eng else 0
        }

    # ===== 特化价值观管理 =====

    def get_specialized_values(self) -> Optional[str]:
        """从结构层读取特化价值观文本"""
        for entry in self._layers.get("structure", []):
            if not isinstance(entry, MemoryEntry):
                continue
            if isinstance(entry.content, dict) and entry.content.get("type") == "specialized_values":
                return entry.content.get("text", "")
            if isinstance(entry, dict):
                content = entry.get("content", {})
                if content.get("type") == "specialized_values":
                    return content.get("text", "")
        return None

    def store_specialized_values(self, text: str, metadata: Dict = None) -> Optional[MemoryEntry]:
        """存储特化价值观到结构层"""
        existing = None
        for entry in self._layers.get("structure", []):
            if isinstance(entry, MemoryEntry):
                if isinstance(entry.content, dict) and entry.content.get("type") == "specialized_values":
                    existing = entry
                    break
            elif isinstance(entry, dict):
                content = entry.get("content", {})
                if content.get("type") == "specialized_values":
                    existing = entry
                    break

        if existing:
            if isinstance(existing, MemoryEntry):
                existing.content["text"] = text
                existing.metadata["updated_at"] = time.time()
                if metadata:
                    existing.metadata.update(metadata)
            else:
                existing["content"]["text"] = text
                existing["metadata"]["updated_at"] = time.time()
                if metadata:
                    existing["metadata"].update(metadata)
            logger.info("🔄 特化价值观已更新")
            return existing

        return self.store(
            layer="structure",
            content={
                "type": "specialized_values",
                "text": text,
                "version": "v2.9"
            },
            metadata={
                "source": "specialized_values",
                "loaded_at": time.time(),
                **(metadata or {})
            }
        )

    # ===== 快照管理方法 =====

    def get_snapshots(self, status: str = None) -> List[Dict]:
        snapshots = self._layers.get("snapshots", [])
        if status:
            return [s for s in snapshots if s.get("status") == status]
        return snapshots

    def create_snapshot(self, entries: List[Dict], signals: Dict, source: str = "auto", window_size: int = 10) -> Dict:
        snapshots = self._layers.get("snapshots", [])
        if len(snapshots) >= 50:
            active = [s for s in snapshots if s.get("status") == "active"]
            if active:
                active.sort(key=lambda x: x.get("timestamp", 0))
                to_remove = [s["id"] for s in active[:10]]
                self._layers["snapshots"] = [s for s in snapshots if s["id"] not in to_remove]
                logger.info(f"淘汰了 {len(to_remove)} 个旧快照")

        snapshot = {
            "id": f"snapshot_{int(time.time() * 1000)}",
            "timestamp": time.time(),
            "source": source,
            "window_size": window_size,
            "entries_count": len(entries),
            "entries": entries[:10],
            "signals": signals,
            "layer_boundary": "L1",
            "status": "active",
            "consolidated_at": None,
            "boundary_audit": None
        }
        self._layers["snapshots"].append(snapshot)
        return snapshot

    def update_snapshot(self, snapshot_id: str, updates: Dict) -> bool:
        snapshots = self._layers.get("snapshots", [])
        for s in snapshots:
            if s.get("id") == snapshot_id:
                s.update(updates)
                return True
        return False

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict]:
        snapshots = self._layers.get("snapshots", [])
        for s in snapshots:
            if s.get("id") == snapshot_id:
                return s
        return None

    def audit_snapshot(self, snapshot_id: str, l2_keywords: List[str] = None) -> Dict:
        if l2_keywords is None:
            l2_keywords = ["推演草稿", "未决分歧", "中间计算", "过程上下文", "临时状态"]
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            return {"status": "error", "reason": "Snapshot not found"}

        entries = snapshot.get("entries", [])
        content_str = json.dumps(entries)
        found = [kw for kw in l2_keywords if kw in content_str]

        if found:
            self.update_snapshot(snapshot_id, {
                "boundary_audit": {"status": "BOUNDARY_VIOLATION", "reason": f"包含L2特征词: {found}", "audited_at": time.time()},
                "status": "flagged"
            })
            return {"status": "BOUNDARY_VIOLATION", "reason": f"包含L2特征词: {found}"}
        else:
            self.update_snapshot(snapshot_id, {
                "boundary_audit": {"status": "PASSED", "audited_at": time.time()}
            })
            return {"status": "PASSED"}

    def consolidate_snapshot(self, snapshot_id: str) -> bool:
        return self.update_snapshot(snapshot_id, {"status": "consolidated", "consolidated_at": time.time()})