"""
API桥接层

连接现有代理服务（src/core/）与协议操作系统内核（src/osi/kernel/）
现有实例仍通过代理服务运行，新增 /osi 前缀接口指向内核
"""

from typing import Dict, Any
import requests


class OSIBridge:
    """内核与代理服务的桥接层"""

    def __init__(self, kernel_memory):
        self._memory = kernel_memory

    def handle_osi_request(self, path: str, method: str, data: Dict) -> Dict:
        """处理 /osi/* 请求"""
        if path == "status":
            return {"status": "ok", "memory": self._memory.summary()}
        elif path == "search":
            query = data.get("query", "")
            layer = data.get("layer")
            limit = data.get("limit", 10)
            results = self._memory.search(query, layer, limit)
            return {
                "results": [
                    {
                        "id": r.id if hasattr(r, "id") else r.get("id"),
                        "layer": r.layer if hasattr(r, "layer") else r.get("layer"),
                        "content": r.content if hasattr(r, "content") else r.get("content")
                    }
                    for r in results
                ]
            }
        elif path == "associate":
            """联想功能：调用记录单元的联想端点"""
            try:
                resp = requests.post(
                    "http://localhost:8001/sleep/random_associate",
                    timeout=10
                )
                if resp.status_code == 200:
                    return resp.json()
                return {"error": "联想请求失败", "status_code": resp.status_code}
            except requests.exceptions.ConnectionError:
                return {"error": "记录单元不可达"}
            except Exception as e:
                return {"error": str(e)}
        elif path == "store":
            layer = data.get("layer")
            content = data.get("content")
            metadata = data.get("metadata", {})
            entry = self._memory.store(layer, content, metadata)
            return {"status": "stored", "id": entry.id if entry else None}
        elif path == "summary":
            return {"summary": self._memory.summary()}
        elif path == "decay":
            decay_rate = data.get("decay_rate", 0.01)
            count = self._memory.batch_decay(decay_rate)
            return {"status": "decayed", "updated_count": count}
        elif path == "protocol":
            """获取协议加载状态"""
            return self._memory.get_protocol_summary()
        else:
            return {"error": f"Unknown /osi path: {path}"}

    def get_memory(self):
        """获取内核记忆系统引用（供内部使用）"""
        return self._memory