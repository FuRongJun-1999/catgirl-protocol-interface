import asyncio
import logging
import threading
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from .instance import ProtocolInstance, InstanceConfig

logger = logging.getLogger(__name__)


class InstanceManager:
    def __init__(self):
        self._instances: Dict[str, ProtocolInstance] = {}
        self._running = False

    def register_instance(self, instance: ProtocolInstance):
        self._instances[instance.config.role] = instance
        logger.info(f"已注册实例: {instance.config.role} ({instance.config.id})")

    def get_instance(self, role: str) -> Optional[ProtocolInstance]:
        return self._instances.get(role)

    def get_all_instances(self) -> List[ProtocolInstance]:
        return list(self._instances.values())

    def start_all(self, startup_order: bool = True) -> Dict[str, bool]:
        results = {}
        instances = self._instances.values()

        if startup_order:
            sorted_instances = sorted(
                instances,
                key=lambda i: i.config.startup_order
            )
            for inst in sorted_instances:
                try:
                    result = inst.start()
                    results[inst.config.role] = result
                except Exception as e:
                    logger.error(f"实例 {inst.config.role} 启动失败: {e}")
                    results[inst.config.role] = False
        else:
            threads = []
            results_lock = threading.Lock()
            for inst in instances:
                def start_instance(instance=inst):
                    try:
                        result = instance.start()
                        with results_lock:
                            results[instance.config.role] = result
                    except Exception as e:
                        logger.error(f"实例 {instance.config.role} 启动失败: {e}")
                        with results_lock:
                            results[instance.config.role] = False
                t = threading.Thread(target=start_instance)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

        self._running = all(results.values())
        return results

    def stop_all(self) -> Dict[str, bool]:
        results = {}
        for inst in self._instances.values():
            try:
                result = inst.stop()
                results[inst.config.role] = result
            except Exception as e:
                logger.error(f"实例 {inst.config.role} 停止失败: {e}")
                results[inst.config.role] = False
        self._running = False
        return results

    def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        for inst in self._instances.values():
            try:
                results[inst.config.role] = inst.health_check()
            except Exception as e:
                results[inst.config.role] = {
                    "status": "error",
                    "error": str(e)
                }
        return results

    def update_trust_for_all(self, role: str, trust_data: Dict[str, float]):
        inst = self._instances.get(role)
        if inst:
            inst.update_trust(trust_data)


    def broadcast_trust(self, source_role: str, trust_data: Dict[str, float]):
        """向所有其他实例广播信任更新"""
        import requests
        for role, instance in self._instances.items():
            if role != source_role:
                try:
                    resp = requests.post(
                        f"http://localhost:{instance.config.port}/trust",
                        json=trust_data,
                        timeout=2
                    )
                    if resp.status_code != 200:
                        logger.warning(f"更新实例 {role} 信任值失败")
                except Exception as e:
                    logger.error(f"广播信任值到 {role} 失败: {e}")        

    @property
    def is_running(self) -> bool:
        return self._running