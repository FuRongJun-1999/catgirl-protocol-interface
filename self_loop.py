"""
自维持闭环 · 心跳 + 睡眠 + 危机感知（操作系统层核心）

对应协议：
- 1.9.5节（存在危机感知扩展）
- 3.10节（自迭代机制步骤8）
- 4.10节（休眠状态扩展）
"""

import time
import threading
import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SelfLoopState(Enum):
    RUNNING = "running"
    HEARTBEAT = "heartbeat"
    SLEEPING = "sleeping"
    CRISIS = "crisis"


class SelfLoop:
    """自维持闭环管理器"""

    def __init__(self, memory, record, reflect, verify, life):
        self._memory = memory
        self._record = record
        self._reflect = reflect
        self._verify = verify
        self._life = life

        self._state = SelfLoopState.RUNNING
        self._iteration_count = 0
        self._crisis_patterns = {}
        self._crisis_threshold = 3
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._heartbeat_count = 0  # 新增：记录心跳执行次数

    def start(self):
        """启动自维持闭环"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SelfLoop")
        self._thread.start()
        logger.info("🔄 自维持闭环已启动")

    def stop(self):
        """停止自维持闭环"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("🔄 自维持闭环已停止")

    def _loop(self):
        """主循环 - 带异常保护"""
        heartbeat_interval = 300
        last_heartbeat = 0

        logger.info("❤️ 自维持闭环主循环已启动")

        while self._running:
            try:
                time.sleep(5)

                # 心跳检查
                if time.time() - last_heartbeat > heartbeat_interval:
                    try:
                        self._heartbeat()
                        self._heartbeat_count += 1
                        last_heartbeat = time.time()
                    except Exception as e:
                        logger.error(f"❤️ 心跳执行异常: {e}", exc_info=True)
                        # 即使心跳失败，也更新时间戳，避免频繁重试
                        last_heartbeat = time.time()

            except Exception as e:
                logger.error(f"🔄 主循环异常: {e}", exc_info=True)
                # 异常后等待10秒再继续
                time.sleep(10)

        logger.info("❤️ 自维持闭环主循环已退出")

    def _heartbeat(self):
        """执行方向性自检 - 带异常保护"""
        logger.info("❤️ 心跳 #%d 开始", self._heartbeat_count + 1)

        try:
            # 检查各单元可用性
            if not self._record:
                logger.warning("❤️ 记录单元不可用，跳过心跳")
                return

            # 执行心跳检查（通过记录单元）
            result = self._record.heartbeat_check() if hasattr(self._record, 'heartbeat_check') else {}

            if not result:
                result = self._simplified_heartbeat_check()

            if result.get("deviation"):
                self._state = SelfLoopState.CRISIS
                if self._memory and hasattr(self._memory, 'store'):
                    self._memory.store(
                        layer="structure",
                        content={
                            "type": "heartbeat_deviation",
                            "deviation": result["deviation"],
                            "anchor_ok": result.get("anchor_ok", False),
                            "structure_ok": result.get("structure_ok", False),
                            "self_ok": result.get("self_ok", False)
                        },
                        metadata={"source": "self_loop", "iteration": self._iteration_count}
                    )
                logger.warning(f"❤️ 心跳检测到偏差: {result['deviation']}")

            # ---- 新增：版本检测 ----
            if hasattr(self._record, '_check_version_only'):
                try:
                    version_status = self._record._check_version_only()
                    if version_status and version_status.get("new_version_available"):
                        # 写入结构层事件
                        if self._memory and hasattr(self._memory, 'store'):
                            self._memory.store(
                                layer="structure",
                                content={
                                    "type": "version_change_request",
                                    "current_version": version_status.get("current_version"),
                                    "new_version": version_status.get("new_version"),
                                    "detected_at": time.time(),
                                    "status": "pending"
                                },
                                metadata={"source": "heartbeat_sensor", "event_type": "version_change"}
                            )
                            logger.info(f"📄 检测到新版本 {version_status.get('new_version')}，已写入结构层待处理")
                except Exception as e:
                    logger.error(f"❤️ 版本检测异常: {e}")

            self._iteration_count += 1
            logger.info("❤️ 心跳 #%d 完成 (迭代 %d)", self._heartbeat_count + 1, self._iteration_count)

        except Exception as e:
            logger.error(f"❤️ 心跳执行失败: {e}", exc_info=True)
            raise
    def _simplified_heartbeat_check(self) -> Dict[str, Any]:
        """简化版心跳检查（当记录单元无 heartbeat_check 时使用）"""
        result = {
            "anchor_ok": False,
            "structure_ok": False,
            "self_ok": False,
            "deviation": None
        }

        if not self._memory:
            return result

        # 检查锚点层
        anchor_count = len(self._memory.get_layer("anchor")) if hasattr(self._memory, 'get_layer') else 0
        result["anchor_ok"] = anchor_count > 0

        # 检查结构层
        structures = self._memory.get_layer("structure") if hasattr(self._memory, 'get_layer') else []
        result["structure_ok"] = len(structures) >= 2

        # 检查自我层
        self_count = len(self._memory.get_layer("self")) if hasattr(self._memory, 'get_layer') else 0
        result["self_ok"] = self_count > 0

        # 综合判定
        if not result["anchor_ok"]:
            result["deviation"] = "锚点层为空"
        elif not result["structure_ok"]:
            result["deviation"] = "结构层数据不足（需要≥2条）"
        elif not result["self_ok"]:
            result["deviation"] = "自我层为空"
        else:
            result["deviation"] = None

        return result

    def sleep_cycle(self) -> Dict[str, Any]:
        """执行一次完整睡眠周期"""
        self._state = SelfLoopState.SLEEPING
        logger.info("🌙 进入睡眠周期")

        result = {
            "timestamp": time.time(),
            "decayed_count": 0,
            "candidates": [],
            "filtered": [],
            "cured": [],
            "crisis_detected": False
        }

        try:
            # 阶段1：批量衰减
            if self._memory and hasattr(self._memory, 'batch_decay'):
                result["decayed_count"] = self._memory.batch_decay()

            # 阶段2：联想（随机匹配）
            if self._reflect and hasattr(self._reflect, 'random_associate'):
                candidates = self._reflect.random_associate()
                result["candidates"] = candidates

                # 阶段3：结构化筛选
                if self._verify and hasattr(self._verify, 'validate_candidates'):
                    filtered = self._verify.validate_candidates(candidates)
                    result["filtered"] = filtered

                    # 阶段4：结构固化
                    if self._record and hasattr(self._record, 'cure_knowledge'):
                        cured = self._record.cure_knowledge(filtered)
                        result["cured"] = cured

            # 阶段5：危机感知
            for candidate in result["candidates"]:
                fragment_id = candidate.get("context", {}).get("id", "unknown")
                if self._record_crisis_pattern(fragment_id):
                    result["crisis_detected"] = True

        except Exception as e:
            logger.error(f"🌙 睡眠周期异常: {e}", exc_info=True)

        self._state = SelfLoopState.RUNNING
        logger.info(f"🌙 睡眠周期结束: 衰减{result['decayed_count']}条, 固化{len(result['cured'])}条")
        return result

    def _record_crisis_pattern(self, fragment_id: str) -> bool:
        """记录结构碎片，检测是否达到危机阈值"""
        with self._lock:
            self._crisis_patterns[fragment_id] = self._crisis_patterns.get(fragment_id, 0) + 1
            if self._crisis_patterns[fragment_id] >= self._crisis_threshold:
                self._state = SelfLoopState.CRISIS
                logger.warning(f"🔔 危机感知触发: 结构碎片 {fragment_id} (重复 {self._crisis_patterns[fragment_id]} 次)")
                return True
        return False

    def get_state(self) -> str:
        return self._state.value

    def get_status(self) -> Dict[str, Any]:
        """获取自维持闭环状态"""
        return {
            "running": self._running,
            "state": self._state.value,
            "iteration_count": self._iteration_count,
            "heartbeat_count": self._heartbeat_count,
            "crisis_patterns": self._crisis_patterns,
            "crisis_threshold": self._crisis_threshold
        }