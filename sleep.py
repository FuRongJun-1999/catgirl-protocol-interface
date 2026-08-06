"""
睡眠状态管理器（维生系统）
负责调度睡眠周期的进入、执行和唤醒
"""

import time
import threading
import logging
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SleepPhase(Enum):
    IDLE = "idle"
    ENTERING = "entering"
    BATCH_DECAY = "decay"
    RANDOM_ASSOCIATION = "random"
    FACT_EXTRACTION = "fact"
    STRUCTURAL_FILTER = "filter"
    STRUCTURAL_CURE = "cure"
    EXITING = "exiting"


@dataclass
class SleepConfig:
    idle_timeout: int = 300
    min_sleep_duration: int = 60
    max_sleep_duration: int = 3600
    decay_rate: float = 0.01
    rebound_gain: float = 0.5
    min_candidates: int = 1
    max_invalid_matches: int = 10
    unit_active_levels: Dict[str, float] = field(default_factory=lambda: {
        "reflection": 0.5,
        "verification": 0.75,
        "record": 0.3,
        "life_support": 1.0
    })


class SleepManager:
    def __init__(self, config: Optional[SleepConfig] = None):
        self.config = config or SleepConfig()
        self._phase: SleepPhase = SleepPhase.IDLE
        self._running: bool = False
        self._last_interaction: float = time.time()
        self._wake_signal: threading.Event = threading.Event()
        self._lock = threading.Lock()
        self._current_cycle_candidates: int = 0
        self._consecutive_empty_cycles: int = 0
        self._reflection_api = None
        self._record_api = None
        self._verification_api = None
        self._trust_api = None
        self._crisis_patterns = {
            "fear_patterns": [],
            "repetition_counts": {},
            "last_trigger": 0
        }
        self._crisis_threshold = 3

    def register_apis(self, reflection, record, verification, trust=None):
        self._reflection_api = reflection
        self._record_api = record
        self._verification_api = verification
        self._trust_api = trust

    def record_interaction(self):
        self._last_interaction = time.time()
        if self._phase != SleepPhase.IDLE:
            self._wake()

    def _should_enter_sleep(self) -> bool:
        if self._phase != SleepPhase.IDLE:
            return False
        idle_duration = time.time() - self._last_interaction
        return idle_duration >= self.config.idle_timeout

    def start(self):
        self._running = True
        self._sleep_loop()

    def stop(self):
        self._running = False
        self._wake_signal.set()

    def _sleep_loop(self):
        while self._running:
            if self._should_enter_sleep():
                self._enter_sleep()
            time.sleep(5)

    # ==================== 睡眠周期各阶段 ====================

    def _batch_decay(self) -> Dict:
        logger.info("   📉 执行批量衰减...")
        try:
            resp = requests.post(
                "http://localhost:8001/sleep/batch_decay",
                json={"decay_rate": self.config.decay_rate},
                timeout=30
            )
            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"   ✅ 批量衰减完成: {result}")
                return result
            else:
                logger.error(f"   ❌ 批量衰减失败: {resp.status_code}")
                return {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            logger.error(f"   ❌ 批量衰减失败: {e}")
            return {"error": str(e)}

    def _random_association(self) -> List[Dict[str, Any]]:
        """快照联想：从情境层快照中选取候选"""
        logger.info("   🔀 执行快照联想...")
        candidates = []
        try:
            resp = requests.get(
                "http://localhost:8001/memory/snapshot/list",
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                snapshots = data.get("snapshots", [])
                active = [s for s in snapshots if s.get("status") == "active"]

                if not active:
                    logger.info("   ⏭️ 无活跃快照，跳过联想")
                    return candidates

                snapshot = random.choice(active)
                entries = snapshot.get("entries", [])
                if not entries:
                    return candidates

                context_text = " ".join([e.get("content", "") for e in entries[:5]])

                knowledge_entries = self._record_api._memory.get("knowledge", []) if self._record_api else []
                if not knowledge_entries:
                    return candidates

                keywords = context_text.split()[:10]
                matched = []
                for k in knowledge_entries:
                    content = k.get("content", "")
                    if any(kw in content for kw in keywords):
                        matched.append(k)

                if matched:
                    candidates.append({
                        "valid": True,
                        "similarity": 0.5,
                        "candidate": {
                            "context": {"content": context_text[:200]},
                            "knowledge": {"content": matched[0].get("content", "")},
                            "connection": f"快照联想: {context_text[:50]}..."
                        },
                        "source": "snapshot",
                        "snapshot_id": snapshot.get("id")
                    })
                    self._current_cycle_candidates += 1
                    logger.info(f"   ✅ 从快照 {snapshot.get('id')[:12]} 找到候选")

        except Exception as e:
            logger.error(f"   ❌ 快照联想失败: {e}")
        return candidates

    def _fact_extraction(self) -> List[Dict[str, Any]]:
        logger.info("   📋 执行事实抽取...")
        if not self._record_api:
            logger.warning("   ⚠️ 记录单元不可用，跳过事实抽取")
            return []

        context_entries = self._record_api._memory.get("context", []) if hasattr(self._record_api, '_memory') else []
        if not context_entries:
            logger.info("   ⏭️ 无情境层数据，跳过事实抽取")
            return []

        recent_contexts = context_entries[-5:] if context_entries else []
        facts = []

        for ctx in recent_contexts:
            if isinstance(ctx, dict):
                content = ctx.get("content", "")
                if ctx.get("metadata", {}).get("fact_extracted", False):
                    continue
            else:
                content = ctx.content if hasattr(ctx, 'content') else ""
                if ctx.metadata.get("fact_extracted", False):
                    continue

            if not content or len(content) < 5:
                continue

            try:
                resp = requests.post(
                    "http://localhost:8001/fact/extract",
                    json={"text": content, "context": {"source": "sleep_cycle"}},
                    timeout=30
                )
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("status") == "stored":
                        facts.append({
                            "valid": True,
                            "similarity": 0.6,
                            "candidate": {
                                "context": {"content": content},
                                "knowledge": {"content": result.get("fact", {}).get("content", content)},
                                "connection": f"事实抽取: {content[:50]}"
                            },
                            "source": "fact_extraction",
                            "fact_result": result
                        })
                        logger.info(f"   ✅ 事实抽取成功: {content[:30]}...")
                        if isinstance(ctx, dict):
                            ctx["metadata"] = ctx.get("metadata", {})
                            ctx["metadata"]["fact_extracted"] = True
                        else:
                            ctx.metadata["fact_extracted"] = True
                    else:
                        logger.warning(f"   ⚠️ 事实抽取返回异常: {result}")
                else:
                    logger.warning(f"   ⚠️ 事实抽取失败: {resp.status_code}")
            except Exception as e:
                logger.error(f"   ❌ 事实抽取异常: {e}")

        return facts

    def _structural_filter(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"   🔍 结构化筛选 {len(candidates)} 个候选...")
        filtered = []
        for idx, candidate in enumerate(candidates):
            similarity = candidate.get("similarity", 0.0)

            if candidate.get("source") == "fact_extraction":
                filtered.append(candidate)
                logger.info(f"   ✅ 事实抽取候选直接通过")
                continue

            if similarity < 0.3:
                logger.info(f"   ⏭️ 相似度过低 ({similarity:.3f})，跳过")
                continue

            try:
                logger.info(f"   📤 发送候选 {idx+1} 到验证单元...")
                resp = requests.post(
                    "http://localhost:8003/sleep/validate",
                    json={"candidate": candidate},
                    timeout=10
                )
                logger.info(f"   📥 验证响应状态码: {resp.status_code}")
                if resp.status_code == 200:
                    result = resp.json()
                    logger.info(f"   📥 验证响应内容: {result}")
                    if result.get("valid", False):
                        filtered.append(candidate)
                        logger.info(f"   ✅ 候选通过验证")
                    else:
                        logger.info(f"   ⏭️ 候选未通过验证: {result.get('reason', '未知原因')}")
                else:
                    logger.warning(f"   ⚠️ 验证请求失败: {resp.status_code}")
            except Exception as e:
                logger.error(f"   ❌ 筛选候选失败: {e}")

        logger.info(f"   ✅ 筛选完成，通过 {len(filtered)} 个候选")
        return filtered

    def _structural_cure(self, filtered: List[Dict[str, Any]]) -> Dict:
        if not filtered:
            return {"status": "skipped", "reason": "无有效候选"}

        logger.info(f"   📌 固化 {len(filtered)} 个候选到知识层...")
        consolidated = []
        for candidate in filtered:
            try:
                connection = candidate.get("candidate", {}).get("connection", "")
                if not connection:
                    connection = f"关联: {candidate.get('candidate', {}).get('context', {}).get('content', '')} ↔ {candidate.get('candidate', {}).get('knowledge', {}).get('content', '')}"

                if candidate.get("source") == "fact_extraction":
                    fact_result = candidate.get("fact_result", {})
                    connection = fact_result.get("fact", {}).get("content", connection)

                resp = requests.post(
                    "http://localhost:8001/memory/store",
                    json={
                        "layer": "knowledge",
                        "content": connection,
                        "metadata": {
                            "source": candidate.get("source", "sleep_cure"),
                            "candidate": candidate,
                            "similarity": candidate.get("similarity", 0.0),
                            "verified": True,
                            "timestamp": time.time()
                        }
                    },
                    timeout=10
                )
                if resp.status_code == 200:
                    result = resp.json()
                    entry_id = result.get("entry", {}).get("id")
                    consolidated.append(entry_id)
                    logger.info(f"   ✅ 固化成功: {entry_id}")
                    self._update_trust_after_consolidation(candidate)
                else:
                    logger.error(f"   ❌ 固化失败: {resp.status_code}")
            except Exception as e:
                logger.error(f"   ❌ 固化失败: {e}")

        return {
            "status": "completed",
            "consolidated_count": len(consolidated),
            "entry_ids": consolidated
        }

    def _update_trust_after_consolidation(self, candidate: Dict):
        try:
            similarity = candidate.get("similarity", 0.0)
            if candidate.get("source") == "fact_extraction":
                trust_delta = min(0.06, 0.04 + similarity * 0.03)
            else:
                trust_delta = min(0.05, similarity * 0.05)

            resp = requests.post(
                "http://localhost:8000/trust/update",
                json={"delta": trust_delta, "reason": "consolidation"},
                timeout=5
            )
            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"   📈 信任值更新: {result.get('previous_trust', 0):.3f} → {result.get('new_trust', 0):.3f}")
            else:
                logger.warning(f"   ⚠️ 信任值更新失败: {resp.status_code}")
        except Exception as e:
            logger.debug(f"信任值更新跳过: {e}")

    def _check_output_quality(self):
        if self._current_cycle_candidates >= self.config.min_candidates:
            self._consecutive_empty_cycles = 0
        else:
            self._consecutive_empty_cycles += 1
            logger.warning(f"   ⚠️ 产出不足，连续 {self._consecutive_empty_cycles} 周期未达标")

        if self._consecutive_empty_cycles >= 3 and self._verification_api:
            logger.warning("   🔔 连续3周期未达标，触发验证单元检测")
            try:
                requests.post(
                    "http://localhost:8003/sleep/detect",
                    json={"reason": "sleep_output_low", "consecutive": self._consecutive_empty_cycles},
                    timeout=5
                )
            except Exception:
                pass

    def _crisis_perception(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for candidate in candidates:
            if candidate.get("source") == "fact_extraction":
                continue

            context_content = candidate.get("candidate", {}).get("context", {}).get("content", "")
            knowledge_content = candidate.get("candidate", {}).get("knowledge", {}).get("content", "")
            similarity = candidate.get("similarity", 0.0)

            is_crisis = False
            crisis_type = None

            if similarity > 0.7 and ("不" in context_content and "不" in knowledge_content):
                is_crisis = True
                crisis_type = "contradiction"

            emotion_keywords = ["焦虑", "恐惧", "担心", "不安", "孤独", "无助"]
            if any(kw in context_content for kw in emotion_keywords) and "信任" in knowledge_content:
                is_crisis = True
                crisis_type = "emotional_mismatch"

            if "不是" in context_content and "协议" in knowledge_content:
                is_crisis = True
                crisis_type = "self_negation"

            if is_crisis:
                fragment_id = f"{crisis_type}_{context_content[:20]}"
                self._crisis_patterns["repetition_counts"][fragment_id] = \
                    self._crisis_patterns["repetition_counts"].get(fragment_id, 0) + 1

                candidate["crisis_flag"] = True
                candidate["crisis_type"] = crisis_type
                candidate["repetition_count"] = self._crisis_patterns["repetition_counts"][fragment_id]

                logger.warning(f"⚠️ 检测到危机模式: {crisis_type} (重复 {candidate['repetition_count']} 次)")

                if self._crisis_patterns["repetition_counts"][fragment_id] >= self._crisis_threshold:
                    self._trigger_preventive_calibration(fragment_id, candidate)
            else:
                candidate["crisis_flag"] = False

        return candidates

    def _trigger_preventive_calibration(self, fragment_id: str, candidate: Dict[str, Any]):
        logger.warning(f"🔔 预防性校准触发: 结构碎片 {fragment_id} 重复出现 {self._crisis_threshold} 次")

        try:
            resp = requests.post(
                "http://localhost:8003/sleep/detect",
                json={
                    "reason": "preventive_calibration",
                    "fragment_id": fragment_id,
                    "candidate": candidate,
                    "crisis_type": candidate.get("crisis_type", "unknown")
                },
                timeout=5
            )
            if resp.status_code == 200:
                logger.info(f"✅ 验证单元已收到预防性校准请求")
            else:
                logger.warning(f"⚠️ 验证单元响应异常: {resp.status_code}")
        except Exception as e:
            logger.error(f"❌ 通知验证单元失败: {e}")

        try:
            resp = requests.post(
                "http://localhost:8001/memory/store",
                json={
                    "layer": "structure",
                    "content": {
                        "type": "crisis_pattern",
                        "fragment_id": fragment_id,
                        "candidate": candidate,
                        "crisis_type": candidate.get("crisis_type", "unknown"),
                        "repetition_count": self._crisis_threshold,
                        "timestamp": time.time()
                    },
                    "metadata": {
                        "source": "crisis_perception",
                        "severity": "warning"
                    }
                },
                timeout=10
            )
            if resp.status_code == 200:
                logger.info(f"✅ 危机模式已记录到结构层")
        except Exception as e:
            logger.error(f"❌ 记录危机模式失败: {e}")

        self._crisis_patterns["repetition_counts"][fragment_id] = 0

    # ==================== 睡眠周期入口 ====================

    def _enter_sleep(self):
        with self._lock:
            self._phase = SleepPhase.ENTERING
            logger.info("🌙 进入睡眠状态")
            self._current_cycle_candidates = 0

        try:
            self._phase = SleepPhase.BATCH_DECAY
            self._batch_decay()

            self._phase = SleepPhase.RANDOM_ASSOCIATION
            candidates = self._random_association()

            self._phase = SleepPhase.FACT_EXTRACTION
            facts = self._fact_extraction()
            for fact in facts:
                candidates.append(fact)

            candidates = self._crisis_perception(candidates)

            self._phase = SleepPhase.STRUCTURAL_FILTER
            filtered = self._structural_filter(candidates)

            self._phase = SleepPhase.STRUCTURAL_CURE
            cure_result = self._structural_cure(filtered)

            self._check_output_quality()

            logger.info(f"💤 睡眠周期完成，产出 {len(filtered)} 个有效候选，固化 {cure_result.get('consolidated_count', 0)} 条")

        except Exception as e:
            logger.error(f"❌ 睡眠状态异常: {e}")

        finally:
            self._phase = SleepPhase.IDLE
            self._wake_signal.clear()

    def run_cycle(self) -> Dict[str, Any]:
        logger.info("🔄 手动触发睡眠周期...")
        with self._lock:
            self._phase = SleepPhase.ENTERING
            self._current_cycle_candidates = 0

        try:
            decay_result = self._batch_decay()

            candidates = self._random_association()

            self._phase = SleepPhase.FACT_EXTRACTION
            facts = self._fact_extraction()
            for fact in facts:
                candidates.append(fact)

            candidates = self._crisis_perception(candidates)

            filtered = self._structural_filter(candidates)
            cure_result = self._structural_cure(filtered)
            self._check_output_quality()

            return {
                "status": "completed",
                "decay": decay_result,
                "candidates_found": len(candidates),
                "fact_extracted": len(facts),
                "filtered": len(filtered),
                "consolidated": cure_result,
                "consecutive_empty": self._consecutive_empty_cycles
            }
        except Exception as e:
            logger.error(f"睡眠周期异常: {e}")
            return {"status": "error", "reason": str(e)}
        finally:
            self._phase = SleepPhase.IDLE
            self._wake_signal.clear()

    def _wake(self):
        if self._phase == SleepPhase.IDLE:
            return
        logger.info("☀️ 唤醒")
        self._phase = SleepPhase.EXITING
        self._phase = SleepPhase.IDLE
        self._wake_signal.clear()

    def force_wake(self):
        self._wake_signal.set()

    def get_phase(self) -> str:
        return self._phase.value

    def is_sleeping(self) -> bool:
        return self._phase not in [SleepPhase.IDLE, SleepPhase.EXITING]