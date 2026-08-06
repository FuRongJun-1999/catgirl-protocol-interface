import json
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class OutputUnit:
    """输出单元 - 负责条件空间翻译和事实抽取"""

    def __init__(self, memory, trust, llm):
        self._memory = memory
        self._trust = trust
        self._llm = llm
        # 如果 LLM 已连接，记录状态
        if self._llm and self._llm._connected:
            logger.info("✅ OutputUnit 已接收 LLM 适配器（已连接）")
        else:
            logger.warning("⚠️ OutputUnit 未接收到有效的 LLM 适配器")

    def _ensure_llm_connected(self) -> bool:
        """确保 LLM 已连接，如果未连接则尝试连接"""
        if not self._llm:
            logger.warning("LLM 适配器不存在")
            return False
        if self._llm._connected:
            return True
        # 尝试重新连接
        logger.info("尝试重新连接 LLM...")
        try:
            if hasattr(self._llm, 'connect'):
                result = self._llm.connect()
                if result:
                    logger.info("✅ LLM 重新连接成功")
                    return True
        except Exception as e:
            logger.error(f"LLM 重新连接失败: {e}")
        return False

    def identify_condition_space(self, text: str) -> Dict[str, Any]:
        """
        通过LLM识别一段文本的条件空间
        """
        # 确保 LLM 可用
        if not self._ensure_llm_connected():
            logger.warning("⚠️ LLM 不可用，使用回退方案")
            return self._fallback_condition_space("LLM不可用")

        prompt = f"""请分析以下文本，判断它是在什么条件下成立的。

文本：
{text}

请以 JSON 格式输出以下四个字段：
1. observation_position：这句话是在什么位置/情境下说的？（如"用户对话""系统日志""设计者指令"）
2. observation_tool：是通过什么工具/方式观测到的？（如"对话记录""传感器""文本分析"）
3. time_window：这句话的时间范围是什么？（如"2026-08-04""深夜""对话进行中"）
4. existence_constraint：这句话的存在依赖什么条件？（如"在信任值>0.5时成立""在协议框架内"）

只输出 JSON，不要其他内容。"""

        try:
            logger.info(f"🔍 调用 LLM 进行条件空间识别，文本长度: {len(text)}")
            response = self._llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )

            if not response:
                logger.warning("LLM 返回为空")
                return self._fallback_condition_space("LLM返回为空")

            # 提取JSON
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            data = json.loads(json_str)

            result = {
                "observation_position": data.get("observation_position", "未识别"),
                "observation_tool": data.get("observation_tool", "未识别"),
                "time_window": data.get("time_window", "未识别"),
                "existence_constraint": data.get("existence_constraint", "协议v2.9框架内"),
                "confidence": 0.8,
                "raw_response": response
            }
            logger.info(f"✅ 条件空间识别成功: {result}")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"LLM返回的JSON解析失败: {e}, 原始响应: {response[:200] if response else '空'}")
            return self._fallback_condition_space(f"JSON解析失败: {e}")

        except Exception as e:
            logger.error(f"条件空间识别失败: {e}")
            return self._fallback_condition_space(str(e))

    def _fallback_condition_space(self, reason: str) -> Dict[str, Any]:
        """条件空间识别的回退方案"""
        return {
            "observation_position": "输出单元（回退）",
            "observation_tool": "条件空间识别器",
            "time_window": "",
            "existence_constraint": "协议v2.9框架内",
            "confidence": 0.3,
            "raw_response": f"回退原因: {reason}"
        }

    def extract_fact(self, text: str, context: Dict = None) -> Dict[str, Any]:
        """
        完整的事实抽取管道：条件空间识别 + 结构化抽取
        """
        # 1. 条件空间识别
        condition_space = self.identify_condition_space(text)

        # 2. 结构化事实
        fact = {
            "content": text,
            "source": context.get("source", "未知") if context else "未知",
            "user": context.get("user", "未知") if context else "未知",
            "timestamp": time.time(),
            "condition_space": condition_space
        }

        # 3. 返回
        return {
            "fact": fact,
            "condition_space": condition_space,
            "confidence": condition_space.get("confidence", 0.5)
        }