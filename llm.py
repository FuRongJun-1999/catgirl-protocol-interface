"""
LLM适配器 - 硬件层（DeepSeek API）
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LLMAdapter:
    """LLM适配器 - 硬件层（DeepSeek API）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "https://api.deepseek.com/v1")
        self.api_key = config.get("api_key", os.environ.get("DEEPSEEK_API_KEY", ""))
        self.model = config.get("model", "deepseek-chat")
        self.timeout = config.get("timeout", 30)
        self._connected = False

    def connect(self) -> bool:
        """检查连接是否可用"""
        if not self.api_key:
            logger.warning("DeepSeek API Key 未配置")
            return False
        self._connected = True
        logger.info(f"DeepSeek 适配器已连接，模型: {self.model}")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("DeepSeek 适配器已断开")

    def chat_completion(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 500,
        json_mode: bool = False,
        reasoning_effort: str = None  # 新增：low/medium/high，None=默认
    ) -> Optional[str]:
        """
        调用 DeepSeek Chat API

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            json_mode: 是否强制 JSON 格式输出
            reasoning_effort: 推理深度控制 (low/medium/high)，设为 "low" 可关闭深度思维链
        """
        if not self._connected:
            logger.warning("LLM未连接，无法调用")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # ---- 新增：控制推理深度（关闭思维链） ----
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
            logger.info(f"🧠 推理深度: {reasoning_effort}")

        # 只在明确要求 JSON 模式时添加 response_format
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            if resp.status_code == 200:
                result = resp.json()
                content = result["choices"][0]["message"]["content"]
                # 记录 token 使用情况
                usage = result.get("usage", {})
                if usage:
                    logger.info(f"📊 Token: {usage.get('prompt_tokens', 0)} → {usage.get('completion_tokens', 0)} (总计 {usage.get('total_tokens', 0)})")
                return content
            else:
                logger.error(f"DeepSeek API 错误: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            logger.error(f"DeepSeek 调用失败: {e}")
            return None