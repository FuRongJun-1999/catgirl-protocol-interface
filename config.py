# 配置加载模块

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """全局配置"""
    instances: list
    runtime: Dict[str, Any] = field(default_factory=dict)
    degraded_mode: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """从YAML文件加载配置"""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            instances=data.get("instances", []),
            runtime=data.get("runtime", {}),
            degraded_mode=data.get("degraded_mode", {})
        )

    def get_instance_configs(self) -> list:
        """获取实例配置列表"""
        return self.instances

    def get_runtime_config(self) -> Dict[str, Any]:
        """获取运行时配置"""
        return self.runtime

    def is_degraded_mode_enabled(self) -> bool:
        """检查是否启用降级模式"""
        return self.degraded_mode.get("enable", False)

    @classmethod
    def load_thresholds(cls, path: str) -> Dict[str, Any]:
        """加载阈值配置"""
        thresholds_path = Path(path)
        if not thresholds_path.exists():
            raise FileNotFoundError(f"阈值配置文件不存在: {path}")

        with open(thresholds_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


# 配置文件路径示例
CONFIG_PATHS = {
    "instances": "config/instances.yaml",
    "thresholds": "config/thresholds.yaml"
}