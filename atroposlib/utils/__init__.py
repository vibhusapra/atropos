"""
Utility functions and classes for the atroposlib package.
"""

from .config_handler import ConfigHandler
from .message_history_utils import (
    strip_thinking,
    truncate_thinking,
    ensure_trajectory_token_limit,
)
from .tokenize_for_trainer import tokenize_for_trainer
from .tool_call_parser import parse_tool_call
from .advantages import (
    allclose_to_first,
    compute_stats,
    compute_discounted_returns,
    compute_grpo_process_supervision_advantages,
)

__all__ = [
    "ConfigHandler",
    "strip_thinking",
    "truncate_thinking",
    "tokenize_for_trainer",
    "parse_tool_call",
    "allclose_to_first",
    "compute_stats",
    "compute_discounted_returns",
    "compute_grpo_process_supervision_advantages",
    "ensure_trajectory_token_limit",
]
