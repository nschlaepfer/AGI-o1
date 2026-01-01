"""
Test-Time Training (TTT) Infrastructure for ARC-AGI.

Implements per-task model adaptation:
- LoRA fine-tuning on augmented examples
- Concept-based pre-training
- Efficient adaptation strategies
"""

from .adapter import LoRAAdapter, TTTConfig
from .trainer import TestTimeTrainer

__all__ = [
    "LoRAAdapter",
    "TTTConfig",
    "TestTimeTrainer",
]
