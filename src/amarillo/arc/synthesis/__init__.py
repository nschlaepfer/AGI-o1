"""
Synthetic Data Generation for ARC-AGI.

Generates ARC-like tasks for:
- Pre-training models
- Test-time training augmentation
- Concept learning
"""

from .generator import SyntheticGenerator, ConceptGenerator
from .augmentation import augment_task, generate_augmentations

__all__ = [
    "SyntheticGenerator",
    "ConceptGenerator",
    "augment_task",
    "generate_augmentations",
]
