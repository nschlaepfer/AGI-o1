"""
Model Adapters for Test-Time Training.

Provides lightweight adaptation mechanisms for LLMs:
- LoRA-style parameter-efficient fine-tuning
- Prompt tuning
- In-context learning optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTTConfig:
    """Configuration for test-time training."""
    # Model settings
    base_model: str = "qwen2.5-4b"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: float = 32.0

    # Training settings
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 4
    warmup_steps: int = 10

    # Augmentation
    augmentation_multiplier: int = 8
    use_geometric_aug: bool = True
    use_color_aug: bool = True
    use_loo_aug: bool = True

    # Regularization
    weight_decay: float = 0.01
    dropout: float = 0.1
    gradient_clip: float = 1.0

    # Efficiency
    max_sequence_length: int = 2048
    use_gradient_checkpointing: bool = True
    mixed_precision: bool = True

    # Early stopping
    patience: int = 3
    min_improvement: float = 0.01


@dataclass
class LoRAAdapter:
    """
    LoRA (Low-Rank Adaptation) for efficient fine-tuning.

    Based on "LoRA: Low-Rank Adaptation of Large Language Models"

    For ARC TTT, we adapt only a few layers with small rank
    to enable fast per-task adaptation.
    """
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # State
    weights_a: Dict[str, np.ndarray] = field(default_factory=dict)
    weights_b: Dict[str, np.ndarray] = field(default_factory=dict)

    def initialize(self, model_config: Dict[str, Any]) -> None:
        """Initialize LoRA weights for target modules."""
        hidden_size = model_config.get("hidden_size", 4096)

        for module_name in self.target_modules:
            # A: hidden_size -> rank (initialized with kaiming)
            self.weights_a[module_name] = np.random.randn(
                hidden_size, self.rank
            ).astype(np.float32) * np.sqrt(2.0 / hidden_size)

            # B: rank -> hidden_size (initialized to zero)
            self.weights_b[module_name] = np.zeros(
                (self.rank, hidden_size), dtype=np.float32
            )

        logger.info(f"Initialized LoRA with rank {self.rank} for {len(self.target_modules)} modules")

    def apply_lora(
        self,
        module_name: str,
        base_output: np.ndarray
    ) -> np.ndarray:
        """Apply LoRA adaptation to module output."""
        if module_name not in self.weights_a:
            return base_output

        # LoRA: output = base_output + (x @ A @ B) * (alpha / rank)
        # Simplified: we apply post-hoc scaling
        a = self.weights_a[module_name]
        b = self.weights_b[module_name]
        scaling = self.alpha / self.rank

        # Assuming base_output has shape [..., hidden_size]
        # This is simplified - real implementation would intercept forward pass
        delta = np.matmul(np.matmul(base_output, a), b) * scaling
        return base_output + delta

    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        total = 0
        for module_name in self.target_modules:
            if module_name in self.weights_a:
                total += self.weights_a[module_name].size
                total += self.weights_b[module_name].size
        return total

    def save(self, path: str) -> None:
        """Save adapter weights."""
        np.savez(
            path,
            **{f"a_{k}": v for k, v in self.weights_a.items()},
            **{f"b_{k}": v for k, v in self.weights_b.items()},
            config={
                "rank": self.rank,
                "alpha": self.alpha,
                "dropout": self.dropout,
                "target_modules": self.target_modules,
            }
        )

    @classmethod
    def load(cls, path: str) -> "LoRAAdapter":
        """Load adapter from file."""
        data = np.load(path, allow_pickle=True)
        config = data["config"].item()

        adapter = cls(
            rank=config["rank"],
            alpha=config["alpha"],
            dropout=config["dropout"],
            target_modules=config["target_modules"],
        )

        for key in data.files:
            if key.startswith("a_"):
                module_name = key[2:]
                adapter.weights_a[module_name] = data[key]
            elif key.startswith("b_"):
                module_name = key[2:]
                adapter.weights_b[module_name] = data[key]

        return adapter


class PromptTuner:
    """
    Prompt tuning for test-time adaptation.

    Learns soft prompt embeddings that are prepended to inputs
    to adapt model behavior for specific tasks.
    """

    def __init__(
        self,
        num_tokens: int = 20,
        embedding_dim: int = 4096,
        init_from_vocab: bool = True,
    ):
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.init_from_vocab = init_from_vocab

        self.prompt_embeddings: Optional[np.ndarray] = None

    def initialize(self, vocab_embeddings: Optional[np.ndarray] = None) -> None:
        """Initialize prompt embeddings."""
        if self.init_from_vocab and vocab_embeddings is not None:
            # Initialize from random vocab embeddings
            indices = np.random.choice(
                len(vocab_embeddings), self.num_tokens, replace=False
            )
            self.prompt_embeddings = vocab_embeddings[indices].copy()
        else:
            # Random initialization
            self.prompt_embeddings = np.random.randn(
                self.num_tokens, self.embedding_dim
            ).astype(np.float32) * 0.01

    def get_prompt_embeddings(self) -> np.ndarray:
        """Get current prompt embeddings."""
        return self.prompt_embeddings

    def update(self, gradients: np.ndarray, learning_rate: float) -> None:
        """Update prompt embeddings with gradients."""
        self.prompt_embeddings -= learning_rate * gradients


class InContextOptimizer:
    """
    Optimize in-context examples for better performance.

    Strategies:
    - Example ordering
    - Example selection
    - Format optimization
    """

    def __init__(self, max_examples: int = 5):
        self.max_examples = max_examples

    def optimize_examples(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        task_description: str = "",
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Optimize which examples to include and their order.

        Strategies:
        1. Diversity: Include examples with different patterns
        2. Complexity: Order from simple to complex
        3. Coverage: Ensure all colors/patterns are represented
        """
        if len(examples) <= self.max_examples:
            return examples

        # Score examples by complexity (simpler first)
        scored = []
        for i, (inp, out) in enumerate(examples):
            complexity = self._compute_complexity(inp, out)
            scored.append((complexity, i, (inp, out)))

        # Sort by complexity
        scored.sort(key=lambda x: x[0])

        # Select diverse subset
        selected = []
        seen_patterns = set()

        for complexity, idx, example in scored:
            if len(selected) >= self.max_examples:
                break

            pattern_hash = self._hash_pattern(example[1])
            if pattern_hash not in seen_patterns:
                selected.append(example)
                seen_patterns.add(pattern_hash)

        # Fill remaining slots
        for complexity, idx, example in scored:
            if len(selected) >= self.max_examples:
                break
            if example not in selected:
                selected.append(example)

        return selected

    def _compute_complexity(
        self,
        input_arr: np.ndarray,
        output_arr: np.ndarray
    ) -> float:
        """Compute complexity score for an example."""
        # Size
        size_score = input_arr.size + output_arr.size

        # Number of colors
        color_score = len(np.unique(input_arr)) + len(np.unique(output_arr))

        # Difference from input to output
        if input_arr.shape == output_arr.shape:
            diff_score = np.sum(input_arr != output_arr)
        else:
            diff_score = abs(input_arr.size - output_arr.size)

        return size_score + color_score * 10 + diff_score * 5

    def _hash_pattern(self, arr: np.ndarray) -> str:
        """Create hash for pattern matching."""
        # Simple hash based on shape and color distribution
        shape_str = f"{arr.shape}"
        colors = tuple(sorted(np.unique(arr).tolist()))
        return f"{shape_str}_{colors}"


class AdapterEnsemble:
    """
    Ensemble of adapted models for robust predictions.

    Combines multiple LoRA adapters trained with different
    hyperparameters or augmentations.
    """

    def __init__(self, num_adapters: int = 4):
        self.num_adapters = num_adapters
        self.adapters: List[LoRAAdapter] = []
        self.weights: List[float] = []

    def add_adapter(self, adapter: LoRAAdapter, weight: float = 1.0) -> None:
        """Add adapter to ensemble."""
        self.adapters.append(adapter)
        self.weights.append(weight)

    def predict_ensemble(
        self,
        predictions: List[np.ndarray]
    ) -> np.ndarray:
        """Combine predictions from ensemble members."""
        if not predictions:
            raise ValueError("No predictions provided")

        # Weighted voting for each position
        if len(predictions) == 1:
            return predictions[0]

        # Ensure all predictions have same shape
        shapes = [p.shape for p in predictions]
        if len(set(shapes)) > 1:
            # Return most common shape's prediction
            from collections import Counter
            shape_counts = Counter(shapes)
            most_common_shape = shape_counts.most_common(1)[0][0]
            predictions = [p for p in predictions if p.shape == most_common_shape]

        if not predictions:
            raise ValueError("No valid predictions after shape filtering")

        # Stack and take mode
        stacked = np.stack(predictions)
        from scipy import stats
        result, _ = stats.mode(stacked, axis=0)
        return result.squeeze()
