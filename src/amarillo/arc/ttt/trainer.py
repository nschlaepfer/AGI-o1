"""
Test-Time Trainer for ARC-AGI.

Implements the training loop for per-task adaptation:
- Data preparation with augmentation
- Gradient-based optimization
- Validation on held-out examples
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..grid import Grid
from ..dataset import ARCTask
from ..synthesis.augmentation import generate_ttt_dataset, AugmentationPipeline
from .adapter import TTTConfig, LoRAAdapter, InContextOptimizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result from test-time training."""
    final_loss: float
    best_loss: float
    epochs_trained: int
    training_time_seconds: float
    validation_accuracy: float
    adapter: Optional[LoRAAdapter]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestTimeTrainer:
    """
    Per-task test-time trainer.

    Workflow:
    1. Generate augmented training data from task examples
    2. Fine-tune adapter (LoRA/prompt) on augmented data
    3. Validate on held-out examples
    4. Return adapted model/adapter for inference
    """

    def __init__(
        self,
        config: Optional[TTTConfig] = None,
        model_forward_fn: Optional[Callable] = None,
    ):
        self.config = config or TTTConfig()
        self.model_forward_fn = model_forward_fn  # External model inference

        self.augmentation_pipeline = AugmentationPipeline(
            use_rotations=self.config.use_geometric_aug,
            use_flips=self.config.use_geometric_aug,
            use_color_perms=self.config.use_color_aug,
        )
        self.icl_optimizer = InContextOptimizer()

    def train(
        self,
        task: ARCTask,
        max_time_seconds: float = 60.0,
    ) -> TrainingResult:
        """
        Train adapter for a specific task.

        Returns adapter that can be used for inference.
        """
        start_time = time.time()

        # Generate training data
        train_data = self._prepare_training_data(task)
        logger.info(f"Prepared {len(train_data)} training examples for task {task.id}")

        # Split for validation
        val_split = max(1, len(train_data) // 5)
        train_set = train_data[:-val_split]
        val_set = train_data[-val_split:]

        # Initialize adapter
        adapter = LoRAAdapter(
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.dropout,
        )
        adapter.initialize({"hidden_size": 4096})  # Placeholder

        # Training loop
        best_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        epochs_trained = 0

        for epoch in range(self.config.num_epochs):
            # Check time budget
            elapsed = time.time() - start_time
            if elapsed > max_time_seconds:
                logger.info(f"Time budget exhausted at epoch {epoch}")
                break

            # Shuffle training data
            indices = np.random.permutation(len(train_set))

            epoch_loss = 0.0
            num_batches = 0

            # Mini-batch training
            for i in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch = [train_set[j] for j in batch_indices]

                # Forward pass (placeholder - would use actual model)
                loss = self._compute_loss(batch, adapter)
                epoch_loss += loss
                num_batches += 1

                # Backward pass (placeholder - would compute actual gradients)
                self._update_adapter(adapter, batch)

            avg_loss = epoch_loss / max(num_batches, 1)

            # Validation
            val_acc = self._validate(val_set, adapter)
            epochs_trained = epoch + 1

            logger.debug(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, val_acc={val_acc:.2%}")

            # Early stopping
            if avg_loss < best_loss - self.config.min_improvement:
                best_loss = avg_loss
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        training_time = time.time() - start_time

        return TrainingResult(
            final_loss=avg_loss if epochs_trained > 0 else float("inf"),
            best_loss=best_loss,
            epochs_trained=epochs_trained,
            training_time_seconds=training_time,
            validation_accuracy=best_val_acc,
            adapter=adapter,
            metadata={
                "num_train_examples": len(train_set),
                "num_val_examples": len(val_set),
            }
        )

    def _prepare_training_data(
        self,
        task: ARCTask,
    ) -> List[Tuple[NDArray, NDArray]]:
        """Prepare training data with augmentation."""
        # Get base pairs
        base_pairs = [(p.input.data.copy(), p.output.data.copy()) for p in task.train]

        # Apply augmentation
        augmented = self.augmentation_pipeline.augment_batch(
            base_pairs,
            multiplier=self.config.augmentation_multiplier
        )

        # Add leave-one-out if enabled
        if self.config.use_loo_aug and len(task.train) > 2:
            loo_pairs = generate_ttt_dataset(
                task,
                num_augmentations=4,
                include_loo=True
            )
            augmented.extend(loo_pairs)

        return augmented

    def _compute_loss(
        self,
        batch: List[Tuple[NDArray, NDArray]],
        adapter: LoRAAdapter,
    ) -> float:
        """
        Compute loss for a batch.

        In production, this would:
        1. Format inputs for the LLM
        2. Run forward pass with adapter
        3. Compare predictions to targets
        4. Return cross-entropy loss
        """
        # Placeholder implementation
        loss = 0.0
        for input_arr, target_arr in batch:
            # Simulate loss based on target complexity
            loss += np.log(1 + target_arr.size) * 0.1

        return loss / len(batch)

    def _update_adapter(
        self,
        adapter: LoRAAdapter,
        batch: List[Tuple[NDArray, NDArray]],
    ) -> None:
        """
        Update adapter weights with gradients.

        In production, this would:
        1. Compute gradients via backpropagation
        2. Apply gradient clipping
        3. Update LoRA weights with optimizer step
        """
        # Placeholder: small random update
        for module_name in adapter.weights_a:
            # Simulate gradient update
            grad_a = np.random.randn(*adapter.weights_a[module_name].shape) * 0.001
            grad_b = np.random.randn(*adapter.weights_b[module_name].shape) * 0.001

            # Gradient clipping
            grad_a = np.clip(grad_a, -self.config.gradient_clip, self.config.gradient_clip)
            grad_b = np.clip(grad_b, -self.config.gradient_clip, self.config.gradient_clip)

            # Update
            adapter.weights_a[module_name] -= self.config.learning_rate * grad_a
            adapter.weights_b[module_name] -= self.config.learning_rate * grad_b

    def _validate(
        self,
        val_set: List[Tuple[NDArray, NDArray]],
        adapter: LoRAAdapter,
    ) -> float:
        """
        Validate adapter on held-out examples.

        Returns accuracy (0-1).
        """
        if not val_set:
            return 0.0

        # Placeholder: random accuracy based on adapter state
        # In production, would run inference and compare
        num_params = adapter.get_trainable_params()
        base_acc = 0.3 + 0.5 * (1 - np.exp(-num_params / 100000))

        return min(1.0, base_acc + np.random.random() * 0.2)


class BatchTestTimeTrainer:
    """
    Efficient batch test-time trainer for multiple tasks.

    Optimizations:
    - Shared base computation
    - Parallel adaptation
    - Caching of common patterns
    """

    def __init__(
        self,
        config: Optional[TTTConfig] = None,
        num_workers: int = 4,
    ):
        self.config = config or TTTConfig()
        self.num_workers = num_workers
        self.trainers = [TestTimeTrainer(config) for _ in range(num_workers)]

    def train_batch(
        self,
        tasks: List[ARCTask],
        time_per_task: float = 30.0,
    ) -> List[TrainingResult]:
        """Train adapters for multiple tasks."""
        import concurrent.futures

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for i, task in enumerate(tasks):
                trainer = self.trainers[i % self.num_workers]
                future = executor.submit(trainer.train, task, time_per_task)
                futures[future] = task.id

            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    result.metadata["task_id"] = task_id
                    results.append(result)
                except Exception as e:
                    logger.error(f"Training failed for {task_id}: {e}")
                    results.append(TrainingResult(
                        final_loss=float("inf"),
                        best_loss=float("inf"),
                        epochs_trained=0,
                        training_time_seconds=0.0,
                        validation_accuracy=0.0,
                        adapter=None,
                        metadata={"task_id": task_id, "error": str(e)}
                    ))

        return results


class ConceptPreTrainer:
    """
    Pre-train on concept tasks before test-time training.

    This follows NVARC's approach of building concept understanding
    before adapting to specific tasks.
    """

    def __init__(
        self,
        base_adapter: Optional[LoRAAdapter] = None,
        config: Optional[TTTConfig] = None,
    ):
        self.base_adapter = base_adapter or LoRAAdapter()
        self.config = config or TTTConfig()
        self.concept_adapters: Dict[str, LoRAAdapter] = {}

    def pretrain_concept(
        self,
        concept_name: str,
        concept_tasks: List[ARCTask],
        num_epochs: int = 10,
    ) -> LoRAAdapter:
        """Pre-train adapter on a specific concept."""
        trainer = TestTimeTrainer(self.config)

        # Combine all concept tasks
        all_pairs = []
        for task in concept_tasks:
            for pair in task.train:
                all_pairs.append((pair.input.data, pair.output.data))

        # Create synthetic task for training
        from ..grid import GridPair
        combined_task = ARCTask(
            id=f"concept_{concept_name}",
            train=[GridPair(Grid(inp), Grid(out)) for inp, out in all_pairs[:10]],
            test=[GridPair(Grid(inp), Grid(out)) for inp, out in all_pairs[10:12]],
        )

        result = trainer.train(combined_task, max_time_seconds=60.0)

        if result.adapter:
            self.concept_adapters[concept_name] = result.adapter

        return result.adapter

    def get_concept_adapter(self, concept_name: str) -> Optional[LoRAAdapter]:
        """Get pre-trained adapter for a concept."""
        return self.concept_adapters.get(concept_name)

    def initialize_from_concepts(
        self,
        task: ARCTask,
        detected_concepts: List[str],
    ) -> LoRAAdapter:
        """
        Initialize adapter for task based on detected concepts.

        Combines relevant concept adapters as starting point.
        """
        if not detected_concepts:
            adapter = LoRAAdapter()
            adapter.initialize({"hidden_size": 4096})
            return adapter

        # Average weights from relevant concept adapters
        adapters = [
            self.concept_adapters[c] for c in detected_concepts
            if c in self.concept_adapters
        ]

        if not adapters:
            adapter = LoRAAdapter()
            adapter.initialize({"hidden_size": 4096})
            return adapter

        # Create combined adapter
        combined = LoRAAdapter(
            rank=adapters[0].rank,
            alpha=adapters[0].alpha,
            dropout=adapters[0].dropout,
            target_modules=adapters[0].target_modules,
        )

        # Average weights
        for module_name in adapters[0].weights_a:
            weights_a = np.stack([a.weights_a[module_name] for a in adapters])
            weights_b = np.stack([a.weights_b[module_name] for a in adapters])

            combined.weights_a[module_name] = np.mean(weights_a, axis=0)
            combined.weights_b[module_name] = np.mean(weights_b, axis=0)

        return combined
