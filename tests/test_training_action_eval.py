from types import SimpleNamespace

import numpy as np
import pytest
from torch import nn

from starVLA.training.train_starvla import VLATrainer, _align_action_targets


def test_align_action_targets_uses_executed_prefix_of_training_window():
    actions = np.arange(40, dtype=np.float32).reshape(1, 40, 1)
    predicted = np.zeros((1, 8, 1), dtype=np.float32)

    targets = _align_action_targets(predicted, actions, action_horizon=32)

    np.testing.assert_array_equal(targets, actions[:, 8:16])


def test_align_action_targets_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="incompatible"):
        _align_action_targets(
            np.zeros((2, 8, 7)),
            np.zeros((1, 32, 7)),
            action_horizon=32,
        )


class _RollingEvalPolicy(nn.Module):
    action_horizon = 4

    def __init__(self):
        super().__init__()
        self.reset_calls = 0
        self.predict_was_training = None

    def reset(self):
        self.reset_calls += 1

    def predict_action(self, examples, **kwargs):
        del kwargs
        self.predict_was_training = self.training
        actions = np.asarray([example["action"] for example in examples])
        return {"normalized_actions": actions[:, -self.action_horizon :][:, :2]}


def test_eval_action_model_isolates_rolling_state_and_restores_train_mode():
    policy = _RollingEvalPolicy()
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.model = policy
    trainer.accelerator = SimpleNamespace(
        is_main_process=True,
        unwrap_model=lambda model: model,
    )
    trainer._get_next_batch = lambda: [
        {"action": np.arange(6, dtype=np.float32).reshape(6, 1)}
    ]

    metrics = trainer.eval_action_model({})

    assert metrics["mse_score"] == 0.0
    assert policy.predict_was_training is False
    assert policy.training is True
    assert policy.reset_calls == 2
