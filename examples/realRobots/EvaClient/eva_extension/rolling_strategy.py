"""Synchronous execution for policies that advance an internal rolling cache."""

import dataclasses
import logging
from typing import ClassVar

import numpy as np

from core.registry import STRATEGY_REGISTRY
from strategy.base_strategy import BaseInferStrategy

from examples.realRobots.EvaClient.bridge_contract import load_contract


@STRATEGY_REGISTRY.register("StarVlaRollingStrategy")
@dataclasses.dataclass
class StarVlaRollingStrategy(BaseInferStrategy):
    # Cropping a rolling chunk would advance the model cache past unexecuted
    # actions. Keep this setting in the shared contract instead of the live UI.
    tune_fields: ClassVar[list[dict]] = []
    contract_path: str = ""

    def __post_init__(self):
        self.contract = load_contract(self.contract_path)
        size = int(self.contract["action"]["chunk_size"])
        if self.execute_horizon is not None and self.execute_horizon != size:
            raise ValueError("Rolling execution horizon must equal the policy chunk size")
        self.execute_horizon = size
        super().__post_init__()

    def prepare_warmup_chunk(self, prompt, fetch_chunk):
        # EVA discards its prewarm predictions. Calling a stateful rolling policy
        # here would move its cache before any action was executed.
        logging.getLogger(__name__).info("Rolling policy: first inference occurs at execution start")
        return np.empty((0, int(self.contract["action"]["dim"])), dtype=np.float32)

    def _crop_chunk(self, action_chunk):
        expected = (int(self.contract["action"]["chunk_size"]),
                    int(self.contract["action"]["dim"]))
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if self.execute_horizon != expected[0] or chunk.shape != expected:
            raise ValueError(f"Rolling strategy requires a complete {expected} chunk")
        return super()._crop_chunk(chunk)
