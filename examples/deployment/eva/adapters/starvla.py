"""Version-dependent checkpoint access stays on the GPU/server side."""

from __future__ import annotations

import inspect
from pathlib import Path


class CheckpointAdapter:
    def __init__(self, wrapper, unnorm_key):
        self.wrapper = wrapper
        for name, arguments in {"predict_action": {"examples", "unnorm_key"},
                                "get_norm_processor": {"unnorm_key"}, "reset": set()}.items():
            fn = getattr(wrapper, name, None)
            if not callable(fn) or not arguments <= set(inspect.signature(fn).parameters):
                raise RuntimeError(f"StarVLA wrapper API changed: {name}; update adapters/starvla.py")
        self.config = getattr(wrapper, "_model_cfg", None)
        if not isinstance(self.config, dict):
            raise RuntimeError("StarVLA checkpoint-config API changed; update adapters/starvla.py")
        self.processor = wrapper.get_norm_processor(unnorm_key)
        data_config = getattr(self.processor, "_data_config", None)
        if data_config is None:
            raise RuntimeError("StarVLA training-transform API changed; update adapters/starvla.py")
        self.camera_keys = list(getattr(data_config, "video_keys", []))
        framework = getattr(wrapper, "_framework", None)
        if framework is None:
            raise RuntimeError("StarVLA framework API changed; update adapters/starvla.py")
        self.is_rolling = getattr(getattr(framework, "action_model", None), "execution_horizon", None) is not None
        objects = (wrapper, framework, framework.action_model, data_config)
        self.source_paths = sorted({Path(inspect.getfile(value if inspect.isclass(value) else type(value)))
                                    for value in objects})

    def normalize_state(self, state_fields):
        import numpy as np
        import torch
        normalized = self.processor.transform.apply(state_fields)
        chunks = []
        for key in self.processor.state_keys:
            value = normalized[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            chunks.append(np.asarray(value, dtype=np.float32))
        return np.concatenate(chunks, axis=-1)
