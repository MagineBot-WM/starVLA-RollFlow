"""Apply checkpoint image interpolation before EVA's existing observation builder.

These subclasses preserve the stock dataset/ZMQ behavior. EVA subsequently
resizes to the same dimensions, so it doesn't resample the prepared pixels.
"""

from dataclasses import replace

import numpy as np
from PIL import Image

from core.registry import TRANSPORT_REGISTRY
from transport.dataset import DatasetTransport
from transport.zmq import ZmqTransport

from examples.realRobots.EvaClient.bridge_contract import load_contract


class ContractImageMixin:
    def get_frame(self):
        frame = super().get_frame()
        if frame is None:
            return None
        contract = load_contract(self._config.policy.backend_options.contract_path)
        height, width = contract["image"]["hw"]
        if (height, width) != (self._config.transport.image_height,
                              self._config.transport.image_width):
            raise ValueError("EVA transport resolution must match the image contract")
        interpolation = contract["image"].get("interpolation", "bicubic")
        resample = {"bicubic": Image.Resampling.BICUBIC,
                    "bilinear": Image.Resampling.BILINEAR}[interpolation]
        images = {
            key: np.asarray(Image.fromarray(np.asarray(value)).resize(
                (width, height), resample=resample)).copy()
            for key, value in frame.images.items()
        }
        return replace(frame, images=images)


class ContractDatasetTransport(ContractImageMixin, DatasetTransport):
    pass


class ContractZmqTransport(ContractImageMixin, ZmqTransport):
    pass


@TRANSPORT_REGISTRY.register("starvla_dataset")
def build_contract_dataset(config, robot):
    return ContractDatasetTransport(config, robot, config.transport.dataset_dir,
                                    config.transport.episode_id)


@TRANSPORT_REGISTRY.register("starvla_zmq")
def build_contract_zmq(config, robot):
    return ContractZmqTransport(config, robot)
