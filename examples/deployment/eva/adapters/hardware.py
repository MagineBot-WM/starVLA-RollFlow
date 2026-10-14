"""Read hardware joint-order declarations without importing any hardware SDK."""

import ast
from pathlib import Path


def configure_action_subscriber(node, endpoint):
    """Execution node subscribes to EVA's bound command publisher."""
    import zmq
    node._action_sub.close(linger=0)
    subscriber = zmq.Context.instance().socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.setsockopt(zmq.RCVTIMEO, 0)
    subscriber.connect(endpoint)
    node._action_sub = subscriber


def inspect_hardware_joint_order(eva_root, contract):
    binding = contract.get("eva", {})
    relative = binding.get("hardware_node")
    expected = binding.get("hardware_joint_names")
    if relative is None:
        return None
    root = Path(eva_root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("eva.hardware_node must be inside the EVA checkout")
    tree = ast.parse(path.read_text())
    actual = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "JOINT_NAMES" for t in node.targets):
            actual = ast.literal_eval(node.value)
    if actual != expected:
        raise ValueError("Hardware JOINT_NAMES differs from the declared wire vector order; update adapters/hardware.py or the contract")
    return dict(path=str(path), joint_names=actual,
                eva_visualization_joint_names=binding.get("joint_names"),
                scope="static hardware order; no physical units/sign/driver verification")
