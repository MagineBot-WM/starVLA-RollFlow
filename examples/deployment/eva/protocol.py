"""Session/chunk validation layered onto the existing StarVLA message envelope."""

import uuid

from .bridge_contract import contract_sha256


def validate_handshake(metadata, contract):
    """Use the same strict handshake checks in diagnostics and the EVA client."""
    if not isinstance(metadata, dict):
        raise ValueError("Server metadata must be an object")
    if metadata.get("contract_sha256") != contract_sha256(contract):
        raise ValueError("StarVLA server/client contract hash mismatch")
    size = metadata.get("action_chunk_size")
    if isinstance(size, bool) or not isinstance(size, int) or size != contract["action"]["chunk_size"]:
        raise ValueError("Server action_chunk_size differs from the contract")
    available = metadata.get("available_unnorm_keys")
    if not isinstance(available, list) or contract["action"]["unnorm_key"] not in available:
        raise ValueError("Server lacks the contract's unnorm key")
    if contract["protocol"].endswith("v2"):
        session = metadata.get("session_id")
        if metadata.get("protocol") != contract["protocol"] or not isinstance(session, str) or not session:
            raise ValueError("V2 server must advertise its protocol and session_id")
    advertised = metadata.get("contract")
    if advertised is not None and (not isinstance(advertised, dict) or contract_sha256(advertised) != contract_sha256(contract)):
        raise ValueError("Advertised server contract differs from its hash")


class StreamSession:
    def __init__(self, contract):
        self.session_id = uuid.uuid4().hex
        self.contract_hash = contract_sha256(contract)
        self.v2 = contract["protocol"].endswith("v2")
        self.sequence = 0
        self.needs_reset = False

    def route(self, msg, fallback, reset):
        if not isinstance(msg, dict):
            return self._error("default", "Message must be an object")
        request_id = msg.get("request_id", "default")
        kind = msg.get("type", "infer")
        if self.v2:
            bridge = msg.get("bridge", {})
            if not isinstance(bridge, dict) or bridge.get("session_id") != self.session_id:
                return self._error(request_id, "Session mismatch")
            if bridge.get("contract_sha256") != self.contract_hash:
                return self._error(request_id, "Contract mismatch")
            if kind in {"infer", "predict_action"}:
                if self.needs_reset:
                    return self._error(request_id, "Previous inference failed; reset is required")
                sequence = bridge.get("sequence")
                if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != self.sequence:
                    return self._error(request_id, f"Expected chunk sequence {self.sequence}")
        if kind == "reset":
            try:
                payload = msg.get("payload", {})
                if not isinstance(payload, dict):
                    raise ValueError("Reset payload must be an object")
                reset(payload)
                self.sequence = 0
                self.needs_reset = False
                return {"status": "ok", "ok": True, "request_id": request_id,
                        "data": {"reset": True, "sequence": 0, "session_id": self.session_id}}
            except Exception as error:
                self.needs_reset = True
                return self._error(request_id, str(error))
        response = fallback(msg)
        if kind in {"infer", "predict_action"}:
            if response.get("status") == "ok":
                response["data"].update(sequence=self.sequence, session_id=self.session_id)
                self.sequence += 1
            else:
                self.needs_reset = True
        return response

    @staticmethod
    def _error(request_id, message):
        return {"status": "error", "ok": False, "request_id": request_id,
                "error": {"message": message}}
