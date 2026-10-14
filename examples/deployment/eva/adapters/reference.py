"""Transport adaptation used only to compare upstream LIBERO ModelClient.step.

Image preprocessing and chunk selection remain upstream. The validation server
requires session fields and accepts algorithm options from the shared contract.
"""

from .. import msgpack_numpy


class ReferenceTransport:
    def __init__(self, stock_transport):
        self.transport = stock_transport
        self.metadata = stock_transport.get_server_metadata()
        if not hasattr(stock_transport, "_ws"):
            raise RuntimeError("LIBERO WebSocket client API changed; update adapters/reference.py")
        self.sequence = 0
        self.request_index = 0

    def request(self, kind, payload):
        request_id = f"reference-{self.request_index}"
        self.request_index += 1
        message = {"type": kind, "request_id": request_id, "payload": payload}
        if self.metadata.get("protocol", "").endswith("v2"):
            message["bridge"] = {
                "session_id": self.metadata["session_id"],
                "contract_sha256": self.metadata["contract_sha256"], "sequence": self.sequence,
            }
        self.transport._ws.send(msgpack_numpy.Packer().pack(message))
        response = msgpack_numpy.unpackb(self.transport._ws.recv(timeout=30))
        if response.get("status") != "ok" or response.get("request_id") != request_id:
            raise RuntimeError(f"Reference request failed: {response}")
        if kind == "reset":
            self.sequence = 0
        else:
            if self.metadata.get("protocol", "").endswith("v2"):
                data = response["data"]
                if data.get("sequence") != self.sequence or data.get("session_id") != self.metadata["session_id"]:
                    raise RuntimeError("Reference session/sequence mismatch")
            self.sequence += 1
        return response

    def predict_action(self, payload):
        # ModelClient supplies generic DDIM options unused by these flow heads.
        # Use the checkpoint-specific options advertised by the validation server.
        payload = {"examples": payload["examples"], "unnorm_key": payload["unnorm_key"],
                   **self.metadata["contract"].get("inference_kwargs", {})}
        return self.request("infer", payload)

    def close(self):
        self.transport.close()
