"""Stub control-plane client for gateway path.

Prevents ChatBedrockConverse from creating a real AWS bedrock client
when we only need the runtime (converse) surface.
"""


class StubBedrockControlClient:
    """Minimal stub satisfying ChatBedrockConverse's bedrock_client interface."""

    class meta:
        region_name = "us-east-1"

    def get_inference_profile(self, **kwargs):
        raise NotImplementedError(
            "Application inference profiles not supported via gateway"
        )
