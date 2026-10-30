import json

import pytest

from segmentation.infer import verify_model
from segmentation.io import sha256


def test_smoke_and_changed_checkpoints_cannot_enter_final_evaluation(tmp_path):
    metadata = {"status": "smoke_only", "smoke_steps": 6}
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="full training"):
        verify_model(tmp_path)
    (tmp_path / "weights").mkdir()
    weights = tmp_path / "weights/best.pt"
    weights.write_bytes(b"original checkpoint test fixture")
    metadata.update(status="complete", smoke_steps=0, artifact_sha256={}, checkpoint_sha256=sha256(weights))
    path.write_text(json.dumps(metadata))
    weights.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint changed"):
        verify_model(tmp_path)
