import json

from data.protocol import identity


def test_dataset_identity_preserves_semantics_across_newlines_and_json_format(tmp_path):
    source = tmp_path / "source_hashes.json"
    recipe = tmp_path / "preprocessing.json"
    split = tmp_path / "splits.csv"
    source.write_text('{"image": "abcd"}\n')
    recipe.write_text('{"shape": [256, 256]}\n')
    split.write_text('patient_id,split\np1,train\np2,test\n')
    original = identity(tmp_path)
    source.write_bytes(b'{\r\n  "image": "abcd"\r\n}\r\n')
    split.write_bytes(b'patient_id,split\r\np2,test\r\np1,train\r\n')
    assert identity(tmp_path) == original
    split.write_text('patient_id,split\np1,train\np2,train\n')
    assert identity(tmp_path) != original
    split.write_text('patient_id,split\np1,train\np2,test\n')
    recipe.write_text(json.dumps({"shape": [128, 128]}))
    assert identity(tmp_path) != original
