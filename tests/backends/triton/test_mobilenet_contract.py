"""Import-level checks for Person A's M1-M5 MobileNet handoff contract."""

from __future__ import annotations

from src.backends.triton.kernels import mobilenet_contract as contract


def test_mobilenet_contract_records_person_b_placeholders() -> None:
    assert contract.INPUT_SHAPE == (1, 3, 64, 64)
    assert contract.CLASSIFIER_WEIGHT_SHAPE == (43, 1280)
    assert contract.CLASSIFIER_BIAS_SHAPE == (43,)
    assert contract.NORMALIZATION_MEAN == (0.485, 0.456, 0.406)
    assert contract.NORMALIZATION_STD == (0.229, 0.224, 0.225)
    assert contract.PERSON_B_MUST_REPLACE in contract.REFERENCE_BATCH_PATH
    assert contract.MISSING_PERSON_B_HANDOFF


def test_missing_person_b_handoff_items_are_documented() -> None:
    missing_items = {item.item for item in contract.MISSING_PERSON_B_HANDOFF}

    assert "Measured GTSRB normalization mean/std" in missing_items
    assert "Exact target layer/block names or feature indices" in missing_items
    assert "CuPy and/or PyTorch reference outputs for selected layers" in missing_items
    assert "Final CuPy/Torch/DLPack handoff decision" in missing_items
