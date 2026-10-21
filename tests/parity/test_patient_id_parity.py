"""Parity test for patient ID and side parsing between golden reference and drdg."""

import pytest
from drdg.data.patient_ids import parse_patient_and_side as drdg_parse


def golden_parse_patient_and_side(image_id: str, dataset_key: str):
    """Exact extracted reference implementation from merge_dataset_lodo.ipynb Cell 4."""
    img_str = str(image_id).lower()
    k = dataset_key.lower()

    if "idrid" in k:
        base = img_str.replace(".jpg", "").replace(".png", "")
        pid = base.split("_")[-1]
    elif "eyepacs" in k or "deepdrid" in k:
        pid = img_str.split("_")[0]
    elif "ddr" in k:
        parts = img_str.replace(".jpg", "").replace(".png", "").split("-")
        pid = "-".join(parts[:2]) if len(parts) >= 2 else img_str.split(".")[0]
    elif "messidor" in k:
        pid = img_str.split("_")[0]
    else:
        pid = img_str

    side = "left" if "left" in img_str or "_l" in img_str else \
           "right" if "right" in img_str or "_r" in img_str else "unknown"
    return f"{dataset_key}::{pid}", side


@pytest.mark.parametrize(
    "image_id,dataset_key",
    [
        ("001_left.jpeg", "eyepacs"),
        ("001_right.jpeg", "eyepacs"),
        ("104_l1.jpg", "deepdrid"),
        ("104_r2.jpg", "deepdrid"),
        ("IDRiD_042.jpg", "idrid"),
        ("train_IDRiD_128.jpg", "idrid"),
        ("007-0004-000.jpg", "ddr"),
        ("007-0004-001.jpg", "ddr"),
        ("20051020_43808_0100_PP.tif", "messidor2"),
        ("20051020_43808_0200_PP.tif", "messidor2"),
        ("c0022359", "aptos"),  # canonical id_code format in aptos_total.csv
    ],
)
def test_patient_id_exact_parity(image_id: str, dataset_key: str):
    gold_pid, gold_side = golden_parse_patient_and_side(image_id, dataset_key)
    cand_pid, cand_side = drdg_parse(image_id, dataset_key)
    assert cand_pid == gold_pid, f"PID mismatch for {dataset_key} {image_id}: {cand_pid} vs {gold_pid}"
    assert cand_side == gold_side, f"Side mismatch for {dataset_key} {image_id}: {cand_side} vs {gold_side}"


def test_aptos_extension_stripping_discrepancy():
    """Documenting audited behavior discrepancy:
    Golden reference does not strip .png if passed directly in image_id (pid = 'c0022359.png'),
    whereas drdg normalizes extensions prior to parsing (pid = 'c0022359').
    """
    raw_img = "c0022359.png"
    gold_pid, _ = golden_parse_patient_and_side(raw_img, "aptos")
    cand_pid, _ = drdg_parse(raw_img, "aptos")
    assert gold_pid == "aptos::c0022359.png"
    assert cand_pid == "aptos::c0022359"
