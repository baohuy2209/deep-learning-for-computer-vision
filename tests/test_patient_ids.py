"""Unit tests for patient ID parsing and namespacing across all six cohorts."""

import pytest
from drdg.data.patient_ids import parse_patient_and_side


def test_aptos_patient_parsing():
    pid, side = parse_patient_and_side("000c1449c6d9.png", "aptos")
    assert pid == "aptos::000c1449c6d9"
    assert side == "unknown"


def test_ddr_patient_parsing():
    pid, side = parse_patient_and_side("007-0001-000.jpg", "ddr")
    assert pid == "ddr::007-0001"
    assert side == "unknown"


def test_deepdrid_patient_parsing():
    pid_left, side_left = parse_patient_and_side("042_l1.jpg", "deepdrid")
    assert pid_left == "deepdrid::042"
    assert side_left == "left"

    pid_right, side_right = parse_patient_and_side("042_r2.jpg", "deepdrid")
    assert pid_right == "deepdrid::042"
    assert side_right == "right"
    # Both eyes belong to the exact same patient
    assert pid_left == pid_right


def test_idrid_patient_parsing():
    pid, side = parse_patient_and_side("IDRiD_056.jpg", "idrid")
    assert pid == "idrid::056"
    assert side == "unknown"


def test_messidor2_patient_parsing():
    pid, side = parse_patient_and_side("20051019_38557_0100_PP.tif", "messidor2")
    assert pid == "messidor2::20051019"


def test_eyepacs_patient_parsing():
    pid_left, side_left = parse_patient_and_side("10_left.jpeg", "eyepacs")
    assert pid_left == "eyepacs::10"
    assert side_left == "left"

    pid_right, side_right = parse_patient_and_side("10_right.jpeg", "eyepacs")
    assert pid_right == "eyepacs::10"
    assert side_right == "right"
    # Both eyes map to the same patient
    assert pid_left == pid_right


def test_cross_cohort_namespace_collision_prevention():
    # Patient 10 in EyePACS vs Patient 10 in DeepDRiD vs Patient 10 in IDRiD
    pid_eye, _ = parse_patient_and_side("10_left.jpeg", "eyepacs")
    pid_deep, _ = parse_patient_and_side("10_l1.jpg", "deepdrid")
    pid_idrid, _ = parse_patient_and_side("IDRiD_10.jpg", "idrid")

    assert pid_eye != pid_deep
    assert pid_deep != pid_idrid
    assert pid_eye != pid_idrid
