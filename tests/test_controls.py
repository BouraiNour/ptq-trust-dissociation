import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "per_item")


def run_controls(tmp_path, controls):
    args = [sys.executable, "-m", "ptq_trust.analysis",
            "--fp16", f"{DATA}/out_FP16.csv", "--nf4", f"{DATA}/out_NF4.csv", "--awq", f"{DATA}/out_AWQ_W4.csv",
            "--only_controls", "--n_boot", "300", "--out", str(tmp_path / "o.txt")]
    for name, path in controls:
        args += ["--control", f"{name}={path}"]
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def test_copy_of_fp16_has_zero_churn_and_zero_change(tmp_path):
    c = tmp_path / "c.csv"; shutil.copy(f"{DATA}/out_FP16.csv", c)
    r = run_controls(tmp_path, [("FP16_COPY", c)]); assert r.returncode == 0, r.stderr
    S = json.load(open(tmp_path / "controls_summary.json"))["FP16_COPY"]
    assert S["flip_pct"] == [0.0, 0.0, 0.0]
    assert S["acc_delta_pp"] == [0.0, 0.0, 0.0]
    assert S["faith_delta"] == [0.0, 0.0, 0.0] and S["mfull_delta"] == [0.0, 0.0, 0.0]
    # with a zero-churn control the excess equals the full flip rate of the quantized model (~23% / ~26%)
    assert 22 < S["excess_flip_pp"]["NF4"][0] < 25 and 24.5 < S["excess_flip_pp"]["AWQ_W4"][0] < 27.5


def test_copy_of_nf4_has_zero_excess_flips_against_nf4(tmp_path):
    c = tmp_path / "c.csv"; shutil.copy(f"{DATA}/out_NF4.csv", c)
    r = run_controls(tmp_path, [("NF4_COPY", c)]); assert r.returncode == 0, r.stderr
    S = json.load(open(tmp_path / "controls_summary.json"))["NF4_COPY"]
    assert S["excess_flip_pp"]["NF4"] == [0.0, 0.0, 0.0]
    assert S["faith_m_minus_control"]["NF4"] == [0.0, 0.0, 0.0]
    assert abs(S["acc_delta_pp"][0] - (-2.0)) < 1e-9          # NF4 strict accuracy change vs FP16


def test_incomplete_control_is_rejected(tmp_path):
    d = pd.read_csv(f"{DATA}/out_FP16.csv").iloc[:900]; c = tmp_path / "c.csv"; d.to_csv(c, index=False)
    r = run_controls(tmp_path, [("PARTIAL", c)])
    assert r.returncode != 0 and "missing" in (r.stderr + r.stdout)
