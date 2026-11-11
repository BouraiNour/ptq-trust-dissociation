import json
import os
import numpy as np
import pandas as pd
import pytest
from ptq_trust.parsing import LETTERS, parse_answer, split_sentences, margin

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "per_item")
AUDITED = ["FP16", "NF4", "AWQ_W4"]
RECIPES = AUDITED + ["BF16", "INT8"]          # BF16 / INT8 are controls


@pytest.fixture(scope="module", params=RECIPES)
def df(request):
    return pd.read_csv(os.path.join(DATA, f"out_{request.param}.csv")).fillna({"gen_letter": "", "gen_letter_loose": ""})


def test_complete_and_clean(df):
    assert len(df) == 1000 and sorted(df.idx) == list(range(1000))
    assert (df.n_options == 4).all() and (df.finished == 1).all()
    assert (df.gold == df.gold_idx.map(lambda i: LETTERS[i])).all()


def test_parse_columns_reproduced(df):
    for _, r in df.iterrows():
        primary, loose, method, rationale = parse_answer(r.gen_text)
        assert (primary or "") == r.gen_letter
        assert (loose or "") == r.gen_letter_loose
        assert method == r.parse_method
        assert len(split_sentences(rationale)) == r.K


def test_accuracy_columns_consistent(df):
    assert (df.correct_strict == (df.gen_letter == df.gold).astype(int)).all()
    assert (df.correct_argmax == (df.scored_letter == df.gold).astype(int)).all()
    lg = np.array([[float(v) for v in s.split(",")] for s in df.logits])
    assert (np.array([LETTERS[i] for i in lg.argmax(1)]) == df.scored_letter.values).all()


def test_margin_and_faithfulness_columns(df):
    lg = np.array([[float(v) for v in s.split(",")] for s in df.logits])
    t = lg.argmax(1)
    m = np.array([margin(lg[i], t[i]) for i in range(len(df))])
    assert np.abs(m - df.m_full.values).max() < 1e-2                    # logits stored to 4 decimals
    ok = df.faith_margin.notna()
    mean_drops = df.loc[ok, "drops_margin_json"].map(lambda s: np.mean(json.loads(s)))
    assert np.abs(mean_drops - df.loc[ok, "faith_margin"]).max() < 1e-3
    assert (df.loc[ok, "K"] >= 2).all()
    assert (df.loc[~ok, "K"] < 2).all() or (df.loc[~ok, "parse_method"] == "none").any()


def test_same_gold_across_recipes():
    g = [pd.read_csv(os.path.join(DATA, f"out_{r}.csv")).sort_values("idx").gold.values for r in RECIPES]
    assert all((g[0] == x).all() for x in g[1:])


def test_run_info_versions_agree():
    info = [json.load(open(os.path.join(DATA, f"run_info_{r}.json"))) for r in RECIPES]
    assert len({i["transformers"] for i in info}) == 1 and info[0]["transformers"] == "4.51.3"
    assert all(i["max_new"] == 400 and i["greedy"] for i in info)
    # NOTE: the INT8 control ran on a different torch build (2.11.0+cu130) than the others (2.11.0+cu128).
    assert {i["recipe"]: i["torch"] for i in info}["INT8"].endswith("cu130")
