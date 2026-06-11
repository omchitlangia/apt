"""Smoke tests for ``apt.reporting.figures``.

Each test asserts that the figure function (a) runs without raising on a
small synthetic fixture and (b) writes BOTH the PNG and the companion
CSV. No image-content assertions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apt.reporting import figures as figs

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_pair_sessions() -> pd.DataFrame:
    """Two pair-folds, 30 sessions each."""
    rng = np.random.default_rng(0)
    rows = []
    for fold_id, pair in [(1, "A/B"), (2, "C/D")]:
        for i in range(30):
            g = rng.normal(0.001, 0.01)
            n = g - 0.0005
            rows.append(
                {
                    "date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=i),
                    "fold_id": fold_id,
                    "pair": pair,
                    "gross_log_ret": g,
                    "net_log_ret": n,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def small_trades() -> pd.DataFrame:
    base = pd.Timestamp("2021-01-15 09:15:00")
    rows = []
    for i, reason in enumerate(figs.EXIT_REASON_VOCAB):
        rows.append(
            {
                "engine": "ou",
                "freq_min": 5,
                "regime": "B",
                "spread_bps": 3,
                "stop_mode": "none",
                "fold_id": 1,
                "pair": "A/B",
                "direction": 1 if i % 2 == 0 else -1,
                "entry_ts": (base + pd.Timedelta(minutes=10 * i)).isoformat(),
                "exit_ts": (base + pd.Timedelta(minutes=10 * i + 30)).isoformat(),
                "entry_z": -0.5 + 0.1 * i,
                "exit_z": 0.1 + 0.1 * i,
                "exit_reason": reason,
                "gross_log_pnl": 0.001 * (i + 1),
                "cost_log": 0.0015,
                "net_log_pnl": 0.001 * (i + 1) - 0.0015,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def small_metrics() -> pd.DataFrame:
    rows = []
    for engine in ("ou", "rolling_z"):
        for freq in (5, 15):
            for cost in (1, 3, 5, 8):
                rows.append(
                    {
                        "engine": engine,
                        "freq_min": freq,
                        "regime": "B",
                        "spread_bps": cost,
                        "stop_mode": "none",
                        "n_pairs": 2,
                        "n_trades": 30 + cost,
                        "net_sharpe": 1.0 - 0.1 * cost,
                        "net_total_pct": 50.0 - 5 * cost,
                        "gross_sharpe": 1.2,
                        "gross_total_pct": 60.0,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _assert_outputs(png: Path, csv: Path) -> None:
    assert png.exists() and png.stat().st_size > 0
    assert csv.exists() and csv.stat().st_size > 0


def test_a_per_pair_fold_equity(tmp_path: Path, small_pair_sessions: pd.DataFrame) -> None:
    png, csv = figs.fig_a_per_pair_fold_equity(
        small_pair_sessions,
        out_dir=tmp_path,
        name="a_test",
        engine="ou",
        freq_min=5,
        regime="B",
        spread_bps=3,
        stop_mode="none",
    )
    _assert_outputs(png, csv)


def test_b_portfolio_nav(tmp_path: Path, small_pair_sessions: pd.DataFrame) -> None:
    png, csv = figs.fig_b_portfolio_nav(
        small_pair_sessions,
        out_dir=tmp_path,
        name="b_test",
        engine="ou",
        freq_min=5,
        regime="B",
        spread_bps=3,
        stop_mode="none",
    )
    _assert_outputs(png, csv)


def test_c_spread_z_markers(tmp_path: Path, small_trades: pd.DataFrame) -> None:
    n = 200
    base = pd.Timestamp("2021-01-15 09:15:00")
    sz = pd.DataFrame(
        {
            "ts": [base + pd.Timedelta(minutes=i) for i in range(n)],
            "spread": np.linspace(0, 0.1, n),
            "z": np.linspace(-1, 1, n),
        }
    )
    png, csv = figs.fig_c_spread_z_markers(
        sz,
        small_trades,
        out_dir=tmp_path,
        name="c_test",
        engine="ou",
        freq_min=5,
        regime="B",
        spread_bps=3,
        stop_mode="none",
        pair_fold_label="fold1·A/B (synthetic)",
    )
    _assert_outputs(png, csv)


def test_d_cost_ladder(tmp_path: Path, small_metrics: pd.DataFrame) -> None:
    png, csv = figs.fig_d_cost_ladder(
        small_metrics,
        out_dir=tmp_path,
        name="d_test",
        freq_min=5,
        regime="B",
    )
    _assert_outputs(png, csv)


def test_e_a_star_vs_cost(tmp_path: Path) -> None:
    a_star = pd.DataFrame(
        {
            "pair": ["A/B"] * 4 + ["C/D"] * 4,
            "freq_min": [5] * 8,
            "spread_bps": [1, 3, 5, 8] * 2,
            "a_star_z": [0.4, 0.45, 0.5, 0.55, 0.46, 0.51, 0.55, 0.61],
            "a_star_bps": [160, 180, 200, 220, 200, 230, 250, 280],
        }
    )
    png, csv = figs.fig_e_a_star_vs_cost(a_star, out_dir=tmp_path, name="e_test")
    _assert_outputs(png, csv)


def test_f_half_life_distribution(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    hl = pd.DataFrame(
        {
            "freq_min": [1] * 30 + [5] * 30,
            "half_life_min": np.exp(rng.normal(np.log(500), 1.0, 60)),
        }
    )
    png, csv = figs.fig_f_half_life_distribution(hl, out_dir=tmp_path, name="f_test")
    _assert_outputs(png, csv)


def test_g_drift_chart(tmp_path: Path) -> None:
    drift = pd.DataFrame(
        {
            "fold_id": [1, 1, 2, 2],
            "pair": ["A/B", "C/D", "A/B", "E/F"],
            "freq_min": [5, 5, 15, 15],
            "regime": ["B", "B", "A", "A"],
            "drift_mean_sigma_eq": [-3.0, 1.2, -0.4, 0.6],
            "traded": [True, True, False, False],
        }
    )
    png, csv = figs.fig_g_drift_chart(drift, out_dir=tmp_path, name="g_test")
    _assert_outputs(png, csv)


def test_h_exit_reason_stacked(tmp_path: Path, small_trades: pd.DataFrame) -> None:
    png, csv = figs.fig_h_exit_reason_stacked(
        small_trades,
        out_dir=tmp_path,
        name="h_test",
        group_by=("engine", "freq_min", "regime"),
    )
    _assert_outputs(png, csv)


def test_h_unknown_exit_reason_folded(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "engine": ["x"] * 3,
            "freq_min": [5] * 3,
            "regime": ["B"] * 3,
            "exit_reason": ["mean_revert", "z_stop", "made_up_reason"],
        }
    )
    png, csv = figs.fig_h_exit_reason_stacked(
        df,
        out_dir=tmp_path,
        name="h_unknown",
        group_by=("engine", "freq_min", "regime"),
    )
    _assert_outputs(png, csv)
    pivot = pd.read_csv(csv)
    assert "__OTHER__" in pivot.columns
    # Unknown 'made_up_reason' is bucketed into __OTHER__
    assert int(pivot["__OTHER__"].sum()) == 1


def test_i_trade_counts(tmp_path: Path, small_metrics: pd.DataFrame) -> None:
    png, csv = figs.fig_i_trade_counts(
        small_metrics,
        out_dir=tmp_path,
        name="i_test",
        regime="B",
        stop_mode="none",
        spread_bps=3,
    )
    _assert_outputs(png, csv)


def test_j_beta_histogram(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    betas = pd.DataFrame(
        {
            "pair": [f"P{i}" for i in range(14)],
            "one_plus_beta_over_2": rng.uniform(0.5, 1.4, 14),
            "traded": [False] * 12 + [True, True],
        }
    )
    png, csv = figs.fig_j_beta_histogram(betas, out_dir=tmp_path, name="j_test")
    _assert_outputs(png, csv)


def test_k_exclusion_funnel(tmp_path: Path) -> None:
    funnel = pd.DataFrame(
        {
            "freq_min": [5, 5, 5, 5, 15, 15, 15, 15],
            "regime": ["B"] * 8,
            "stage": ["attempted", "ar1_valid", "hl_band", "traded"] * 2,
            "n": [19, 19, 2, 2, 19, 19, 2, 2],
        }
    )
    png, csv = figs.fig_k_exclusion_funnel(funnel, out_dir=tmp_path, name="k_test")
    _assert_outputs(png, csv)


def test_all_figure_names_complete() -> None:
    # All 11 letters present in FIGURE_LETTERS
    assert set(figs.FIGURE_LETTERS.keys()) == {
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
    }


def test_report_skeleton_writes(tmp_path: Path) -> None:
    from apt.reporting.report import SKELETON_SECTIONS, write_report_skeleton

    out = tmp_path / "skeleton.md"
    path = write_report_skeleton(
        out,
        title="Test report",
        unit_name="unit-test",
        branch="feature/test",
    )
    assert path.exists()
    text = path.read_text()
    for sec_name, _ in SKELETON_SECTIONS:
        assert sec_name in text
