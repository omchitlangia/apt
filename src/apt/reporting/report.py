"""Report skeleton writer.

:func:`write_report_skeleton` emits a Markdown file pre-populated with the
section list from ``docs/reporting_standard.md``. Section bodies are
left as ``[TODO]`` placeholders; the calling unit fills them in.
"""

from __future__ import annotations

from pathlib import Path

SKELETON_SECTIONS: list[tuple[str, str]] = [
    (
        "Objective",
        "[TODO] one-sentence statement of what this unit set out to establish or refute.",
    ),
    (
        "Pre-registered expectations",
        "[TODO] hypotheses written BEFORE running anything (cite the prior unit "
        "where these were declared if applicable).",
    ),
    (
        "Config grid",
        "[TODO] table of every parameter actually varied in this run. One row "
        "per (engine, freq_min, regime, spread_bps, stop_mode) cell, plus any "
        "additional knobs (z-thresholds, windows, …).",
    ),
    (
        "Headline tables (gross AND net side by side)",
        "[TODO] one row per cell, gross AND net for every metric. NEVER report "
        "net without the matching gross column.",
    ),
    (
        "Figures",
        "[TODO] required figures per reporting_standard.md — at minimum (a), "
        "(b), (d), (h). Add (c), (e), (f), (g), (i), (j), (k) when applicable. "
        "Paths are relative to the report file.",
    ),
    (
        "Exclusion funnel",
        "[TODO] pair-folds attempted → liquidity → AR(1) valid → HL-band → "
        "traded. Visible to the reader.",
    ),
    (
        "Diagnostics",
        "[TODO] HL distribution, drift chart, β distribution, etc. Anything "
        "needed to justify or refute the headline claim.",
    ),
    (
        "Caveats / [TODO]",
        "[TODO] every approximation, omission, scope limit. Use a [TODO] tag "
        "instead of inventing a number.",
    ),
    (
        "Verdict",
        "[TODO] primary finding stated in order of importance. The most-"
        "interesting headline goes LAST when it is exploratory — see "
        "reporting_standard.md §4.",
    ),
    ("Test results", "[TODO] paste the literal pytest terminal summary. Reconcile counts."),
]


def write_report_skeleton(
    out_path: Path,
    *,
    title: str,
    unit_name: str,
    branch: str | None = None,
    extra_top_matter: str = "",
) -> Path:
    """Write a Markdown skeleton to ``out_path``. Returns the path written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Unit:** `{unit_name}`")
    if branch is not None:
        lines.append(f"**Branch:** `{branch}`")
    lines.append("")
    lines.append(
        "This report follows the contract in "
        "[`docs/reporting_standard.md`](reporting_standard.md). "
        "Section list and the gross+net rule are fixed there."
    )
    lines.append("")
    if extra_top_matter:
        lines.append(extra_top_matter.rstrip())
        lines.append("")
    for i, (sec, body) in enumerate(SKELETON_SECTIONS, start=1):
        lines.append(f"## {i}. {sec}")
        lines.append("")
        lines.append(body)
        lines.append("")
    out_path.write_text("\n".join(lines))
    return out_path


__all__ = ["SKELETON_SECTIONS", "write_report_skeleton"]
