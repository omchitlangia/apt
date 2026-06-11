"""Reusable reporting + figures module.

Public surface:

* :mod:`apt.reporting.figures` — one function per standard figure.
  Each function takes typed inputs (DataFrames + an output directory),
  writes a deterministically-named PNG plus a companion CSV holding
  the exact data behind the figure, and returns the two paths as a
  tuple. All figure titles carry full cell identity
  (engine / freq / regime / cost / stop_mode) when applicable.
* :mod:`apt.reporting.report` — writes a Markdown report skeleton
  pre-populated with the report-standard section list, ready to be
  filled with text + tables + figures by the calling unit.

See ``docs/reporting_standard.md`` for the standing contract every
future unit must comply with. The figures here implement the eleven
required figure types (a-k) listed in that doc.
"""

from apt.reporting import figures, report

__all__ = ["figures", "report"]
