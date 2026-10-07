from guessthecondition import analyze
from guessthecondition.report import write_report

from test_analysis import balanced


def test_report_writes_pdf_figures_and_tables(tmp_path):
    table = balanced([0.9, 0.8, 0.7])
    paths = write_report(analyze(table), table, tmp_path / "out")
    for name in ("analysis_results.pdf", "confusion_matrix.png", "condition_accuracy.png", "decision_time.png", "decision_time.csv", "summary.txt", "per_repeat.csv", "per_condition.csv", "confusion_counts.csv"):
        assert paths[name].is_file() and paths[name].stat().st_size > 0, name
    assert paths["analysis_results.pdf"].read_bytes().startswith(b"%PDF")
    assert "Distinguishable across repeats" in paths["summary.txt"].read_text(encoding="utf-8")


def test_report_without_decision_times(tmp_path):
    table = balanced([0.9, 0.8, 0.7]).drop(columns="decision_time_s")
    paths = write_report(analyze(table), table, tmp_path / "out")
    assert "decision_time.png" not in paths and paths["analysis_results.pdf"].is_file()
