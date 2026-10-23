import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_window_runs_demo_and_exports(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog

    from turning_movement_counter.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.run_demo()
    assert win.table.peak is not None
    assert abs(win.stats["counted"] - win.stats["true_vehicles"]) <= 0.02 * win.stats["true_vehicles"]
    assert win.counts.rowCount() == 8 + 2           # eight intervals, total, peak hour
    assert len(win.view.zones.zones) == 4
    assert not win.view.pixmap().isNull()

    out = tmp_path / "demo.tmc.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    win.export_csv()
    assert out.exists() and (tmp_path / "demo.tmc.long.csv").exists()

    win.interval.setCurrentIndex(1)                 # 5-minute bins
    win.run_demo()
    assert win.counts.rowCount() == 24 + 2
    app.processEvents()
