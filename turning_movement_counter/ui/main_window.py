"""PySide6 main window."""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime

from PySide6.QtCore import QDateTime, QPointF, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDateTimeEdit, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)

from .. import __version__
from ..counting import CountSettings, Movement, TrackAccumulator
from ..export import FROM_LEG, write_long_csv, write_tmc_csv
from ..intervals import CountTable, build_table
from ..memo import api_key, save_api_key, summary_dict, write_memo
from ..zones import APPROACHES, BOUND_NAME, DIRECTIONS, TURN_NAME, TURNS, Zone, ZoneSet

LEG_COLORS = {"N": QColor(0, 200, 255), "E": QColor(255, 170, 0), "S": QColor(120, 230, 90), "W": QColor(240, 90, 200)}
LEG_NAMES = {"N": "North leg", "E": "East leg", "S": "South leg", "W": "West leg"}


def _qimage(rgb) -> QImage:
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class FrameView(QLabel):
    """Shows a video frame with the approach zones; click to add points to the zone being drawn."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background:#202124; color:#bbb;")
        self.setText("Open an intersection video (File > Open video)\nor try Help > Run demo count")
        self.image: QImage | None = None
        self.zones = ZoneSet()
        self.drawing: str | None = None     # leg being drawn
        self.points: list[tuple[float, float]] = []

    def set_frame(self, rgb) -> None:
        self.image = _qimage(rgb)
        self.zones.frame_size = (self.image.width(), self.image.height())
        self.render()

    def _geometry(self):
        if self.image is None:
            return None
        iw, ih = self.image.width(), self.image.height()
        s = min(self.width() / iw, self.height() / ih)
        return s, (self.width() - iw * s) / 2, (self.height() - ih * s) / 2

    def render(self) -> None:
        if self.image is None:
            return
        pix = QPixmap.fromImage(self.image)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont()
        font.setPointSize(max(10, self.image.height() // 30))
        font.setBold(True)
        p.setFont(font)
        for z in self.zones.zones:
            col = LEG_COLORS[z.approach]
            fill = QColor(col)
            fill.setAlpha(60)
            p.setPen(QPen(col, 3))
            p.setBrush(fill)
            p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in z.points]))
            cx = sum(x for x, _ in z.points) / len(z.points)
            cy = sum(y for _, y in z.points) / len(z.points)
            p.setPen(QPen(QColor("white"), 1))
            p.drawText(QPointF(cx - 10, cy + 10), z.approach)
        if self.drawing:
            col = LEG_COLORS[self.drawing]
            p.setPen(QPen(col, 3, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            pts = [QPointF(x, y) for x, y in self.points]
            if len(pts) > 1:
                p.drawPolyline(QPolygonF(pts))
            p.setBrush(col)
            for q in pts:
                p.drawEllipse(q, 5, 5)
        p.end()
        self.setPixmap(pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self.render()

    def mousePressEvent(self, e):  # noqa: N802
        g = self._geometry()
        if g is None or not self.drawing:
            return
        if e.button() == Qt.MouseButton.RightButton:
            self.finish()
            return
        s, ox, oy = g
        x, y = (e.position().x() - ox) / s, (e.position().y() - oy) / s
        if 0 <= x <= self.image.width() and 0 <= y <= self.image.height():
            self.points.append((round(x, 1), round(y, 1)))
            self.render()

    def start(self, leg: str) -> None:
        self.drawing, self.points = leg, []
        self.render()

    def undo(self) -> None:
        if self.points:
            self.points.pop()
            self.render()

    def finish(self) -> bool:
        if not self.drawing or len(self.points) < 3:
            return False
        self.zones.zones = [z for z in self.zones.zones if z.approach != self.drawing]
        self.zones.zones.append(Zone(self.drawing, list(self.points)))
        self.zones.zones.sort(key=lambda z: APPROACHES.index(z.approach))
        self.drawing, self.points = None, []
        self.render()
        self.changed.emit()
        return True

    def remove(self, leg: str) -> None:
        self.zones.zones = [z for z in self.zones.zones if z.approach != leg]
        self.render()
        self.changed.emit()


class CountWorker(QThread):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, path: str, zones: ZoneSet, opts):
        super().__init__()
        self.path, self.zones, self.opts = path, zones, opts
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from ..detect import count_video

            self.progress.emit(0.0, "Loading the YOLO model...")
            run = count_video(self.path, self.zones, self.opts, progress=self.progress.emit,
                              cancelled=lambda: self._cancel)
            self.done.emit(run)
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())


class MemoWorker(QThread):
    done = Signal(str, str)

    def __init__(self, summary: dict):
        super().__init__()
        self.summary = summary

    def run(self):
        self.done.emit(*write_memo(self.summary))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Turning Movement Counter {__version__}")
        self.resize(1360, 860)
        self.video: str | None = None
        self.duration_s = 0.0
        self.movements: list[Movement] = []
        self.stats: dict = {}
        self.table: CountTable | None = None
        self.worker: CountWorker | None = None
        self.memo_worker: MemoWorker | None = None
        self._build_menu()
        self._build_ui()
        self._update_zone_list()

    # --- layout ----------------------------------------------------------------------

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        for text, slot in [("Open video...", self.open_video), ("Load zones...", self.load_zones),
                           ("Save zones...", self.save_zones), ("Export CSV...", self.export_csv)]:
            a = QAction(text, self)
            a.triggered.connect(slot)
            m.addAction(a)
        m.addSeparator()
        q = QAction("Quit", self)
        q.triggered.connect(self.close)
        m.addAction(q)
        s = self.menuBar().addMenu("&Settings")
        k = QAction("Claude API key...", self)
        k.triggered.connect(self.set_key)
        s.addAction(k)
        h = self.menuBar().addMenu("&Help")
        d = QAction("Run demo count", self)
        d.triggered.connect(self.run_demo)
        h.addAction(d)
        a = QAction("About", self)
        a.triggered.connect(lambda: QMessageBox.about(self, "Turning Movement Counter", (
            f"Turning Movement Counter {__version__}\n\nYOLO vehicle detection + ByteTrack tracking, "
            "approach zones, 15-minute turning movement counts, peak hour and PHF.\n\n"
            "Student-built tool. Spot check the counts against a hand count before relying on them.\n\n"
            "AGPL-3.0. Uses Ultralytics YOLO (AGPL-3.0).")))
        h.addAction(a)

    def _build_ui(self):
        split = QSplitter()
        self.setCentralWidget(split)

        left = QWidget()
        lv = QVBoxLayout(left)
        self.view = FrameView()
        self.view.changed.connect(self._update_zone_list)
        lv.addWidget(self.view, 1)
        self.video_label = QLabel("No video loaded")
        lv.addWidget(self.video_label)

        zbox = QGroupBox("Approach zones (draw one polygon over each leg, both directions of travel)")
        zl = QHBoxLayout(zbox)
        self.leg = QComboBox()
        for a in APPROACHES:
            self.leg.addItem(LEG_NAMES[a], a)
        zl.addWidget(self.leg)
        for text, slot in [("Draw zone", self.start_zone), ("Undo point", self.view.undo),
                           ("Finish zone", self.finish_zone), ("Delete zone", self.delete_zone)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            zl.addWidget(b)
        self.zone_list = QLabel()
        zl.addWidget(self.zone_list, 1)
        lv.addWidget(zbox)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        sbox = QGroupBox("Count settings")
        f = QFormLayout(sbox)
        self.start_clock = QDateTimeEdit(QDateTime.currentDateTime().addSecs(-QDateTime.currentDateTime().time().second()))
        self.start_clock.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.start_clock.setCalendarPopup(True)
        f.addRow("Video starts at", self.start_clock)
        self.interval = QComboBox()
        self.interval.addItem("15 minutes", 15)
        self.interval.addItem("5 minutes", 5)
        f.addRow("Count interval", self.interval)
        self.model = QComboBox()
        from ..detect import MODELS

        for k, v in MODELS.items():
            self.model.addItem(v, k)
        f.addRow("Detector", self.model)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(2, 30)
        self.fps.setValue(10)
        self.fps.setSuffix(" frames/s")
        f.addRow("Analyze", self.fps)
        self.conf = QDoubleSpinBox()
        self.conf.setRange(0.05, 0.9)
        self.conf.setSingleStep(0.05)
        self.conf.setValue(0.30)
        f.addRow("Min. confidence", self.conf)
        self.u_turns = QCheckBox("Count U-turns")
        self.u_turns.setChecked(True)
        self.stitch = QCheckBox("Join tracks broken by occlusion")
        self.stitch.setChecked(True)
        self.annotated = QCheckBox("Save an annotated video next to the source")
        f.addRow(self.u_turns)
        f.addRow(self.stitch)
        f.addRow(self.annotated)
        rv.addWidget(sbox)

        row = QHBoxLayout()
        self.run_btn = QPushButton("Count vehicles")
        self.run_btn.setStyleSheet("font-weight:bold; padding:6px;")
        self.run_btn.clicked.connect(self.run_count)
        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        export = QPushButton("Export CSV")
        export.clicked.connect(self.export_csv)
        row.addWidget(self.run_btn, 1)
        row.addWidget(self.cancel_btn)
        row.addWidget(export)
        rv.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        rv.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        rv.addWidget(self.status)

        self.tabs = QTabWidget()
        self.counts = QTableWidget()
        self.counts.verticalHeader().setVisible(False)
        self.tabs.addTab(self.counts, "Counts")
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.tabs.addTab(self.summary, "Summary")
        mw = QWidget()
        ml = QVBoxLayout(mw)
        self.memo = QTextEdit()
        self.memo_btn = QPushButton("Write count memo")
        self.memo_btn.clicked.connect(self.write_memo)
        ml.addWidget(self.memo_btn)
        ml.addWidget(self.memo, 1)
        self.tabs.addTab(mw, "Memo")
        rv.addWidget(self.tabs, 1)
        split.addWidget(right)
        split.setSizes([820, 540])

    # --- zones -----------------------------------------------------------------------

    def start_zone(self):
        if self.view.image is None:
            QMessageBox.information(self, "No video", "Open a video first.")
            return
        leg = self.leg.currentData()
        self.view.start(leg)
        self.status.setText(f"Click the corners of the {LEG_NAMES[leg].lower()} zone, then Finish zone "
                            "(or right-click). Cover the lanes going in and out, a car length or two back "
                            "from the stop bar.")

    def finish_zone(self):
        if not self.view.finish():
            self.status.setText("A zone needs at least 3 points.")
        else:
            self.status.setText("")
            i = self.leg.currentIndex()
            self.leg.setCurrentIndex((i + 1) % self.leg.count())

    def delete_zone(self):
        self.view.remove(self.leg.currentData())

    def _update_zone_list(self):
        legs = [z.approach for z in self.view.zones.zones]
        self.zone_list.setText("Zones: " + (", ".join(legs) if legs else "none yet"))

    def load_zones(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load zones", "", "Zones (*.json)")
        if not path:
            return
        with open(path, encoding="utf-8") as fh:
            zs = ZoneSet.from_json(fh.read())
        if self.view.zones.frame_size and zs.frame_size and tuple(zs.frame_size) != tuple(self.view.zones.frame_size):
            sx = self.view.zones.frame_size[0] / zs.frame_size[0]
            sy = self.view.zones.frame_size[1] / zs.frame_size[1]
            zs = ZoneSet([Zone(z.approach, [(x * sx, y * sy) for x, y in z.points]) for z in zs.zones],
                         self.view.zones.frame_size)
        zs.frame_size = self.view.zones.frame_size or zs.frame_size
        self.view.zones = zs
        self.view.render()
        self._update_zone_list()

    def save_zones(self):
        default = os.path.splitext(self.video)[0] + ".zones.json" if self.video else "zones.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save zones", default, "Zones (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.view.zones.to_json())

    # --- video and counting ----------------------------------------------------------

    def open_video(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open intersection video", "", "Video (*.mp4 *.mov *.avi *.mkv *.m4v *.wmv);;All files (*)")
        if not path:
            return
        try:
            from ..detect import first_frame, video_info

            info = video_info(path)
            frame = first_frame(path, at_s=min(1.0, info.duration_s / 2))
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Could not open video", str(e))
            return
        self.video = path
        self.duration_s = info.duration_s
        self.view.zones = ZoneSet()
        self.view.set_frame(frame)
        self.video_label.setText(f"{os.path.basename(path)}  |  {info.width}x{info.height}, "
                                 f"{info.fps:.1f} fps, {info.duration_s / 60:.1f} min")
        side = os.path.splitext(path)[0] + ".zones.json"
        if os.path.exists(side):
            self.load_zones(side)
        self._update_zone_list()

    def _settings(self) -> CountSettings:
        return CountSettings(count_u_turns=self.u_turns.isChecked(), stitch=self.stitch.isChecked())

    def run_count(self):
        if not self.video:
            QMessageBox.information(self, "No video", "Open a video first, or try Help > Run demo count.")
            return
        problems = self.view.zones.validate()
        if problems:
            QMessageBox.warning(self, "Zones", "\n".join(problems))
            return
        from ..detect import RunOptions

        opts = RunOptions(model=self.model.currentData(), analysis_fps=self.fps.value(), conf=self.conf.value(),
                          count=self._settings())
        if self.annotated.isChecked():
            opts.annotated_path = os.path.splitext(self.video)[0] + ".annotated.mp4"
        self.worker = CountWorker(self.video, self.view.zones, opts)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("Stopping after this frame; counts so far will be shown.")

    def _on_progress(self, frac: float, msg: str):
        self.progress.setValue(int(frac * 1000))
        self.status.setText(msg)

    def _on_failed(self, tb: str):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status.setText("Counting failed.")
        QMessageBox.critical(self, "Counting failed", tb[-2500:])

    def _on_done(self, run):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setValue(1000)
        dur = run.seconds_analyzed if run.seconds_analyzed < run.info.duration_s - 1 else run.info.duration_s
        self.show_results(run.movements, run.stats, dur)

    def run_demo(self):
        """Count synthetic tracks on a drawn intersection: no video or model needed."""
        from ..synthetic import demo_frame, demo_zones, generate

        self.video = None
        self.view.zones = demo_zones()
        self.view.set_frame(demo_frame())
        self._update_zone_list()
        self.video_label.setText("Demo: two hours of simulated tracks, 10% of them broken by occlusion")
        self.start_clock.setDateTime(QDateTime(datetime(2026, 10, 22, 7, 0)))
        points, truth = generate(duration_s=7200, break_rate=0.1)
        acc = TrackAccumulator(self.view.zones, CountSettings(frame_diag=1469, count_u_turns=self.u_turns.isChecked(),
                                                              stitch=self.stitch.isChecked()))
        for tid, t, x, y, cls in points:
            acc.add(tid, t, x, y, cls)
        movements, stats = acc.movements()
        stats["true_vehicles"] = sum(truth.values())
        self.show_results(movements, stats, 7200.0)

    # --- results ---------------------------------------------------------------------

    def show_results(self, movements: list[Movement], stats: dict, duration_s: float):
        self.movements, self.stats, self.duration_s = movements, stats, duration_s
        start = self.start_clock.dateTime().toPython()
        self.table = build_table(movements, duration_s, self.interval.currentData(), start)
        t = self.table
        heads = ["Interval"] + [f"{d} {t_}" for d in DIRECTIONS for t_ in TURNS] + ["Total"]
        self.counts.setColumnCount(len(heads))
        self.counts.setHorizontalHeaderLabels(heads)
        for i, (d, t_) in enumerate([(d, t_) for d in DIRECTIONS for t_ in TURNS]):
            self.counts.horizontalHeaderItem(i + 1).setToolTip(
                f"{BOUND_NAME[d]} (from the {FROM_LEG[d].lower()} leg), {TURN_NAME[t_]}")
        rows = [(t.label(iv.start_s) + ("*" if not iv.full else ""), iv.counts, False) for iv in t.intervals]
        rows.append(("Total", t.totals(), True))
        if t.peak:
            rows.append(("Peak hr", t.peak.counts, True))
        self.counts.setRowCount(len(rows))
        peak_rows = set(range(t.peak.first * (15 // t.interval_min), (t.peak.first + 4) * (15 // t.interval_min))) if t.peak else set()
        for r, (label, c, bold) in enumerate(rows):
            vals = [label] + [c[(d, t_)] for d in DIRECTIONS for t_ in TURNS] + [sum(c.values())]
            for k, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if bold:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                if r in peak_rows:
                    it.setBackground(QColor(255, 230, 150))
                    it.setForeground(QColor("black"))
                self.counts.setItem(r, k, it)
        self.counts.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.summary.setHtml(self._summary_html())
        self.memo.clear()
        self.status.setText(f"Counted {len(movements)} vehicles. Peak hour rows are highlighted.")
        self.tabs.setCurrentIndex(0)

    def _summary_html(self) -> str:
        t = self.table
        s = summary_dict(t, self.video or "demo", self.stats)
        html = [f"<h3>{s['total_vehicles']} vehicles, {s['count_start']} to {s['count_end']}</h3>"]
        p = s["peak_hour"]
        if p:
            html.append(f"<p><b>Peak hour</b> {p['start']} to {p['end']}: {p['volume']} vehicles, "
                        f"highest 15 min {p['highest_15_min_volume']}, <b>PHF {p['phf']:.2f}</b>, "
                        f"heavy vehicles {p['heavy_vehicle_pct']}%</p>")
        else:
            html.append("<p>Peak hour and PHF need at least 60 minutes of video.</p>")
        html.append("<p><b>Approach totals</b><br>" + "<br>".join(f"{k}: {v}" for k, v in s["approach_totals"].items()) + "</p>")
        html.append("<p><b>Vehicle classes</b><br>" + "<br>".join(f"{k}: {v}" for k, v in s["vehicle_classes"].items()) + "</p>")
        st = self.stats
        html.append(f"<p><b>Tracking</b><br>Tracks seen: {st.get('tracks', 0)}<br>Vehicles counted: {st.get('counted', 0)}"
                    f"<br>Broken tracks joined: {st.get('stitched', 0)}<br>Tracks not counted (seen in fewer than two legs): "
                    f"{st.get('not_counted', 0)}</p>")
        if "true_vehicles" in st:
            html.append(f"<p>Demo truth: {st['true_vehicles']} vehicles were simulated.</p>")
        return "".join(html)

    def export_csv(self):
        if self.table is None:
            QMessageBox.information(self, "Nothing to export", "Run a count first.")
            return
        base = os.path.splitext(self.video)[0] if self.video else "demo"
        path, _ = QFileDialog.getSaveFileName(self, "Export counts", base + ".tmc.csv", "CSV (*.csv)")
        if not path:
            return
        write_tmc_csv(path, self.table, os.path.basename(self.video or "demo"), self.stats)
        long_path = os.path.splitext(path)[0] + ".long.csv"
        write_long_csv(long_path, self.table)
        self.status.setText(f"Saved {os.path.basename(path)} and {os.path.basename(long_path)}")

    def write_memo(self):
        if self.table is None:
            QMessageBox.information(self, "No counts", "Run a count first.")
            return
        self.memo_btn.setEnabled(False)
        self.memo.setPlainText("Writing...")
        self.memo_worker = MemoWorker(summary_dict(self.table, self.video or "demo", self.stats))
        self.memo_worker.done.connect(self._on_memo)
        self.memo_worker.start()

    def _on_memo(self, text: str, source: str):
        self.memo_btn.setEnabled(True)
        note = "Written by Claude from the computed counts." if source == "claude" else \
            "Template memo (add a Claude API key under Settings for a written one)."
        self.memo.setPlainText(text + "\n\n" + note)

    def set_key(self):
        key, ok = QInputDialog.getText(self, "Claude API key", "Anthropic API key (stored in your AppData folder):",
                                       QLineEdit.EchoMode.Password, api_key() or "")
        if ok:
            save_api_key(key)


def selftest() -> int:
    """Build the window, run the demo count, and load the detector. Exit code 0 on success."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.run_demo()
    assert win.table is not None and win.table.peak is not None
    assert abs(win.stats["counted"] - win.stats["true_vehicles"]) <= 0.02 * win.stats["true_vehicles"]
    assert win.counts.rowCount() == 8 + 2
    from ..detect import selftest_model

    assert selftest_model() >= 80
    app.processEvents()
    return 0


def main():
    if "--selftest" in sys.argv:
        try:
            sys.exit(selftest())
        except Exception:  # noqa: BLE001
            # The windowed exe has no console, so leave the traceback where CI can print it.
            import tempfile

            with open(os.path.join(tempfile.gettempdir(), "tmc-selftest.log"), "w", encoding="utf-8") as fh:
                fh.write(traceback.format_exc())
            traceback.print_exc()
            sys.exit(1)
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args and os.path.exists(args[0]):
        win.open_video(args[0])
    sys.exit(app.exec())
