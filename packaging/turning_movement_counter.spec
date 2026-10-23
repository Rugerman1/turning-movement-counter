# PyInstaller spec: one-folder build (PyTorch is too big to unpack on every launch
# of a single-file exe).  Run from the repo root after downloading the model:
#   python -c "from ultralytics.utils.downloads import attempt_download_asset as d; d('models/yolo26n.pt')"
#   pyinstaller packaging/turning_movement_counter.spec --noconfirm
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
MODEL = os.path.join(ROOT, "models", "yolo26n.pt")
if not os.path.exists(MODEL):
    raise SystemExit(f"Missing {MODEL}: download the model first (see the comment at the top of this file).")

datas = [(MODEL, "models")] + collect_data_files("ultralytics")  # tracker and model yaml files

# torchvision 0.2x ships its C++ ops as _C_stable.pyd, which the stock PyInstaller hook
# (still looking for _C) misses; without it importing torchvision fails with
# "operator torchvision::nms does not exist".  Copy every native file in the package.
import glob
import torchvision

TV = os.path.dirname(torchvision.__file__)
binaries = [(f, os.path.normpath(os.path.join("torchvision", os.path.dirname(os.path.relpath(f, TV)))))
            for pat in ("*.pyd", "*.dll") for f in glob.glob(os.path.join(TV, "**", pat), recursive=True)]
print("torchvision native files:", [os.path.basename(b[0]) for b in binaries])
hiddenimports = collect_submodules("ultralytics") + ["turning_movement_counter.ui.main_window", "lap"]

a = Analysis(
    [os.path.join(ROOT, "turning_movement_counter", "__main__.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "IPython", "jupyter", "notebook", "tensorboard", "tensorflow",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
              "PySide6.QtMultimedia", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtCharts",
              "PySide6.QtPdf", "PySide6.QtDataVisualization"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TurningMovementCounter",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TurningMovementCounter")
