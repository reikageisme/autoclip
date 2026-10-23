# PyInstaller: đóng gói dịch vụ AutoClip thành thư mục autoclip-service\ (onedir).
#   uv run --with pyinstaller pyinstaller packaging/autoclip-service.spec --noconfirm
# Kết quả: dist/autoclip-service/autoclip-service.exe — đặt cạnh concat.exe trong bộ cài.
# ruff: noqa
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPECPATH).parent

datas = [
    (str(root / "recipes"), "recipes"),
    (str(root / "autoclip" / "api" / "static"), "autoclip/api/static"),
    (str(root / "autoclip" / "db" / "migrations"), "autoclip/db/migrations"),
]
binaries = []
hiddenimports = (
    collect_submodules("uvicorn")
    + collect_submodules("autoclip")
    + collect_submodules("alembic")
    + collect_submodules("yt_dlp")
    + ["sqlalchemy.dialects.sqlite"]
)
for package in (
    "faster_whisper",
    "ctranslate2",
    "av",
    "scenedetect",
    "mcp",
    "imagehash",
    "tokenizers",
    "sherpa_onnx",
    "onnxruntime",
):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(root / "packaging" / "service_entry.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="autoclip-service",
    console=True,  # ReiCut bật nó với CREATE_NO_WINDOW nên không hiện cửa sổ
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="autoclip-service")
