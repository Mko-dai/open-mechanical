# -*- mode: python ; coding: utf-8 -*-
# 配布版（インストール不要の実行ファイル一式）を作る:  pyinstaller packaging/open-mechanical.spec --noconfirm
import os
import sys

import gmsh
from PyInstaller.utils.hooks import collect_all, collect_submodules

root = os.path.abspath(os.path.join(SPECPATH, ".."))
datas = [(os.path.join(root, "examples"), "examples"), (os.path.join(root, "README.md"), "."),
         (os.path.join(root, "LICENSE"), ".")]
binaries = [(gmsh.libpath, ".")]            # gmsh の本体（gmsh.py は自分と同じフォルダから探す）
hidden = collect_submodules("openmech") + ["gmsh"]
for pkg in ("pyvista", "pyvistaqt", "pyamg", "vtkmodules"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hidden += h

a = Analysis([os.path.join(SPECPATH, "run_openmech.py")], pathex=[root], binaries=binaries, datas=datas,
             hiddenimports=hidden, excludes=["tkinter", "pytest", "PyQt5", "PyQt6", "IPython", "notebook"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="open-mechanical", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="open-mechanical")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="open-mechanical.app", bundle_identifier="io.github.mko-dai.open-mechanical",
                 info_plist={"NSHighResolutionCapable": True, "CFBundleShortVersionString": "0.1.0"})
