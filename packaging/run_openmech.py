"""配布版（PyInstaller）の入口。 引数なしで GUI、`selfcheck --out ファイル` で動作確認"""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from openmech.cli import main
    sys.exit(main())
