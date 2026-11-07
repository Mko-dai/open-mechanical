# open-mechanical

Ansys Mechanical の代わりに使える、**無料の静的構造解析ソフト**（Python デスクトップアプリ）です。
操作の流れは Ansys と同じく **形状 → メッシュ → 材料 → 境界条件 → 解析 → 結果** です。

## できること（v0.1）

| 項目 | 内容 |
|---|---|
| 解析 | 線形静的構造解析（3次元・等方弾性・微小変形） |
| 要素 | 2次四面体 Tet10（Ansys の SOLID187 相当・既定）／1次四面体 Tet4 |
| 形状 | 簡単形状（直方体・円柱・パイプ・穴あき板・L字ブラケット・球）、**STEP**、IGES、**STL** |
| 拘束 | 固定支持、強制変位（成分ごとに拘束/自由 → ローラー支持など） |
| 荷重 | 力、圧力、モーメント、重力（加速度） |
| 結果 | 全変形・方向変形、相当応力（ミーゼス）、主応力、応力成分、**安全率**、反力、質量 |
| 表示 | コンター（Ansys 風 9 段階）、変形倍率、最大/最小の位置、右クリックで値を確認 |
| 出力 | HTML レポート（印刷→PDF）、VTU（ParaView）、CSV（Excel） |

### 対応していないもの
- **.x_t / .x_b（Parasolid）**：Siemens の独自形式で、無償で読めるライブラリが存在しません。
  CAD から **STEP** で書き出して読み込んでください（CAD の面がそのまま選択できます）。
- 非線形（塑性・大変形）、接触、モーダル、熱 … 今後の拡張候補です。
- 複数部品のアセンブリは「くっついた1つの物体」として扱います（接触なし）。

## インストール（Mac）

ターミナルで：

```bash
cd ~/kouchan_program/open-mechanical
./setup_mac.sh          # 初回だけ（仮想環境を作って必要なライブラリを入れます）
./run.sh                # 起動
```

手動で入れる場合：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m openmech
```

## 使い方

1. **形状**：簡単形状を作るか、「ファイルを開く…」で STEP/STL を読み込む
2. **メッシュ**：要素サイズ（0 = 自動）を決めて作成。2次要素推奨
3. **材料**：一覧（SS400、S45C、SUS304、A5052、A6061 など）から選ぶか値を入力
4. **境界条件**：3D 画面で面を **右クリック** で選択（Shift/⌘ で追加選択）→「固定支持」「力」などを押す
5. **解析**：「▶ 解析を実行」
6. **結果**：表示する結果を選択。右クリックでその点の値を表示。「レポート」で HTML レポートを作成

マウス：左ドラッグ＝回転、Shift＋左ドラッグ／中ドラッグ＝移動、ホイール＝拡大縮小

**単位系**：長さ mm、力 N、応力 MPa、モーメント N·mm、加速度 mm/s²（重力 = 9806.65）

## コマンドラインでの実行（バッチ解析）

```bash
python -m openmech faces examples/cantilever.omech      # 面番号の一覧
python -m openmech run   examples/cantilever.omech --out results/
```

プロジェクトファイル（.omech）は JSON なので、パラメータを書き換えて一括で回すこともできます。

## 精度の検証

`python -m pytest -q` で理論解との比較テストが走ります。

| 検証 | 結果 |
|---|---|
| 引張棒の応力・伸び（F/A, FL/EA） | 誤差 < 0.0001% |
| 片持ち梁の先端たわみ（Tet10, 断面 3×3 分割） | 理論値（ティモシェンコ梁）との差 約 1% |
| 片持ち梁の曲げ応力 Mc/I | 誤差 < 5% |
| 圧力・自重の反力の釣り合い | 誤差 < 0.0001% |

※ 1次要素 Tet4 は曲げで剛くなりすぎる（同条件で約 −20%）ので、通常は 2次要素を使ってください。Ansys も既定は 2次要素です。

## 計算速度について

ソルバーは自動で最速のものを選びます。大きなモデル（数十万自由度〜）を速く解きたい場合：

- `pip install pyamg` … AMG 反復法（Apple Silicon でも動作・requirements に含む）
- `brew install suite-sparse && pip install scikit-sparse` … CHOLMOD 直接法（高速）
- Intel Mac / Windows なら `pip install pypardiso`

## 構成

```
openmech/
  mesh.py        四面体メッシュ、境界面の抽出と面の自動分割
  solver.py      剛性マトリクス組立・境界条件・求解・応力計算
  geometry.py    gmsh による形状作成・CAD 読込・メッシュ生成
  materials.py   材料ライブラリ
  project.py     プロジェクトの保存/読込（.omech）
  export.py      VTU / CSV / HTML レポート
  cli.py         コマンドライン
  gui/           デスクトップアプリ（PySide6 + PyVista）
tests/           理論解との比較テスト
```

## 注意

本ソフトの結果は参考値です。重要な設計判断では、収束確認（メッシュを細かくして結果が変わらないこと）や
手計算・実験・商用ソフトとの比較で妥当性を確認してください。
