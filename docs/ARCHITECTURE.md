# プロジェクト構造ドキュメント

## ディレクトリ構造

```
damage_text_score/
│
├── README.md                    # プロジェクト概要
├── QUICKSTART.md               # クイックスタートガイド
├── config.yaml                 # システム設定
├── requirements.txt            # Python依存関係
├── quickstart.py              # クイックスタート実行スクリプト
├── .gitignore                 # Git除外設定
│
├── src/                       # ソースコード
│   ├── __init__.py
│   │
│   ├── preprocessing/         # 前処理モジュール
│   │   ├── __init__.py
│   │   └── image_preprocessor.py
│   │
│   ├── vision/               # Vision分析モジュール
│   │   ├── __init__.py
│   │   └── granite_vision.py
│   │
│   ├── structuring/          # JSON構造化モジュール
│   │   ├── __init__.py
│   │   └── json_structurer.py
│   │
│   ├── scoring/              # スコアリングモジュール
│   │   ├── __init__.py
│   │   └── priority_scorer.py
│   │
│   ├── pipeline/             # パイプラインモジュール
│   │   ├── __init__.py
│   │   └── end_to_end.py
│   │
│   └── utils/                # ユーティリティ
│       ├── __init__.py
│       └── config.py
│
├── notebooks/                # Jupyter ノートブック
│   └── demo.ipynb           # デモノートブック
│
├── data/                    # データディレクトリ
│   ├── images_human_inspect_n254/  # 入力画像（254枚）
│   │   ├── kensg-rebarexposureRb_001.png
│   │   ├── kensg-rebarexposureRb_002.png
│   │   └── ...
│   │
│   ├── preprocessed/        # 前処理済み画像
│   │
│   ├── outputs/             # 出力結果
│   │   ├── descriptions/    # テキスト説明
│   │   ├── structured/      # JSON構造化データ
│   │   ├── scores/         # スコアリング結果
│   │   ├── results.csv     # 全体結果（CSV）
│   │   └── results.json    # 全体結果（JSON）
│   │
│   └── annotations.csv      # 人手ラベル（オプション）
│
├── models/                  # モデル設定
│   ├── scoring_rules.yaml  # スコアリングルール
│   └── gam_scorer.pkl      # GAMモデル（オプション）
│
├── logs/                   # ログファイル
│
├── docs/                   # ドキュメント
│   └── ConceptNote_20260320.jpg
│
└── 0_LogBAK/              # バックアップログ
    ├── DamageImageToText - ショートカット.lnk
    ├── DamageTextRepairClassifier - ショートカット.lnk
    └── HealthTabNetRepair - ショートカット.lnk
```

## モジュール説明

### 1. preprocessing/ - 前処理モジュール
**役割**: 画像の品質向上
- ノイズ除去（Non-local Means Denoising）
- リサイズ（アスペクト比維持）
- コントラスト調整（CLAHE）

**主要クラス**:
- `ImagePreprocessor`: 前処理実行
- `PreprocessConfig`: 前処理設定

### 2. vision/ - Vision分析モジュール
**役割**: 画像から損傷説明テキスト生成
- Granite-Visionによる画像理解
- 専門用語での説明生成

**主要クラス**:
- `GraniteVisionAnalyzer`: Vision分析実行
- `VisionConfig`: Vision設定

### 3. structuring/ - JSON構造化モジュール
**役割**: テキストをJSON形式に構造化
- 損傷種別の識別
- 重症度の判定
- 位置・リスクの抽出

**主要クラス**:
- `JSONStructurer`: JSON構造化実行
- `DamageStructure`: 構造化データモデル
- `StructuringConfig`: 構造化設定

### 4. scoring/ - スコアリングモジュール
**役割**: 補修優先度の計算
- ルールベーススコアリング
- GAM補正（オプション）
- 優先度レベル判定（1-5）

**主要クラス**:
- `PriorityScorer`: スコアリング実行
- `PriorityScore`: スコア結果
- `ScoringConfig`: スコアリング設定

### 5. pipeline/ - パイプラインモジュール
**役割**: エンドツーエンド処理
- 全モジュールの統合
- バッチ処理
- 結果保存

**主要クラス**:
- `DamageAnalysisPipeline`: パイプライン実行
- `PipelineResult`: 処理結果

### 6. utils/ - ユーティリティ
**役割**: 共通機能
- 設定ファイル読み込み
- ディレクトリ管理

**主要クラス**:
- `Config`: 設定管理

## データフロー

```
入力画像 (PNG)
    ↓
[前処理] ImagePreprocessor
    ├─ ノイズ除去
    ├─ リサイズ
    └─ コントラスト調整
    ↓
[Vision分析] GraniteVisionAnalyzer
    └─ テキスト説明生成
    ↓
[構造化] JSONStructurer
    ├─ damage_type
    ├─ severity
    ├─ location
    └─ risk
    ↓
[スコアリング] PriorityScorer
    ├─ ルールベース計算
    ├─ 組み合わせボーナス
    └─ GAM補正（オプション）
    ↓
出力 (CSV/JSON)
    ├─ priority_score (0.0-1.0)
    ├─ priority_level (1-5)
    └─ priority_description
```

## 設定ファイル

### config.yaml
システム全体の設定を管理

```yaml
data:                      # データパス
preprocessing:             # 前処理設定
granite_vision:           # Visionモデル設定
structuring:              # 構造化モデル設定
scoring:                  # スコアリング設定
execution:                # 実行設定
logging:                  # ログ設定
```

### models/scoring_rules.yaml
スコアリングルールを管理

```yaml
damage_type_scores:       # 損傷種別スコア
severity_scores:          # 重症度スコア
location_scores:          # 位置スコア
risk_scores:              # リスクスコア
combination_bonuses:      # 組み合わせボーナス
priority_thresholds:      # 優先度閾値
priority_descriptions:    # 優先度説明
```

## 実行方法

### 1. コマンドライン
```powershell
python src/pipeline/end_to_end.py --input data/images_human_inspect_n254 --output data/outputs
```

### 2. Python スクリプト
```python
from src.pipeline.end_to_end import DamageAnalysisPipeline
pipeline = DamageAnalysisPipeline('config.yaml')
results = pipeline.process_batch('data/images_human_inspect_n254')
```

### 3. Jupyter ノートブック
```powershell
jupyter notebook notebooks/demo.ipynb
```

### 4. クイックスタート
```powershell
python quickstart.py
```

## 出力フォーマット

### CSV出力 (results.csv)
| 列名 | 説明 |
|-----|------|
| image_name | 画像ファイル名 |
| damage_type | 損傷種別 |
| severity | 重症度 |
| location | 位置 |
| risk | リスク種別 |
| priority_score | 優先度スコア（0.0-1.0） |
| priority_level | 優先度レベル（1-5） |
| priority_description | 優先度説明 |
| description | 損傷説明（200文字まで） |
| processing_time | 処理時間（秒） |
| status | 処理ステータス |

### JSON出力 (個別ファイル)

**descriptions/**
```json
"鉄筋露出が確認されます。コンクリート表面が剥離し..."
```

**structured/**
```json
{
  "damage_type": "rebar_exposure",
  "severity": "high",
  "location": "girder_bottom",
  "risk": "structural",
  "description_ja": "...",
  "key_features": ["鉄筋露出", "腐食"]
}
```

**scores/**
```json
{
  "raw_score": 0.950,
  "priority_level": 5,
  "priority_description": "即時補修が必要（構造安全性に重大な影響）",
  "damage_type_score": 0.95,
  "severity_score": 1.0,
  "location_score": 1.0,
  "risk_score": 1.0,
  "combination_bonus": 0.1
}
```

## 拡張ポイント

### 1. 新しい損傷種別の追加
`models/scoring_rules.yaml` に追加

### 2. スコアリングルールの調整
- 重み変更: `config.yaml` の `scoring.weights`
- ルール追加: `models/scoring_rules.yaml`

### 3. GAMモデルの学習
```python
from sklearn.preprocessing import LabelEncoder
from pygam import LinearGAM

# 訓練データ準備
# GAMモデル学習
# models/gam_scorer.pkl に保存
```

### 4. カスタムプロンプト
`config.yaml` の `granite_vision.prompt_template` を編集

## 依存関係

主要ライブラリ：
- `torch` >= 2.0.0: PyTorch
- `transformers` >= 4.35.0: Hugging Face Transformers
- `opencv-python` >= 4.8.0: 画像処理
- `pandas` >= 2.0.0: データ処理
- `pygam` >= 0.9.0: GAM（オプション）

## パフォーマンス

### 処理速度（GPU: 16GB VRAM）
- 単一画像: 約5-10秒
- 10枚バッチ: 約50-100秒
- 254枚全体: 約20-40分

### メモリ使用量
- Granite-Vision-3B: 約8GB VRAM
- Swallow-7B: 約10GB VRAM
- 合計: 約15GB VRAM（モデル切り替え時）

## トラブルシューティング

よくある問題と解決策は [QUICKSTART.md](QUICKSTART.md) を参照。
