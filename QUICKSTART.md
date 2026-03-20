# 損傷読解・スコアリングシステム クイックスタートガイド

## 📋 前提条件

- Python 3.10以上
- NVIDIA GPU 16GB VRAM
- CUDA 11.8以上

## 🚀 セットアップ

### 1. 依存関係のインストール

```powershell
# 仮想環境作成（推奨）
python -m venv venv
.\venv\Scripts\Activate.ps1

# 依存関係インストール
pip install -r requirements.txt
```

### 2. モデルのダウンロード

初回実行時に自動的にHugging Face Hubからモデルがダウンロードされます：
- Granite-Vision: `ibm-granite/granite-vision-3b`
- Swallow-LLM: `tokyotech-llm/Swallow-7b-instruct-v0.1`

※ Hugging Faceのトークンが必要な場合があります：
```powershell
$env:HF_TOKEN="your_huggingface_token"
```

## 📊 使い方

### A. デモノートブック（推奨）

Jupyterノートブックで段階的に試す：

```powershell
jupyter notebook notebooks/demo.ipynb
```

### B. コマンドライン実行

#### 単一画像を処理

```powershell
python src/pipeline/end_to_end.py `
  --input data/images_human_inspect_n254/kensg-rebarexposureRb_001.png `
  --output data/outputs
```

#### ディレクトリ内の画像を一括処理

```powershell
# 全画像（254枚）を処理
python src/pipeline/end_to_end.py `
  --input data/images_human_inspect_n254 `
  --output data/outputs `
  --pattern "*.png"

# 最初の10枚だけ処理（テスト用）
python src/pipeline/end_to_end.py `
  --input data/images_human_inspect_n254 `
  --output data/outputs `
  --limit 10
```

### C. Pythonスクリプトから使用

```python
from pathlib import Path
from src.pipeline.end_to_end import DamageAnalysisPipeline

# パイプライン初期化
pipeline = DamageAnalysisPipeline('config.yaml')

# 単一画像処理
result = pipeline.process_image('data/images_human_inspect_n254/kensg-rebarexposureRb_001.png')

print(f"優先度: {result.score['priority_level']}")
print(f"説明: {result.score['priority_description']}")

# 一括処理
results = pipeline.process_batch(
    input_dir='data/images_human_inspect_n254',
    pattern='*.png',
    limit=10
)

# 結果保存
pipeline.save_results(results, 'data/outputs')
```

## 📁 出力ファイル

処理結果は `data/outputs/` に保存されます：

```
data/outputs/
├── descriptions/          # テキスト説明
│   ├── kensg-rebarexposureRb_001_description.txt
│   └── ...
├── structured/           # JSON構造化データ
│   ├── kensg-rebarexposureRb_001_structured.json
│   └── ...
├── scores/              # スコアリング結果
│   ├── kensg-rebarexposureRb_001_score.json
│   └── ...
├── results.csv          # 全体結果（CSV）
└── results.json         # 全体結果（JSON）
```

### 結果CSVの形式

| image_name | damage_type | severity | location | risk | priority_score | priority_level | priority_description |
|------------|-------------|----------|----------|------|----------------|----------------|---------------------|
| kensg-rebarexposureRb_001.png | rebar_exposure | high | girder_bottom | structural | 0.950 | 5 | 即時補修が必要 |

## ⚙️ 設定カスタマイズ

`config.yaml` で各種設定を調整できます：

```yaml
# モデル変更
granite_vision:
  model_name: "ibm-granite/granite-vision-8b"  # より大きいモデル

# スコアリング重み調整
scoring:
  weights:
    damage_type: 0.40  # 損傷種別の重要度を上げる
    severity: 0.35
    location: 0.15
    risk: 0.10
```

## 🔧 トラブルシューティング

### メモリ不足エラー

GPU VRAMが不足する場合：
- より小さいモデルを使用（granite-vision-3b）
- バッチサイズを減らす

### モデルダウンロードエラー

```powershell
# プロキシ設定（必要に応じて）
$env:HTTP_PROXY="http://proxy.example.com:8080"
$env:HTTPS_PROXY="http://proxy.example.com:8080"

# オフライン環境の場合、事前にモデルをダウンロード
python -c "from transformers import AutoModel; AutoModel.from_pretrained('ibm-granite/granite-vision-3b')"
```

## 📈 次のステップ

1. **精度向上**: GAMモデルで補正
   ```python
   # 人手ラベルデータで学習
   # models/gam_scorer.pkl を生成
   ```

2. **評価**: 人手ラベルとの比較
   ```python
   # data/annotations.csv に正解データを用意
   ```

3. **UI開発**: Gradio/Streamlitでインタラクティブなインターフェース

## 📞 サポート

問題が発生した場合：
- ログを確認: `logs/`
- エラーメッセージをコピーして報告
- 環境情報を含める（GPU、Python、CUDAバージョン）
