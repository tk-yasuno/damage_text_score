# VLM比較実験 v0.2 - 実行可能モデル案

## 現状
- GPU: RTX 4060 Ti 16GB VRAM
- 現在: LLaVA 1.5 7B GGUF (50枚テスト実行中、5.3GB使用)
- 残りVRAM: ~10GB

## 問題
1. **Granite Vision 7B**: HuggingFaceに該当モデルなし (ibm-granite/granite-vision-7bは404)
2. **Qwen2-VL 7B**: `qwen_vl_utils`パッケージ未インストール

## 実用的な3モデル比較案

### プラン A: GGUF量子化レベル比較（最も簡単）
すでにダウンロード済みのモデルで量子化レベルを比較

1. **LLaVA 1.5 7B Q4_K_M** (現在使用中) - 4GB VRAM
2. **LLaVA 1.5 7B Q5_K_M** (要ダウンロード) - 5GB VRAM
3. **LLaVA 1.5 7B Q8_0** (要ダウンロード) - 7GB VRAM

**メリット**: 
- すぐ実行可能（ダウンロードのみ）
- 量子化による精度とサイズのトレードオフを検証
- 同一アーキテクチャで公平な比較

**必要な作業**:
```bash
# Q5_K_Mダウンロード (約5GB)
python download_llava_gguf.py --quantization q5_k_m

# Q8_0ダウンロード (約7GB)
python download_llava_gguf.py --quantization q8_0
```

### プラン B: HuggingFace VLM比較（時間必要）
異なるVLMアーキテクチャを比較

1. **LLaVA 1.5 7B GGUF** (現在使用中) - 4GB VRAM
2. **LLaVA 1.6 Mistral 7B** (HuggingFace) - 約14GB VRAM
3. **BLIP-2** (HuggingFace) - 約6GB VRAM

**メリット**: 
- 異なるアーキテクチャで多様性
- 最新のLLaVA 1.6を評価可能

**デメリット**:
- LLaVA 1.6は14GBなので、他のモデルと同時実行不可
- 初回ロード時にダウンロード発生

### プラン C: Qwen2-VL 導入（追加セットアップ必要）
パッケージインストールして新モデルテスト

1. **LLaVA 1.5 7B GGUF** (現在使用中)
2. **Qwen2-VL 2B** (軽量版) - 約4GB VRAM
3. **Qwen2-VL 7B** (フル版) - 約14GB VRAM

**必要な作業**:
```bash
pip install qwen-vl-utils
pip install git+https://github.com/huggingface/transformers@main
```

## 推奨：プランA (最も現実的)

LLaVA 50枚テストが実行中なので、完了を待ってから以下を実行：

### ステップ1: 追加モデルダウンロード
```bash
# Q8_0をダウンロード (約7GB、15-20分)
huggingface-cli download mys/ggml_llava-v1.5-7b --include "ggml-model-q8_0.gguf" --local-dir models/
```

### ステップ2: 比較テスト実行
```python
# test_quantization_compare.py を作成して実行
python test_quantization_compare.py --models q4km,q8_0 --images 10
```

### ステップ3: 評価軸
- **精度**: 損傷記述の詳細度と正確性
- **速度**: 1枚あたりの処理時間
- **VRAM**: メモリ使用量
- **JSON安定性**: Swallow-8Bでの構造化成功率

## 次のアクション

どのプランで進めますか？

A. GGUF量子化比較（推奨・30分で完了）
B. 異なるVLMアーキテクチャ比較（2-3時間）
C. Qwen2-VL導入（セットアップ1時間 + テスト2時間）
