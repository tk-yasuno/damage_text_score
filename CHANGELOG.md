# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-20

### 🎉 Initial Release - MVP v0.1

#### Added
- **3つのVisionモード実装**
  - llama-cpp-python + GGUF (推奨): LLaVA-1.5-7B Q4_K_M量子化版
  - HuggingFace Transformers: llava-1.5-7b-hf完全版
  - Ollama統合: llava:7b (CPU動作)

- **完全なパイプライン構築**
  - 前処理モジュール (`preprocessing/image_preprocessor.py`)
    - ノイズ除去 (Non-local Means Denoising)
    - リサイズ (最大1024x1024)
    - コントラスト調整 (CLAHE)
  - Vision分析モジュール (`vision/`)
    - llama_cpp_vision.py - GGUF推論エンジン
    - granite_vision.py - HuggingFace Transformers
    - ollama_vision.py - Ollama API統合
  - JSON構造化モジュール (`structuring/json_structurer.py`)
    - Swallow-8B via Ollama
    - 多段階JSONパース (標準/コードブロック/プレーンテキスト)
  - スコアリングモジュール (`scoring/priority_scorer.py`)
    - ルールベーススコアリング
    - 重み付け評価: 重症度(40%), 損傷タイプ(35%), 位置(15%), リスク(10%)

- **クイックスタートスクリプト** (`quickstart.py`)
  - 4段階テストモード: 1枚/10枚/50枚/254枚
  - UTF-8エンコーディング対応 (Windows PowerShell)
  - 進捗バー表示
  - CSV/JSON出力

- **モデルダウンロードスクリプト**
  - `download_llava_gguf.py` - LLaVA GGUF自動取得
  - HuggingFace Hub統合
  - 中断からの再開対応

- **設定管理** (`config.yaml`)
  - Vision/構造化/スコアリングパラメータ
  - モード切り替え対応
  - パス設定管理

#### Fixed
- **Windows文字化け問題解決**
  - PowerShell cp932エンコーディング対応
  - llama.cpp C++ログ抑制
  - UTF-8環境変数設定 (`LLAMA_CPP_LOG_LEVEL=0`)
  - stdout/stderrリダイレクト実装

- **Ollama GPU認識問題の回避**
  - llama-cpp-python採用によりOllama依存を排除
  - 直接CUDA統合で確実なGPU活用

- **JSON構造化の堅牢性向上**
  - 複数正規表現パターンによるパース
  - デバッグ出力追加
  - エラーハンドリング強化

#### Performance
- **実測パフォーマンス** (RTX 4060 Ti 16GB)
  - 1枚テスト: 42秒/枚
  - 10枚バッチ: 平均51.6秒/枚 (成功率100%)
  - GPU使用率: 100% (全レイヤーGPU配置)
  - VRAM使用量: 約8GB

- **モデル比較結果**
  | モード | 処理時間 | モデルサイズ | GPU使用 |
  |--------|----------|--------------|---------|
  | llama-cpp-python | 51.6秒 | 4.08GB | 100% |
  | HuggingFace | 45秒 | 14GB | 100% |
  | Ollama | 88秒 | 4.7GB | 0% (CPU) |

#### Documentation
- README.md v0.1 - 包括的なドキュメント
  - セットアップガイド
  - モデル比較表
  - トラブルシューティング
  - 使用方法・サンプルコード
- CHANGELOG.md - バージョン履歴管理

#### Tested
- ✅ 1枚テスト完了 (42秒, 優先度5判定)
- ✅ 10枚バッチテスト完了 (平均51.6秒, 成功率100%)
  - 優先度5: 6枚 (60%)
  - 優先度3: 4枚 (40%)

#### Known Issues
- Ollama版はCPU動作のため低速 (88秒/枚)
  - → llama-cpp-python版を推奨
- HuggingFace版は高VRAM要求 (14GB)
  - → GGUF量子化版で4GBに削減
- 初回実行時にTransformersインポートに時間がかかる
  - → 2回目以降は正常速度

#### Dependencies
- Python 3.12.10
- PyTorch 2.6.0+cu124
- Transformers 4.57.6
- llama-cpp-python 0.3.16 (CUDA)
- OpenCV 4.12.0
- pandas 2.2.3
- pyyaml 6.0.2
- tqdm 4.67.1

#### Infrastructure
- CUDA 12.4 support
- GPU: NVIDIA GeForce RTX 4060 Ti (16GB VRAM)
- OS: Windows 11
- Storage: 20GB required

---

## [Unreleased]

### Planned for v0.2 (2026 Q2)
- [ ] 50枚テスト実行・結果検証
- [ ] 全254枚処理完了
- [ ] 人間アノテーションとの精度比較
- [ ] バッチ処理最適化 (並列化)
- [ ] メモリ使用量削減

### Planned for v1.0
- [ ] Web UI実装 (Streamlit/Gradio)
- [ ] REST API サーバー
- [ ] Docker環境構築
- [ ] CI/CD パイプライン
- [ ] ユニットテスト追加
- [ ] GAMモデル統合
- [ ] リアルタイム処理対応

---

## Version History

- **v0.1.0** (2026-03-20) - Initial MVP release
- Future versions TBD

---

**Legend**:
- `Added` - 新機能
- `Changed` - 既存機能の変更
- `Deprecated` - 今後削除予定の機能
- `Removed` - 削除された機能
- `Fixed` - バグ修正
- `Security` - セキュリティ修正
- `Performance` - パフォーマンス改善
