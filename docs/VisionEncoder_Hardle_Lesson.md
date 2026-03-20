# Vision Encoder実装失敗から学んだ厳しい教訓

**作成日**: 2026年3月20日  
**対象バージョン**: v0.2開発中  
**目的**: 複数VLM統合試行の失敗を記録し、今後の実装判断に活用

---

## エグゼクティブサマリー

v0.2では、異なるVLMアーキテクチャ（Granite Vision / Qwen2-VL / LLaVA）の比較を目標に、4つのモデル統合を試行しました。**結果として全て失敗**し、最終的にLLaVA量子化比較（プランA）に方針転換しました。

### 試行結果一覧

| モデル | サイズ | 実装方法 | 結果 | 失敗理由 |
|--------|--------|----------|------|----------|
| Granite Vision 3.3-2B | 6GB (HF) / 1.55GB (GGUF) | HuggingFace | ❌ 失敗 | CUDA device-side assert triggered |
| Granite Vision 3.3-2B | 1.55GB (GGUF) | llama-cpp-python | ❌ 失敗 | mmproj非互換（Failed to load mtmd context） |
| Granite Vision 3.3-2B | 1.55GB | Ollama | ❌ 失敗 | 文字化け（"I I I..."） |
| Granite Vision 3.3-2B | 6GB | CPU版 | ❌ キャンセル | 処理時間が長すぎる（ユーザーキャンセル） |
| Qwen2-VL 7B AWQ | 6.92GB | HuggingFace + autoawq | ❌ 失敗 | Windows非対応（triton依存） |
| Qwen2-VL 7B GPTQ | 未検証 | - | 未実施 | AWQ失敗後、試行せず |
| InternVL2-2B | 4.41GB | HuggingFace | ❌ 失敗 | chat関数インターフェース問題 |
| MoE-LLaVA | - | - | 検討中止 | 7Bクラス不在、低人気（最大101 downloads） |

**成功率**: 0/8試行（0%）  
**最終方針**: LLaVA量子化比較（Q4_K_M / Q5_K_M / Q8_0）に転換

---

## 1. Granite Vision 3.3-2B: 4つの実装方法で完全失敗

### 1.1 HuggingFace Transformers実装（試行1）

**実装ファイル**: `src/vision/granite_vision_7b.py`  
**モデル**: `ibm-granite/granite-3.3-2b-instruct`  
**ダウンロード**: 成功（6GB、約15分）  
**結果**: ❌ CUDAエラーで失敗

#### エラー詳細

```
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call...
```

**発生箇所**: `AutoProcessor` + `AutoModelForVision2Seq`のトークナイザー処理  
**推定原因**:
- Granite Visionのトークナイザー実装が特殊
- CUDAカーネル内でのアサーション違反（インデックス範囲外？）
- Transformers 4.57.6との互換性問題の可能性

#### 試行した対策

1. **CPU版への切り替え**（test_granite_cpu.py）
   - 結果: ユーザーが処理時間の長さでキャンセル
   - 推定時間: 数分/枚（GPU版の10倍以上）

2. **改善版実装**（granite_vision_improved.py）
   - `torch.cuda.empty_cache()`追加
   - `with torch.no_grad()`コンテキスト追加
   - 結果: 同じCUDAエラー継続

3. **エラーハンドリング強化**
   - `try-except`で詳細エラー取得
   - 結果: "CUDA error: device-side assert triggered"のみ、詳細不明

**教訓**:
- ✅ HuggingFace Transformersは万能ではない
- ✅ 新しいVLMアーキテクチャは十分な検証期間が必要
- ✅ CUDAエラーはデバッグが困難（低レベルエラー）
- ❌ CPU版は実用性なし（処理時間が長すぎる）

---

### 1.2 GGUF + llama-cpp-python実装（試行2）

**実装ファイル**: `src/vision/granite_vision_gguf.py`  
**モデル**: `bartowski/granite-vision-3.3-2b-instruct-GGUF` (Q4_K_M)  
**ダウンロード**: 成功（1.55GB、約2分）  
**結果**: ❌ mmproj非互換で失敗

#### エラー詳細

```
ValueError: Failed to load mtmd context from: models\mmproj-model-f16.gguf
```

**原因分析**:
- llama-cpp-pythonは**LLaVAアーキテクチャ専用**のmmproj実装
- Granite VisionのVision Encoder形式が異なる
- mmproj-model-f16.gguf（LLaVA用）が互換性なし

#### 試行した対策

1. **Granite Vision用mmproj検索**
   - HuggingFace検索: bartowskiリポジトリを確認
   - 結果: mmproj GGUFファイルなし（model-Q4_K_M.ggufのみ）

2. **llama-cpp-pythonドキュメント確認**
   - Vision対応: LLaVA 1.5/1.6のみ明記
   - 他VLMアーキテクチャ: 非対応

**教訓**:
- ✅ llama-cpp-python/Ollama = LLaVAアーキテクチャ専用
- ✅ GGUF量子化 ≠ 全VLMで利用可能
- ✅ mmproj形式はモデルアーキテクチャ固有
- ❌ 他VLMでllama-cpp-pythonを使用する前提は誤り

---

### 1.3 Ollama実装（試行3）

**実装**: Modelfile作成 + ollama create  
**モデル**: `bartowski/granite-vision-3.3-2b-instruct-GGUF` (Q4_K_M)  
**インポート**: 成功（ollama list で確認）  
**結果**: ❌ 文字化けで失敗

#### 問題詳細

**出力サンプル**:
```
I I I I I I I I I I I I I I I I I I I I I...
（無限ループのような繰り返し）
```

**推定原因**:
- トークナイザーの不正動作（デコード失敗）
- Ollama内部のVision機能が動作していない
- GGUF形式とOllamaの互換性問題

#### 試行した対策

1. **Modelfileパラメータ調整**
   - `temperature 0.3` → `0.1`に変更
   - `num_predict 300`追加
   - 結果: 同じ文字化け継続

2. **Ollama再起動**
   - `ollama stop` + `ollama serve`
   - 結果: 変化なし

3. **テキストプロンプトのみでテスト**
   - 画像なし、テキストのみ
   - 結果: 正常動作（Vision機能のみ問題）

**教訓**:
- ✅ OllamaのVision対応 = LLaVAアーキテクチャのみ
- ✅ GGUF形式でインポート成功 ≠ 動作保証なし
- ✅ Vision機能は特に互換性が厳しい
- ❌ Ollamaの"vision"タグは信頼性低い

---

### 1.4 Granite Vision総括

**失敗要因まとめ**:
1. **アーキテクチャ互換性**: LLaVA専用ツール（llama-cpp-python/Ollama）では動作不可
2. **HuggingFace Transformers**: CUDAエラー（低レベル、デバッグ困難）
3. **ドキュメント不足**: Granite Vision固有の実装情報が少ない
4. **検証期間不足**: リリース後の実用化にはさらに時間が必要

**最終判断**: Granite Visionは現環境では実用不可と断念

---

## 2. Qwen2-VL 7B AWQ: Windows環境制約で失敗

### 2.1 AWQ量子化実装

**実装ファイル**: `src/vision/qwen2vl_awq.py`  
**モデル**: `Qwen/Qwen2-VL-7B-Instruct-AWQ` (4-bit)  
**ダウンロード**: 成功（6.92GB、約10分、hf_transfer有効）  
**結果**: ❌ Windows非対応で失敗

#### エラー詳細

```bash
pip install autoawq

ERROR: Cannot install autoawq... because these package versions have conflicting dependencies.

The conflict is caused by:
    autoawq depends on triton
```

**原因分析**:
- `autoawq`パッケージが`triton`に依存
- `triton`パッケージ: **CUDA/Linux専用**、Windows非対応
- PyPI上のautoawqはLinux環境前提

#### 試行した対策

1. **autoawq-tricksなど代替パッケージ検索**
   - 結果: Windows対応版なし

2. **GPTQ版への切り替え検討**
   ```python
   # Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4
   # Downloads: 25,491
   ```
   - 結果: AWQ失敗後、時間制約で試行せず

3. **WSL2環境での実行検討**
   - 要考慮: CUDA Toolkit再インストール必要
   - 結果: 時間制約で試行せず

**教訓**:
- ✅ 量子化ライブラリのWindows対応は要確認
- ✅ AWQ/GPTQ/GGUF: それぞれ異なる依存関係
- ✅ triton依存 = Linux専用と判断すべき
- ❌ HuggingFaceの人気モデル ≠ Windows対応とは限らない

---

### 2.2 Qwen2-VL総括

**失敗要因**:
- **環境制約**: autoawq（triton依存）がWindows非対応
- **代替案の時間不足**: GPTQ版は未検証

**最終判断**: Windows環境ではQwen2-VL AWQは実装不可

**参考**: 
- GPTQ版（`Qwen2-VL-7B-Instruct-GPTQ-Int4`）はWindows対応の可能性あり
- 今後の検討候補として保留

---

## 3. InternVL2-2B: chat関数インターフェース問題で失敗

### 3.1 実装試行

**実装ファイル**: `src/vision/internvl2.py`  
**モデル**: `OpenGVLab/InternVL2-2B`  
**ダウンロード**: 成功（4.41GB、約9分、速度7.99MB/s）  
**依存パッケージ**: einops 0.8.2、timm 1.0.25（追加インストール成功）  
**結果**: ❌ chat関数エラーで失敗

#### エラー詳細

```
AttributeError: 'Image' object has no attribute 'shape'
```

**発生箇所**: `model.chat()`関数内でpixel_values処理時  
**推定原因**: InternVL2固有のchat関数インターフェース、ドキュメント不足

#### 試行した対策（複数回修正）

1. **PIL Imageを文字列パスに変換**
   ```python
   response = model.chat(
       tokenizer,
       pixel_values=None,
       question=prompt,
       generation_config=gen_config,
       image_path=image_path  # 文字列パス
   )
   ```
   - 結果: 同じエラー（`image_path`パラメータ無効）

2. **pixel_values=Noneアプローチ**
   ```python
   response = model.chat(
       tokenizer,
       pixel_values=None,
       question=f"<image>\n{prompt}",
       generation_config=gen_config
   )
   ```
   - 結果: 同じエラー（内部でpixel_values.shapeアクセス）

3. **model.load_image()メソッド使用**
   ```python
   pixel_values = model.load_image(image_path)
   response = model.chat(...)
   ```
   - 結果: `AttributeError: 'InternVLChatModel' object has no attribute 'load_image'`

4. **torchvision.transforms使用**
   ```python
   from torchvision import transforms
   preprocess = transforms.Compose([
       transforms.Resize((448, 448)),
       transforms.ToTensor(),
       transforms.Normalize(...)
   ])
   pixel_values = preprocess(image).unsqueeze(0)
   ```
   - 結果: 同じエラー（model.chat()内部でshape属性アクセス）

**問題の本質**:
- InternVL2の`chat()`関数インターフェースが独特
- `pixel_values`の期待される形式が不明
- 公式ドキュメント/サンプルコードが不足

**教訓**:
- ✅ 新しいVLMモデルは公式サンプルコード必須
- ✅ chat関数のインターフェースはモデル固有
- ✅ エラーメッセージだけでは問題解決困難
- ❌ HuggingFaceのTransformersライブラリも万能ではない

---

### 3.2 InternVL2総括

**失敗要因**:
- **インターフェース問題**: chat関数の正しい使用方法が不明
- **ドキュメント不足**: 公式サンプルコード不在
- **時間制約**: 複数回修正試行も解決できず

**最終判断**: InternVL2-2Bは現時点で実装困難

**参考**:
- einops、timmのインストールは成功
- モデルのロード自体は成功
- 推論部分のみ問題

---

## 4. MoE-LLaVA: 検討中止

### 4.1 初期調査

**検索結果**:
```
mllava/mllava-phi-3-mini-4k-mix-v2 - 101 downloads
```

**問題点**:
- 人気が極めて低い（最大101 downloads）
- 7Bクラスのモデルが不在
- 実験的モデルのみ（1.6-2.7Bクラス）
- GGUF対応も確認できず

**判断**: 実用性が低いため、試行せず

**教訓**:
- ✅ ダウンロード数は実用性の重要指標
- ✅ 人気モデル = コミュニティサポート期待できる
- ❌ 100 downloads未満のモデルは避けるべき

---

## 5. 技術的洞察と教訓

### 5.1 VLMツールの制約

| ツール | 対応アーキテクチャ | 制約 | 推奨用途 |
|--------|-------------------|------|----------|
| llama-cpp-python | LLaVA 1.5/1.6のみ | mmproj形式固有 | LLaVA専用 |
| Ollama | LLaVA系のみ | Vision機能限定 | LLaVA専用 |
| HuggingFace Transformers | 多様（モデル依存） | CUDAエラー多発 | 公式サポート確認必須 |
| autoawq | 多様 | Windows非対応 | Linux専用 |

**重要な結論**:
- **llama-cpp-python/Ollama = LLaVAアーキテクチャ専用**
- 他VLMアーキテクチャでは使用不可（mmproj非互換）
- GGUF量子化 ≠ 全VLMで利用可能

### 5.2 環境制約の理解

1. **Windows制約**:
   - `triton`パッケージ: CUDA/Linux専用
   - `autoawq`: Windows非対応（triton依存）
   - WSL2でも要考慮: CUDA Toolkit再構築必要

2. **CUDA制約**:
   - デバイスアサーションエラー: デバッグ困難
   - 低レベルエラーは再現性が低い
   - CPU版は実用性なし（処理時間長すぎる）

### 5.3 モデル選定基準の再定義

**成功するために必要な条件**:
1. ✅ **豊富なコミュニティサポート**（ダウンロード数10万以上）
2. ✅ **公式サンプルコード存在**（GitHubリポジトリ活発）
3. ✅ **環境互換性確認**（Windows/Linux、CUDA要件）
4. ✅ **量子化対応確認**（GGUF/GPTQ/AWQ、ツール依存関係）
5. ✅ **十分な検証期間**（リリース後3ヶ月以上）

**失敗する可能性が高い条件**:
1. ❌ ダウンロード数1万未満（MoE-LLaVA: 101）
2. ❌ リリース直後のモデル（Granite Vision: 2024年11月リリース）
3. ❌ 公式サンプルコードなし（InternVL2: chat関数ドキュメント不足）
4. ❌ 特殊な依存関係（Qwen2-VL AWQ: triton依存）

---

## 6. 最終方針: プランA採用

### 6.1 方針転換の理由

**当初目標**: Granite Vision / Qwen2-VL / LLaVAの3モデル比較  
**失敗**: 4モデル、8試行全てで失敗（成功率0%）  
**方針転換**: LLaVA量子化比較（Q4_K_M / Q5_K_M / Q8_0）

**プランA採用理由**:
1. ✅ **既存実装の信頼性**: LLaVA 1.5 7Bは50枚テスト100%成功
2. ✅ **実装容易性**: 同一アーキテクチャ、モデルファイル変更のみ
3. ✅ **実用的価値**: 量子化トレードオフ（サイズ/速度/精度）分析
4. ✅ **早期完了可能**: ダウンロード数時間、比較実行30分
5. ✅ **リスクゼロ**: 技術的不確実性なし

### 6.2 プランA実装状況

**モデル**:
- Q4_K_M: 既存（models/ggml-model-q4_k.gguf、4.1GB）
- Q5_K_M: ダウンロード完了（models/llava-1.5-7b-Q5_K_M.gguf、4.78GB）
- Q8_0: ダウンロード中（models/llava-1.5-7b-Q8_0.gguf、7.16GB、推定10時間）

**スクリプト**:
- `compare_llava_quantizations.py`: 完成（341行）
- 機能: 3モデル比較、VRAM測定、速度測定、CSV出力

**実行予定**:
- Q8_0ダウンロード完了後、3モデル比較テスト（30分）

---

## 7. 今後の推奨事項

### 7.1 新規VLM統合時のチェックリスト

**実装前**:
- [ ] ダウンロード数10万以上確認
- [ ] 公式サンプルコード存在確認（GitHub/HuggingFace）
- [ ] 依存パッケージのWindows互換性確認（triton等）
- [ ] 量子化ツールの対応確認（llama-cpp-python/autoawq/GPTQ）
- [ ] リリース後3ヶ月以上経過確認

**実装中**:
- [ ] 小規模テスト（1-2枚）で動作確認
- [ ] エラー発生時、再現性確認（3回試行）
- [ ] CPU版も準備（CUDA問題時のフォールバック）
- [ ] 処理時間測定（実用性判断）

**失敗判断基準**:
- 同じエラーが3回連続で発生
- 公式サンプルコードで動作しない
- 処理時間がGPU版で1分/枚以上
- CUDAエラーが低レベル（device-side assert等）

### 7.2 代替VLM候補（今後検討）

**実績あり（ダウンロード数多い）**:
1. **LLaVA-NeXT（LLaVA 1.6）**: 
   - HuggingFace人気モデル
   - llama-cpp-python対応確認必要

2. **Qwen2-VL GPTQ**: 
   - `Qwen2-VL-7B-Instruct-GPTQ-Int4`（25,491 downloads）
   - Windows互換性要確認

3. **ShareGPT4V（8B）**: 
   - LLaVAアーキテクチャ
   - GGUF対応確認必要

**実装優先順位**:
1. LLaVA-NeXT（llama-cpp-python対応確認後）
2. Qwen2-VL GPTQ（Windows互換性確認後）
3. ShareGPT4V（GGUF対応確認後）

### 7.3 プランAの次のステップ

**v0.2完成後**:
1. LLaVA量子化比較結果分析
2. 推奨量子化レベル決定
3. README更新（パフォーマンス比較表）
4. GitHub公開（v0.2完成）

**v0.3検討**:
- 代替VLM候補の慎重な検証
- LLaVA-NEXTが最有力（llama-cpp-python対応前提）
- 実装前に十分なリサーチ期間設定（1週間以上）

---

## 8. 結論

### 8.1 失敗からの学び

**技術的教訓**:
1. ✅ llama-cpp-python/Ollama = LLaVAアーキテクチャ専用
2. ✅ HuggingFace Transformers ≠ 万能（CUDAエラー多発）
3. ✅ 量子化ライブラリの環境制約は重大（Windows非対応多い）
4. ✅ 新しいVLMは公式サンプルコード必須
5. ✅ ダウンロード数 = 実用性の重要指標

**プロジェクト管理教訓**:
1. ✅ 早期失敗判断が重要（無駄な試行を避ける）
2. ✅ 複数実装方法の並行試行は効率的（Granite: 4方法）
3. ✅ 代替プラン準備が重要（プランA: LLaVA量子化比較）
4. ✅ 実証済み実装を優先すべき（LLaVA 50枚100%成功）

### 8.2 v0.2の最終目標

**当初目標**: ❌ Granite Vision / Qwen2-VL / LLaVAの3モデル比較  
**修正目標**: ✅ LLaVA量子化比較（Q4_K_M / Q5_K_M / Q8_0）

**実装完了予定**:
- Q8_0ダウンロード完了後（推定10時間）
- 3モデル比較テスト実行（30分）
- 比較結果レポート作成（1時間）
- v0.2完成、GitHub公開（30分）

### 8.3 メッセージ

> **「失敗は成功の母」ではなく「失敗は早期判断の材料」**
> 
> 8つの試行、全て失敗しましたが、それぞれから重要な技術的知見を獲得しました。特に、llama-cpp-python/OllamaがLLaVAアーキテクチャ専用であることは、今後のVLM統合判断に決定的な影響を与えます。
> 
> v0.2はプランAで完成させ、v0.3では今回の教訓を活かした慎重なモデル選定を行います。

---

## 付録A: 失敗ファイル一覧

### A.1 Granite Vision関連（7ファイル）

1. `src/vision/granite_vision.py` - 初期実装（古い）
2. `src/vision/granite_vision_7b.py` - HuggingFace実装（CUDAエラー）
3. `src/vision/granite_vision_gguf.py` - GGUF実装（mmproj非互換）
4. `src/vision/granite_vision_improved.py` - 改善版（CUDAエラー継続）
5. `test_granite_cpu.py` - CPU版テスト（ユーザーキャンセル）
6. `download_granite_vision.py` - ダウンローダー
7. `Modelfile.granite` - Ollama用（文字化け）

### A.2 Qwen2-VL関連（3ファイル）

1. `src/vision/qwen2vl_vision.py` - 初期実装（未使用）
2. `src/vision/qwen2vl_awq.py` - AWQ実装（Windows非対応）
3. `download_qwen2vl_awq.py` - AWQダウンローダー

### A.3 InternVL2関連（2ファイル）

1. `src/vision/internvl2.py` - InternVL2-2B実装（chat関数エラー）
2. `download_internvl2.py` - ダウンローダー

### A.4 v0.2試行関連（4ファイル）

1. `compare_vlm_v02.py` - 3モデル比較スクリプト（未使用）
2. `test_ollama_vlm_v02.py` - Ollamaテストスクリプト（文字化け確認）
3. `test_single_v02.py` - 単体テストスクリプト（未使用）
4. `download_gguf_vlms_v02.py` - 複数GGUFダウンローダー（未使用）

**合計**: 16ファイル

これらのファイルは、`failed/`フォルダーに移動し、今後の参照用に保管します。

---

## 付録B: 推奨リソース

### B.1 公式ドキュメント

- **llama-cpp-python**: https://github.com/abetlen/llama-cpp-python
  - Vision対応: LLaVA 1.5/1.6のみ
  - mmproj形式: LLaVA専用

- **Ollama**: https://ollama.ai/
  - Vision機能: LLaVA系のみ対応

- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
  - AutoModelForVision2Seq: モデル固有の実装確認必須

### B.2 モデルハブ

- **LLaVA GGUF（second-state）**: https://huggingface.co/second-state/Llava-v1.5-7B-GGUF
  - 12種類の量子化版（Q2_K～Q8_0）
  - 安定性高い、ダウンロード速度良好

- **bartowski GGUF**: https://huggingface.co/bartowski
  - 多様なモデルGGUF化
  - Granite Vision、LLaVA等

### B.3 量子化ライブラリ

- **autoawq**: Linux/CUDA専用（Windows非対応）
- **GPTQ**: Windows互換性あり（要確認）
- **GGUF**: llama.cpp形式、llama-cpp-python経由

---

**ドキュメント終了**  
**次のアクション**: 失敗ファイルを`failed/`フォルダーに移動

