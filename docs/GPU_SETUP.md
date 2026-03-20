# llama-cpp-python GPU対応セットアップガイド

**作成日**: 2026年3月20日  
**対象**: NVIDIA GPU（RTX 4060 Ti 16GB）  
**CUDA**: 12.4 / 12.6対応

---

## 現状

**インストール済み**: llama-cpp-python 0.3.16 (CPU版)  
**問題**: CUDA Toolkitは検出されるが、Visual Studio統合に問題がありビルドに失敗

```
Found CUDAToolkit: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/...
CMake Error: No CUDA toolset found.
```

**影響**:
- モデルロード: CPU版でも動作
- 推論速度: CPU版はGPU版の10-15倍遅い
- VRAM使用: GPU未使用（0.0 MB）

---

## GPU対応方法（3つ）

### 方法1: Visual Studio CUDA統合を修正（推奨、時間1-2時間）

#### 1.1 前提条件確認

```powershell
# CUDA Toolkitバージョン確認
nvcc --version

# Visual Studio確認
Get-Command cl.exe
```

**期待される出力**:
```
nvcc: NVIDIA (R) Cuda compiler driver
Visual Studio 2022 (または2019)
```

#### 1.2 Visual Studio C++コンポーネントインストール

1. **Visual Studio Installerを起動**
   ```powershell
   & "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe"
   ```

2. **必要なコンポーネント**:
   - ✅ "C++によるデスクトップ開発"
   - ✅ "MSVC v143 - VS 2022 C++ x64/x86ビルドツール"
   - ✅ "Windows 10 SDK"
   - ✅ "CMake tools for Windows"

#### 1.3 CUDA Toolkit再インストール（必要に応じて）

**CUDA 12.4推奨**（PyTorchと互換性が高い）

1. ダウンロード: [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)
2. **カスタムインストール**を選択
3. **必須コンポーネント**:
   - ✅ CUDA Compiler
   - ✅ CUDA Runtime
   - ✅ Visual Studio Integration
   - ✅ cuBLAS

#### 1.4 llama-cpp-pythonをCUDA対応でビルド

```powershell
# CPython版をアンインストール
pip uninstall llama-cpp-python -y

# 環境変数設定
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:FORCE_CMAKE = "1"

# CUDA対応版ビルド（5-10分）
pip install llama-cpp-python==0.3.16 --no-cache-dir --force-reinstall --verbose
```

**成功確認**:
```python
from llama_cpp import Llama

model = Llama(
    model_path="models/ggml-model-q4_k.gguf",
    n_gpu_layers=-1,  # 全レイヤーをGPU
    verbose=True
)
# "layer   0 assigned to device CUDA0" が表示されればOK
```

#### 1.5 トラブルシューティング

**エラー: "No CUDA toolset found"**

```powershell
# Visual StudioのCUDA統合確認
Get-ChildItem "C:\Program Files\Microsoft Visual Studio\*\*\MSBuild\Microsoft\VC\*\BuildCustomizations\CUDA*.props" -Recurse
```

存在しない場合:
1. CUDA Toolkitを**完全アンインストール**
2. Visual Studioを**完全アンインストール**
3. Visual Studio再インストール（C++コンポーネント含む）
4. CUDA Toolkit再インストール（Visual Studio Integration有効）

**エラー: "CUDA driver version is insufficient"**

```powershell
# ドライバー更新
# NVIDIA公式サイトから最新ドライバーダウンロード
# https://www.nvidia.com/Download/index.aspx
```

---

### 方法2: プリビルドCUDA版ホイールを使用（最速、5分）

#### 2.1 jllllll氏のプリビルド版（非公式、実績あり）

```powershell
pip uninstall llama-cpp-python -y

# CUDA 12.4対応版
pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu124

# または CUDA 12.1対応版
pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu121
```

**注意**: 
- 非公式ビルド（コミュニティ提供）
- セキュリティリスク考慮

#### 2.2 abetlen公式プリビルド版

```powershell
pip uninstall llama-cpp-python -y

# CUDA 12.4対応版
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

**注意**: 
- 2026年3月時点では、ソースからビルドしようとする可能性あり
- Windows版のプリビルドホイールが不足

---

### 方法3: CPU版のまま使用（現在の状態）

#### 3.1 利点

- ✅ 設定不要、すぐ使える
- ✅ 安定性が高い
- ✅ メモリ使用量が少ない

#### 3.2 欠点

- ❌ 処理時間が10-15倍長い
  - LLaVA Q4_K_M: 約3-5分/枚
  - LLaVA Q5_K_M: 約4-6分/枚
  - LLaVA Q8_0: 約5-7分/枚
  - **合計**: 10-15分（3モデル比較）

#### 3.3 CPU版用の設定

`compare_llava_quantizations.py`は既にCPU用に調整済み:

```python
model = Llama(
    model_path=config.model_path,
    chat_handler=chat_handler,
    n_gpu_layers=0,  # CPU版（CUDA未対応のため）
    n_ctx=4096,
    verbose=False,
    logits_all=True,
)
```

GPU対応後は`n_gpu_layers=-1`に変更。

---

## GPU対応確認方法

### 確認1: ログ出力でGPU確認

```python
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import os

os.environ['LLAMA_CPP_LOG_LEVEL'] = '1'  # 詳細ログ有効

handler = Llava15ChatHandler(clip_model_path='models/mmproj-model-f16.gguf')
model = Llama(
    model_path='models/ggml-model-q4_k.gguf',
    chat_handler=handler,
    n_gpu_layers=-1,  # 全レイヤーをGPU
    verbose=True
)

# GPU対応なら以下が表示される:
# load_tensors: layer   0 assigned to device CUDA0
# load_tensors: layer   1 assigned to device CUDA0
# ...
```

**CPU版の場合**:
```
load_tensors: layer   0 assigned to device CPU, is_swa = 0
load_tensors: layer   1 assigned to device CPU, is_swa = 0
```

### 確認2: nvidia-smiでVRAM使用量確認

```powershell
# モデルロード前
nvidia-smi --query-gpu=memory.used --format=csv

# モデルロード後（新しいターミナルで）
nvidia-smi --query-gpu=memory.used --format=csv
```

**期待される変化**:
- ロード前: 数百MB
- ロード後: 4-7GB（モデルサイズに応じて）

### 確認3: 比較テストで速度確認

```powershell
# GPU対応版
python compare_llava_quantizations.py
# 期待される時間: 各モデル15-30秒、合計1-2分

# CPU版
# 期待される時間: 各モデル3-5分、合計10-15分
```

---

## 推奨アクション

### 今すぐテストしたい場合

**CPU版のまま実行**:
```powershell
python compare_llava_quantizations.py
```
- 所要時間: 10-15分
- 結果は正確（速度のみ違い）

### GPU対応したい場合（優先順位）

1. **方法2（プリビルド版）を試す** - 5分
   - jllllll氏のCUDA 12.4版
   - 失敗したら方法3へ

2. **方法3（CPU版使用）を選択** - 0分（現状維持）
   - テスト実行して結果を得る
   - GPU対応は後日実施

3. **方法1（完全ビルド）を実施** - 1-2時間
   - Visual Studio + CUDA Toolkit設定
   - 最も確実だが時間がかかる

---

## compare_llava_quantizations.pyのGPU対応

GPU対応版がインストールできたら、以下を変更:

```python
# 現在（CPU版）
model = Llama(
    model_path=config.model_path,
    chat_handler=chat_handler,
    n_gpu_layers=0,  # CPU版
    n_ctx=4096,
    verbose=False,
    logits_all=True,
)

# GPU対応後
model = Llama(
    model_path=config.model_path,
    chat_handler=chat_handler,
    n_gpu_layers=-1,  # 全レイヤーをGPU
    n_ctx=4096,
    verbose=False,
    logits_all=True,
)
```

---

## 参考リソース

### 公式ドキュメント

- **llama-cpp-python**: https://github.com/abetlen/llama-cpp-python
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads
- **Visual Studio**: https://visualstudio.microsoft.com/downloads/

### プリビルド版

- **jllllll氏**: https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels
- **abetlen公式**: https://github.com/abetlen/llama-cpp-python/releases

### トラブルシューティング

- **Issue: CUDA支援**: https://github.com/abetlen/llama-cpp-python/issues
- **Windows環境**: https://github.com/abetlen/llama-cpp-python/discussions

---

## 次のステップ

### CPU版でテスト完了後

1. ✅ 3モデル比較結果を取得
2. ✅ v0.2完成（LLaVA量子化比較）
3. ⏳ GPU対応を改善（時間のある時）

### GPU対応完了後

1. ✅ compare_llava_quantizations.pyを`n_gpu_layers=-1`に変更
2. ✅ 再実行（所要時間1-2分に短縮）
3. ✅ README更新（GPU推論速度記載）

---

**最終更新**: 2026年3月20日  
**ステータス**: CPU版動作中、GPU対応は要設定
