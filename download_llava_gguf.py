# LLaVA GGUF Q4_K_Mモデルをダウンロード
from huggingface_hub import hf_hub_download
from pathlib import Path

print("LLaVA-1.5-7B GGUF Q4_K_Mモデルをダウンロード中...")
print("リポジトリ: mys/ggml_llava-v1.5-7b")
print("ファイル1: ggml-model-q4_k.gguf (Vision Model, 約4GB)")
print("ファイル2: mmproj-model-f16.gguf (Projection Model)")
print()

# Vision Model
print("Vision Modelをダウンロード中...")
model_path = hf_hub_download(
    repo_id="mys/ggml_llava-v1.5-7b",
    filename="ggml-model-q4_k.gguf",
    local_dir="models",
    resume_download=True
)
print(f"OK: {model_path}")

# MMProj (Vision Projection)
print("\nMMProj Modelをダウンロード中...")
mmproj_path = hf_hub_download(
    repo_id="mys/ggml_llava-v1.5-7b",
    filename="mmproj-model-f16.gguf",
    local_dir="models",
    resume_download=True
)
print(f"OK: {mmproj_path}")

print("\n✓ すべてのモデルファイルのダウンロードが完了しました")
