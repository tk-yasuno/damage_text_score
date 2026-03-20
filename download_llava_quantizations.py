"""
LLaVA 1.5 7B 量子化版ダウンロードスクリプト (v0.2)
Q4_K_M / Q5_K_M / Q8_0 の3つをダウンロード
"""
import os
from huggingface_hub import hf_hub_download

# hf_transfer有効化（高速ダウンロード）
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

def download_llava_quantizations():
    """LLaVA 1.5 7B の量子化版をダウンロード"""
    
    # Q4_K_Mは既存リポジトリから、Q5_K_M/Q8_0はsecond-stateから
    downloads = [
        {
            "repo_id": "mys/ggml_llava-v1.5-7b",
            "filename": "ggml-model-q4_k.gguf",
            "local_name": "llava-1.5-7b-Q4_K_M.gguf",
            "size": "4.1GB",
            "skip": True  # 既存
        },
        {
            "repo_id": "second-state/Llava-v1.5-7B-GGUF",
            "filename": "llava-v1.5-7b-Q5_K_M.gguf",
            "local_name": "llava-1.5-7b-Q5_K_M.gguf",
            "size": "4.8GB",
            "skip": False
        },
        {
            "repo_id": "second-state/Llava-v1.5-7B-GGUF",
            "filename": "llava-v1.5-7b-Q8_0.gguf",
            "local_name": "llava-1.5-7b-Q8_0.gguf",
            "size": "7.2GB",
            "skip": False
        },
        {
            "repo_id": "second-state/Llava-v1.5-7B-GGUF",
            "filename": "llava-v1.5-7b-mmproj-model-f16.gguf",
            "local_name": "mmproj-model-f16.gguf",
            "size": "624MB",
            "skip": True  # 既存
        }
    ]
    
    local_dir = "models"
    
    print("=" * 70)
    print("LLaVA 1.5 7B 量子化版ダウンロード")
    print("=" * 70)
    print(f"Local Dir: {local_dir}")
    print()
    
    # ダウンロード
    for item in downloads:
        print()
        print("-" * 70)
        print(f"Model: {item['local_name']}")
        print(f"Repo: {item['repo_id']}")
        print(f"Size: {item['size']}")
        print("-" * 70)
        
        if item["skip"]:
            print(f"✓ スキップ（既存）")
            continue
        
        try:
            print("ダウンロード中...")
            downloaded_path = hf_hub_download(
                repo_id=item["repo_id"],
                filename=item["filename"],
                local_dir=local_dir,
                resume_download=True
            )
            print(f"✓ ダウンロード完了: {item['local_name']}")
            
        except Exception as e:
            print(f"✗ ダウンロード失敗: {str(e)}")
            raise
    
    print()
    print("=" * 70)
    print("✓ 全ダウンロード完了！")
    print("=" * 70)
    print()
    print("ダウンロードされたモデル:")
    print("  - Q4_K_M: llava-1.5-7b-Q4_K_M.gguf (既存)")
    print("  - Q5_K_M: llava-1.5-7b-Q5_K_M.gguf")
    print("  - Q8_0: llava-1.5-7b-Q8_0.gguf")
    print("  - MM-Proj: mmproj-model-f16.gguf (既存)")
    print()
    print("次のコマンドで比較テスト:")
    print("  python compare_llava_quantizations.py")


if __name__ == "__main__":
    download_llava_quantizations()
