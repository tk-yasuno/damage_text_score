"""
必要なモデルを事前にダウンロードするスクリプト
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

print("=" * 70)
print("必要なモデルのダウンロード")
print("=" * 70)
print("\n注意: モデルのダウンロードには時間がかかります（約14GB）")
print("GPU VRAM 16GBで動作します\n")

print("注意:")
print("  - テキスト構造化: Ollama swallow8b-lora-n4000-v09-q4 を使用（ダウンロード済み）")
print("  - Vision分析: LLaVA-1.5-7B をダウンロード\n")

# LLaVA-1.5-7B のダウンロード
print("\nLLaVA-1.5-7B Vision モデルをダウンロード中...")
print("-" * 70)
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import torch
    
    model_name = "llava-hf/llava-1.5-7b-hf"
    print(f"モデル: {model_name}")
    
    print("  - プロセッサーをダウンロード中...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("  ✓ プロセッサー完了")
    
    print("  - モデルをダウンロード中（約14GB）...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("  ✓ LLaVA-1.5-7B ダウンロード完了")
    
    # メモリ解放
    del model
    del processor
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"  ✗ エラー: {e}")
    print("  → 後でダウンロードされます")

print("\n" + "=" * 70)
print("ダウンロード完了")
print("=" * 70)
print("\n次のコマンドでクイックスタートを実行できます:")
print("  python quickstart.py")
