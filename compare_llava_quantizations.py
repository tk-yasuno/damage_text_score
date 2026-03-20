"""
LLaVA 1.5 7B 量子化比較スクリプト (v0.2)
Q4_K_M / Q5_K_M / Q8_0 の精度・速度・VRAM使用量を比較
"""
import torch
import time
import csv
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from io import BytesIO
import psutil
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np

# llama-cpp-pythonのログ抑制
os.environ['LLAMA_CPP_LOG_LEVEL'] = '0'

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


@dataclass
class QuantizationConfig:
    """量子化設定"""
    name: str
    model_path: str
    mmproj_path: str
    expected_size_gb: float


# 3つの量子化レベル
QUANTIZATIONS = [
    QuantizationConfig(
        name="Q4_K_M",
        model_path="models/ggml-model-q4_k.gguf",
        mmproj_path="models/mmproj-model-f16.gguf",
        expected_size_gb=4.1
    ),
    QuantizationConfig(
        name="Q5_K_M",
        model_path="models/llava-v1.5-7b-Q5_K_M.gguf",
        mmproj_path="models/mmproj-model-f16.gguf",
        expected_size_gb=4.8
    ),
    QuantizationConfig(
        name="Q8_0",
        model_path="models/llava-v1.5-7b-Q8_0.gguf",
        mmproj_path="models/mmproj-model-f16.gguf",
        expected_size_gb=7.2
    )
]


# テスト用画像リスト（全254枚）
import glob
TEST_IMAGES = sorted(glob.glob("data/images_human_inspect_n254/*.png"))

DEFAULT_PROMPT = """You are a civil engineering expert specializing in bridge inspection.
Describe the structural damage visible in this image using technical terminology.

Focus on the following:
- Damage type (crack, rebar exposure, corrosion, spalling, section loss)
- Severity level (minor, moderate, severe)
- Location and extent

Be precise and avoid speculation. Use Japanese for the description."""


def evaluate_description_quality(text: str) -> Dict:
    """損傷記述の品質を評価（v0.2設計に基づく）
    
    評価軸:
    - 損傷タイプの言及数 (0-2点)
    - 重症度の言及有無 (0-1点)
    - 場所情報の言及有無 (0-1点)
    - 範囲情報の言及有無 (0-1点)
    - 合計5点満点
    """
    text_lower = text.lower()
    score_breakdown = {}
    
    # 損傷タイプ（最大2点）
    damage_types = ['crack', 'rebar', '鉄筋', 'exposure', '露出', 'corrosion', '腐食', 
                    'spalling', '剥離', 'section loss', '断面欠損', '割れ', '裂', 
                    'damage', '損傷', '破損']
    damage_count = sum(1 for keyword in damage_types if keyword in text_lower)
    damage_score = min(damage_count / 2, 2.0)  # 4つ以上で満点
    score_breakdown['damage_types'] = damage_score
    
    # 重症度（最大1点）
    severity_keywords = ['minor', 'moderate', 'severe', '軽度', '中度', '重度', 
                         '軽微', '深刻', '著しい', 'slight', 'significant']
    severity_score = 1.0 if any(keyword in text_lower for keyword in severity_keywords) else 0.0
    score_breakdown['severity'] = severity_score
    
    # 場所（最大1点）
    location_keywords = ['location', '場所', '位置', '箇所', 'area', '部分', 'section', 
                        '上部', '下部', '左', '右', '中央', 'top', 'bottom', 'left', 'right']
    location_score = 1.0 if any(keyword in text_lower for keyword in location_keywords) else 0.0
    score_breakdown['location'] = location_score
    
    # 範囲（最大1点）
    extent_keywords = ['extent', '範囲', '広', 'wide', 'narrow', '局所', 'local', 
                      'widespread', '全体', '一部', 'partial', 'entire']
    extent_score = 1.0 if any(keyword in text_lower for keyword in extent_keywords) else 0.0
    score_breakdown['extent'] = extent_score
    
    # 合計スコア（5点満点）
    total_score = sum(score_breakdown.values())
    
    # テキスト長（詳細度の指標）
    text_length = len(text)
    
    return {
        'total_score': total_score,
        'score_breakdown': score_breakdown,
        'text_length': text_length,
        'max_score': 5.0
    }


def get_vram_usage_mb():
    """GPUのVRAM使用量を取得（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def image_to_data_url(image_path: str) -> str:
    """画像をdata URLに変換（Base64エンコード）"""
    pil_image = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def test_quantization_single(config: QuantizationConfig, image_path: str, model, show_result: bool = False) -> Dict:
    """1枚の画像で推論テスト（モデルは既にロード済み）"""
    
    try:
        # 推論開始
        inference_start = time.time()
        
        # 画像をdata URLに変換
        image_url = image_to_data_url(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": DEFAULT_PROMPT}
                ]
            }
        ]
        
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.3,
        )
        
        inference_time = time.time() - inference_start
        result_text = response['choices'][0]['message']['content']
        
        # 精度評価（v0.2設計）
        quality_eval = evaluate_description_quality(result_text)
        
        if show_result:
            print(f"  {Path(image_path).name}: {inference_time:.2f}s (品質: {quality_eval['total_score']:.1f}/5.0)")
        
        return {
            "image": Path(image_path).name,
            "inference_time_sec": inference_time,
            "result": result_text,
            "quality_score": quality_eval['total_score'],
            "quality_breakdown": quality_eval['score_breakdown'],
            "text_length": quality_eval['text_length'],
            "success": True
        }
        
    except Exception as e:
        print(f"  ✗ エラー ({Path(image_path).name}): {str(e)}")
        return {
            "image": Path(image_path).name,
            "success": False,
            "error": str(e)
        }


def test_quantization(config: QuantizationConfig, image_paths: List[str]) -> Dict:
    """1つの量子化レベルで複数画像をテスト"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {config.name}")
    print(f"{'='*70}")
    
    # モデル存在確認
    if not Path(config.model_path).exists():
        print(f"✗ モデルが見つかりません: {config.model_path}")
        return None
    
    # 初期化開始時間
    init_start = time.time()
    
    try:
        # Chat handlerの設定
        chat_handler = Llava15ChatHandler(
            clip_model_path=config.mmproj_path,
            verbose=False
        )
        
        # モデルの読み込み
        model = Llama(
            model_path=config.model_path,
            chat_handler=chat_handler,
            n_gpu_layers=-1,  # 全レイヤーをGPUに割り当て
            n_ctx=4096,
            verbose=False,
            logits_all=True,
        )
        
        init_time = time.time() - init_start
        
        # VRAM使用量（初期化後）
        vram_after_load = get_vram_usage_mb()
        
        print(f"✓ モデル読み込み完了 ({init_time:.1f}秒)")
        print(f"  VRAM使用量: {vram_after_load:.1f} MB")
        
        # 各画像で推論テスト
        print(f"\n推論中（{len(image_paths)}枚）...")
        inference_results = []
        
        for idx, image_path in enumerate(image_paths, 1):
            result = test_quantization_single(config, image_path, model, show_result=True)
            if result['success']:
                inference_results.append(result)
        
        # 統計計算
        inference_times = [r['inference_time_sec'] for r in inference_results if r['success']]
        avg_inference = np.mean(inference_times) if inference_times else 0
        std_inference = np.std(inference_times) if inference_times else 0
        total_inference = sum(inference_times)
        
        # 精度スコア統計
        quality_scores = [r['quality_score'] for r in inference_results if r['success']]
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        std_quality = np.std(quality_scores) if quality_scores else 0
        
        # テキスト長統計
        text_lengths = [r['text_length'] for r in inference_results if r['success']]
        avg_text_length = np.mean(text_lengths) if text_lengths else 0
        
        print(f"\n✓ 推論完了 ({len(inference_results)}/{len(image_paths)}枚成功)")
        print(f"  平均推論時間: {avg_inference:.2f}s (±{std_inference:.2f}s)")
        print(f"  合計推論時間: {total_inference:.1f}s")
        print(f"  平均品質スコア: {avg_quality:.2f}/5.0 (±{std_quality:.2f})")
        print(f"  平均テキスト長: {avg_text_length:.0f}文字")
        
        # VRAM使用量（推論後）
        vram_after_inference = get_vram_usage_mb()
        
        return {
            "quantization": config.name,
            "model_size_gb": config.expected_size_gb,
            "init_time_sec": init_time,
            "avg_inference_time_sec": avg_inference,
            "std_inference_time_sec": std_inference,
            "total_inference_time_sec": total_inference,
            "avg_quality_score": avg_quality,
            "std_quality_score": std_quality,
            "avg_text_length": avg_text_length,
            "num_images": len(image_paths),
            "num_success": len(inference_results),
            "total_time_sec": init_time + total_inference,
            "vram_after_load_mb": vram_after_load,
            "vram_after_inference_mb": vram_after_inference,
            "inference_times": inference_times,
            "results": inference_results,
            "success": True
        }
        
    except Exception as e:
        print(f"✗ エラー発生: {str(e)}")
        return {
            "quantization": config.name,
            "success": False,
            "error": str(e)
        }


def plot_comparison(results: List[Dict], output_file: str = "llava_quantization_comparison.png"):
    """結果を可視化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LLaVA 1.5 7B 量子化比較結果', fontsize=16, fontweight='bold')
    
    quantizations = [r['quantization'] for r in results if r.get('success')]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # 1. 平均推論時間の比較
    ax1 = axes[0, 0]
    avg_times = [r['avg_inference_time_sec'] for r in results if r.get('success')]
    std_times = [r['std_inference_time_sec'] for r in results if r.get('success')]
    bars = ax1.bar(quantizations, avg_times, color=colors, alpha=0.7, yerr=std_times, capsize=5)
    ax1.set_ylabel('平均推論時間 (秒)', fontsize=11)
    ax1.set_title('平均推論時間比較', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 値をバーの上に表示
    for bar, val, std in zip(bars, avg_times, std_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{val:.2f}s\\n±{std:.2f}s',
                ha='center', va='bottom', fontsize=9)
    
    # 2. モデルサイズ vs 合計時間
    ax2 = axes[0, 1]
    sizes = [r['model_size_gb'] for r in results if r.get('success')]
    total_times = [r['total_time_sec'] for r in results if r.get('success')]
    scatter = ax2.scatter(sizes, total_times, c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, q in enumerate(quantizations):
        ax2.annotate(q, (sizes[i], total_times[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    ax2.set_xlabel('モデルサイズ (GB)', fontsize=11)
    ax2.set_ylabel('合計時間 (秒)', fontsize=11)
    ax2.set_title('モデルサイズ vs 合計時間', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 精度スコア比較（v0.2設計）
    ax3 = axes[1, 0]
    quality_scores = [r['avg_quality_score'] for r in results if r.get('success')]
    std_qualities = [r['std_quality_score'] for r in results if r.get('success')]
    bars = ax3.bar(quantizations, quality_scores, color=colors, alpha=0.7, yerr=std_qualities, capsize=5)
    ax3.set_ylabel('品質スコア (5点満点)', fontsize=11)
    ax3.set_title('損傷記述の品質比較', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 5.5)
    ax3.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label='基準値(3.0)')
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)
    
    # 値をバーの上に表示
    for bar, val, std in zip(bars, quality_scores, std_qualities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{val:.2f}\\n±{std:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 4. 総合比較表（テキスト）
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [['量子化', 'サイズ', '平均推論', '品質', '成功率']]
    for r in results:
        if r.get('success'):
            table_data.append([
                r['quantization'],
                f"{r['model_size_gb']:.1f}GB",
                f"{r['avg_inference_time_sec']:.2f}s",
                f"{r['avg_quality_score']:.2f}/5",
                f"{r['num_success']}/{r['num_images']}"
            ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # ヘッダー行のスタイル
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # データ行のスタイル（交互に色を変える）
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            table[(i, j)].set_text_props(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ グラフを保存: {output_file}")
    
    return output_file


def plot_comparison_en(results: List[Dict], output_file: str = "llava_quantization_comparison_EN.png"):
    """Visualize results (English version)"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LLaVA 1.5 7B Quantization Comparison Results', fontsize=16, fontweight='bold')
    
    quantizations = [r['quantization'] for r in results if r.get('success')]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # 1. Average Inference Time Comparison
    ax1 = axes[0, 0]
    avg_times = [r['avg_inference_time_sec'] for r in results if r.get('success')]
    std_times = [r['std_inference_time_sec'] for r in results if r.get('success')]
    bars = ax1.bar(quantizations, avg_times, color=colors, alpha=0.7, yerr=std_times, capsize=5)
    ax1.set_ylabel('Average Inference Time (sec)', fontsize=11)
    ax1.set_title('Average Inference Time Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Display values above bars
    for bar, val, std in zip(bars, avg_times, std_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{val:.2f}s\n±{std:.2f}s',
                ha='center', va='bottom', fontsize=9)
    
    # 2. Model Size vs Total Time
    ax2 = axes[0, 1]
    sizes = [r['model_size_gb'] for r in results if r.get('success')]
    total_times = [r['total_time_sec'] for r in results if r.get('success')]
    scatter = ax2.scatter(sizes, total_times, c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, q in enumerate(quantizations):
        ax2.annotate(q, (sizes[i], total_times[i]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    ax2.set_xlabel('Model Size (GB)', fontsize=11)
    ax2.set_ylabel('Total Time (sec)', fontsize=11)
    ax2.set_title('Model Size vs Total Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Quality Score Comparison (v0.2 design)
    ax3 = axes[1, 0]
    quality_scores = [r['avg_quality_score'] for r in results if r.get('success')]
    std_qualities = [r['std_quality_score'] for r in results if r.get('success')]
    bars = ax3.bar(quantizations, quality_scores, color=colors, alpha=0.7, yerr=std_qualities, capsize=5)
    ax3.set_ylabel('Quality Score (out of 5)', fontsize=11)
    ax3.set_title('Damage Description Quality Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 5.5)
    ax3.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (3.0)')
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)
    
    # Display values above bars
    for bar, val, std in zip(bars, quality_scores, std_qualities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{val:.2f}\n±{std:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 4. Summary Comparison Table (Text)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [['Quantization', 'Size', 'Avg Inference', 'Quality', 'Success Rate']]
    for r in results:
        if r.get('success'):
            table_data.append([
                r['quantization'],
                f"{r['model_size_gb']:.1f}GB",
                f"{r['avg_inference_time_sec']:.2f}s",
                f"{r['avg_quality_score']:.2f}/5",
                f"{r['num_success']}/{r['num_images']}"
            ])
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header row styling
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Data row styling (alternate colors)
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            table[(i, j)].set_text_props(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ English plot saved: {output_file}")
    
    return output_file


def main():
    """メイン実行"""
    
    print("=" * 70)
    print("LLaVA 1.5 7B 量子化比較テスト")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "")
    print()
    
    # テスト画像の確認
    available_images = [img for img in TEST_IMAGES if Path(img).exists()]
    
    if not available_images:
        print("✗ テスト画像が見つかりません")
        return
    
    print(f"テスト画像: {len(available_images)}枚")
    print()
    
    # 各量子化レベルでテスト
    results = []
    for config in QUANTIZATIONS:
        result = test_quantization(config, available_images)
        if result:
            results.append(result)
        
        # GPUメモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        time.sleep(2)  # クールダウン
    
    # 結果比較表示
    print("\n" + "=" * 70)
    print("比較結果")
    print("=" * 70)
    
    if not results:
        print("✗ 有効な結果がありません")
        return
    
    print("\n【パフォーマンス比較】")
    print(f"{'量子化':<10} {'サイズ':>8} {'初期化':>10} {'平均推論':>10} {'品質':>10} {'成功率':>8}")
    print("-" * 72)
    
    for r in results:
        if r.get("success"):
            print(f"{r['quantization']:<10} "
                  f"{r['model_size_gb']:>7.1f}GB "
                  f"{r['init_time_sec']:>9.1f}s "
                  f"{r['avg_inference_time_sec']:>9.2f}s "
                  f"{r['avg_quality_score']:>7.2f}/5 "
                  f"{r['num_success']:>3}/{r['num_images']:<3}")
    
    # CSV出力（サマリー）
    output_file = "llava_quantization_comparison_summary.csv"
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'quantization', 'model_size_gb', 'init_time_sec', 'avg_inference_time_sec',
            'std_inference_time_sec', 'total_inference_time_sec', 'total_time_sec',
            'avg_quality_score', 'std_quality_score', 'avg_text_length',
            'num_images', 'num_success', 'vram_after_load_mb', 'vram_after_inference_mb'
        ])
        writer.writeheader()
        for r in results:
            if r.get("success"):
                row = {k: v for k, v in r.items() if k in writer.fieldnames}
                writer.writerow(row)
    
    print(f"\n✓ サマリー結果を保存: {output_file}")
    
    # CSV出力（詳細：画像ごと）
    detail_file = "llava_quantization_comparison_detail.csv"
    with open(detail_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['quantization', 'image', 'inference_time_sec', 'quality_score', 'text_length', 'result_preview'])
        for r in results:
            if r.get("success") and 'results' in r:
                for img_result in r['results']:
                    if img_result['success']:
                        preview = img_result['result'][:100].replace('\n', ' ')
                        writer.writerow([
                            r['quantization'],
                            img_result['image'],
                            f"{img_result['inference_time_sec']:.3f}",
                            f"{img_result['quality_score']:.2f}",
                            img_result['text_length'],
                            preview
                        ])
    
    print(f"✓ 詳細結果を保存: {detail_file}")
    
    # グラフ作成（日本語版）
    plot_comparison(results)
    
    # グラフ作成（英語版）
    plot_comparison_en(results)
   
    # 推奨事項
    print("\n【推奨事項】")
    
    # 成功したテストのみ抽出
    successful_results = [r for r in results if r.get("success")]
    
    q4 = next((r for r in successful_results if r['quantization'] == 'Q4_K_M'), None)
    q5 = next((r for r in successful_results if r['quantization'] == 'Q5_K_M'), None)
    q8 = next((r for r in successful_results if r['quantization'] == 'Q8_0'), None)
    
    if q4 and q5:
        speed_diff = ((q5['avg_inference_time_sec'] - q4['avg_inference_time_sec']) / q4['avg_inference_time_sec']) * 100
        size_diff = ((q5['model_size_gb'] - q4['model_size_gb']) / q4['model_size_gb']) * 100
        quality_diff = ((q5['avg_quality_score'] - q4['avg_quality_score']) / q4['avg_quality_score']) * 100
        print(f"Q5_K_M vs Q4_K_M: +{size_diff:.1f}% サイズ, {speed_diff:+.1f}% 速度, {quality_diff:+.1f}% 品質")
    
    if q8 and q5:
        speed_diff = ((q8['avg_inference_time_sec'] - q5['avg_inference_time_sec']) / q5['avg_inference_time_sec']) * 100
        size_diff = ((q8['model_size_gb'] - q5['model_size_gb']) / q5['model_size_gb']) * 100
        quality_diff = ((q8['avg_quality_score'] - q5['avg_quality_score']) / q5['avg_quality_score']) * 100
        print(f"Q8_0 vs Q5_K_M: +{size_diff:.1f}% サイズ, {speed_diff:+.1f}% 速度, {quality_diff:+.1f}% 品質")
    
    print("\n✓ テスト完了")


if __name__ == "__main__":
    main()
