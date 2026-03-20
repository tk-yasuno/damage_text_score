"""
Create English version of quantization comparison plot from existing CSV results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_english_plot():
    """Generate English version plot from CSV data"""
    
    # Load summary results
    df = pd.read_csv('llava_quantization_comparison_summary.csv')
    
    # Prepare data
    quantizations = df['quantization'].tolist()
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LLaVA 1.5 7B Quantization Comparison Results', fontsize=16, fontweight='bold')
    
    # 1. Average Inference Time Comparison
    ax1 = axes[0, 0]
    avg_times = df['avg_inference_time_sec'].tolist()
    std_times = df['std_inference_time_sec'].tolist()
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
    sizes = df['model_size_gb'].tolist()
    total_times = df['total_time_sec'].tolist()
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
    quality_scores = df['avg_quality_score'].tolist()
    std_qualities = df['std_quality_score'].tolist()
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
    for _, row in df.iterrows():
        table_data.append([
            row['quantization'],
            f"{row['model_size_gb']:.1f}GB",
            f"{row['avg_inference_time_sec']:.2f}s",
            f"{row['avg_quality_score']:.2f}/5",
            f"{int(row['num_success'])}/{int(row['num_images'])}"
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
    output_file = 'llava_quantization_comparison_EN.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ English plot saved: {output_file}")
    
    return output_file


if __name__ == "__main__":
    print("=" * 70)
    print("Creating English Version of Quantization Comparison Plot")
    print("=" * 70)
    print()
    
    create_english_plot()
    
    print("\n✓ Complete!")
