"""
Statistical Visualization for LLaVA Quantization Comparison
Creates violin plots for quality scores and text lengths
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

# Set English font
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

def create_statistical_plots():
    """Create comprehensive statistical visualizations"""
    
    # Load detail data
    df = pd.read_csv('llava_quantization_comparison_detail.csv')
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color palette
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    quantizations = ['Q4_K_M', 'Q5_K_M', 'Q8_0']
    
    # ===== 1. Quality Score Violin Plot =====
    ax1 = fig.add_subplot(gs[0, 0])
    parts = ax1.violinplot(
        [df[df['quantization'] == q]['quality_score'].values for q in quantizations],
        positions=[0, 1, 2],
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # Color the violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(quantizations)
    ax1.set_ylabel('Quality Score (0-5)', fontsize=11)
    ax1.set_title('Quality Score Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.5, 5.5)
    ax1.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (3.0)')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # ===== 2. Text Length Violin Plot =====
    ax2 = fig.add_subplot(gs[0, 1])
    parts2 = ax2.violinplot(
        [df[df['quantization'] == q]['text_length'].values for q in quantizations],
        positions=[0, 1, 2],
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # Color the violin plots
    for i, pc in enumerate(parts2['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(quantizations)
    ax2.set_ylabel('Text Length (characters)', fontsize=11)
    ax2.set_title('Text Length Distribution', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # ===== 3. Inference Time Violin Plot =====
    ax3 = fig.add_subplot(gs[0, 2])
    parts3 = ax3.violinplot(
        [df[df['quantization'] == q]['inference_time_sec'].values for q in quantizations],
        positions=[0, 1, 2],
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # Color the violin plots
    for i, pc in enumerate(parts3['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(quantizations)
    ax3.set_ylabel('Inference Time (seconds)', fontsize=11)
    ax3.set_title('Inference Time Distribution', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # ===== 4. Quality Score Box Plot with Stats =====
    ax4 = fig.add_subplot(gs[1, 0])
    bp = ax4.boxplot(
        [df[df['quantization'] == q]['quality_score'].values for q in quantizations],
        labels=quantizations,
        patch_artist=True,
        notch=True,
        widths=0.6
    )
    
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Quality Score (0-5)', fontsize=11)
    ax4.set_title('Quality Score Box Plot', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add mean values
    for i, q in enumerate(quantizations):
        mean_val = df[df['quantization'] == q]['quality_score'].mean()
        ax4.text(i+1, -0.3, f'μ={mean_val:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    # ===== 5. Quality vs Text Length Scatter =====
    ax5 = fig.add_subplot(gs[1, 1])
    for i, q in enumerate(quantizations):
        data = df[df['quantization'] == q]
        ax5.scatter(data['text_length'], data['quality_score'], 
                   c=colors[i], label=q, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    ax5.set_xlabel('Text Length (characters)', fontsize=11)
    ax5.set_ylabel('Quality Score (0-5)', fontsize=11)
    ax5.set_title('Quality Score vs Text Length', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Add correlation info
    for i, q in enumerate(quantizations):
        data = df[df['quantization'] == q]
        corr = np.corrcoef(data['text_length'], data['quality_score'])[0, 1]
        print(f"{q} - Correlation (Text Length vs Quality): {corr:.3f}")
    
    # ===== 6. Inference Time vs Quality Score =====
    ax6 = fig.add_subplot(gs[1, 2])
    for i, q in enumerate(quantizations):
        data = df[df['quantization'] == q]
        ax6.scatter(data['inference_time_sec'], data['quality_score'], 
                   c=colors[i], label=q, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    ax6.set_xlabel('Inference Time (seconds)', fontsize=11)
    ax6.set_ylabel('Quality Score (0-5)', fontsize=11)
    ax6.set_title('Quality Score vs Inference Time', fontsize=12, fontweight='bold')
    ax6.legend(loc='best', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # ===== 7. Statistical Summary Table =====
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Calculate statistics
    stats_data = []
    for q in quantizations:
        data = df[df['quantization'] == q]
        stats_data.append([
            q,
            f"{data['quality_score'].mean():.2f} ± {data['quality_score'].std():.2f}",
            f"{data['quality_score'].median():.2f}",
            f"{data['text_length'].mean():.1f} ± {data['text_length'].std():.1f}",
            f"{data['text_length'].median():.0f}",
            f"{data['inference_time_sec'].mean():.2f} ± {data['inference_time_sec'].std():.2f}",
            f"{data['inference_time_sec'].median():.2f}",
            f"{len(data)}"
        ])
    
    table_data = [
        ['Quantization', 'Quality\n(Mean ± SD)', 'Quality\n(Median)', 
         'Text Length\n(Mean ± SD)', 'Text Length\n(Median)',
         'Inference Time\n(Mean ± SD)', 'Inference Time\n(Median)', 'N']
    ] + stats_data
    
    table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.12, 0.15, 0.12, 0.15, 0.12, 0.15, 0.12, 0.07])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white', fontsize=10)
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            cell.set_text_props(fontsize=9)
    
    # Add title
    fig.suptitle('LLaVA 1.5 7B Quantization Comparison - Statistical Analysis (N=254 images)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_file = 'llava_quantization_statistical_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Statistical analysis plot saved: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    for q in quantizations:
        data = df[df['quantization'] == q]
        print(f"\n{q}:")
        print(f"  Quality Score:    Mean={data['quality_score'].mean():.2f}, "
              f"Median={data['quality_score'].median():.2f}, "
              f"SD={data['quality_score'].std():.2f}")
        print(f"  Text Length:      Mean={data['text_length'].mean():.1f}, "
              f"Median={data['text_length'].median():.0f}, "
              f"SD={data['text_length'].std():.1f}")
        print(f"  Inference Time:   Mean={data['inference_time_sec'].mean():.2f}s, "
              f"Median={data['inference_time_sec'].median():.2f}s, "
              f"SD={data['inference_time_sec'].std():.2f}s")
        
        # Quality score distribution
        q_dist = data['quality_score'].value_counts(bins=[0, 1, 2, 3, 4, 5]).sort_index()
        print(f"  Quality Distribution: {dict(q_dist)}")
    
    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL COMPARISONS")
    print("="*70)
    
    from scipy import stats
    
    q4_quality = df[df['quantization'] == 'Q4_K_M']['quality_score'].values
    q5_quality = df[df['quantization'] == 'Q5_K_M']['quality_score'].values
    q8_quality = df[df['quantization'] == 'Q8_0']['quality_score'].values
    
    # Mann-Whitney U test (non-parametric)
    stat_q5_q4, p_q5_q4 = stats.mannwhitneyu(q5_quality, q4_quality, alternative='greater')
    stat_q8_q5, p_q8_q5 = stats.mannwhitneyu(q8_quality, q5_quality, alternative='greater')
    stat_q8_q4, p_q8_q4 = stats.mannwhitneyu(q8_quality, q4_quality, alternative='greater')
    
    print(f"\nQuality Score - Mann-Whitney U Test:")
    print(f"  Q5_K_M vs Q4_K_M: U={stat_q5_q4:.1f}, p={p_q5_q4:.4f} {'***' if p_q5_q4 < 0.001 else '**' if p_q5_q4 < 0.01 else '*' if p_q5_q4 < 0.05 else 'ns'}")
    print(f"  Q8_0 vs Q5_K_M:   U={stat_q8_q5:.1f}, p={p_q8_q5:.4f} {'***' if p_q8_q5 < 0.001 else '**' if p_q8_q5 < 0.01 else '*' if p_q8_q5 < 0.05 else 'ns'}")
    print(f"  Q8_0 vs Q4_K_M:   U={stat_q8_q4:.1f}, p={p_q8_q4:.4f} {'***' if p_q8_q4 < 0.001 else '**' if p_q8_q4 < 0.01 else '*' if p_q8_q4 < 0.05 else 'ns'}")
    
    print("\n✓ Analysis complete")


if __name__ == "__main__":
    create_statistical_plots()
