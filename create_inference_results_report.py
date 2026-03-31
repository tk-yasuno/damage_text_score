"""
Generate plots and markdown report for n=10789 inference results.

Outputs:
- data/outputs/reports/inference_n10789_damage_type_distribution.png
- data/outputs/reports/inference_n10789_severity_distribution.png
- data/outputs/reports/inference_n10789_description_length_distribution.png
- data/outputs/reports/inference_n10789_score_histogram.png
- Results_inference_n10789.md
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def _normalize_series(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip()
    normalized = normalized.replace("", "(blank)")
    return normalized


def _find_latest_merged_csv(outputs_dir: Path) -> Path:
    candidates = sorted(outputs_dir.glob("inference_n10789_merged_final_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No merged final CSV found in data/outputs")
    return candidates[-1]


def _make_bar_plot(counts: pd.Series, title: str, xlabel: str, ylabel: str, out_path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index.astype(str), counts.values, color=color, alpha=0.9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    total = int(counts.sum())
    for bar, value in zip(bars, counts.values):
        ratio = (value / total) * 100 if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value} ({ratio:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _make_description_length_plot(lengths: pd.Series, out_path: Path) -> dict:
    valid = lengths.dropna().astype(int)
    stats = {
        "count": int(valid.count()),
        "mean": float(valid.mean()) if len(valid) else 0.0,
        "median": float(valid.median()) if len(valid) else 0.0,
        "std": float(valid.std()) if len(valid) else 0.0,
        "p90": float(valid.quantile(0.90)) if len(valid) else 0.0,
        "max": int(valid.max()) if len(valid) else 0,
        "min": int(valid.min()) if len(valid) else 0,
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid, bins=40, color="#2ca02c", alpha=0.85, edgecolor="white")
    ax.axvline(stats["mean"], color="#d62728", linestyle="--", linewidth=2, label=f"Mean: {stats['mean']:.1f}")
    ax.axvline(stats["median"], color="#1f77b4", linestyle="-.", linewidth=2, label=f"Median: {stats['median']:.1f}")

    ax.set_title("Damage Description Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Description Length (characters)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return stats


def _make_score_histogram(scores: pd.Series, out_path: Path) -> dict:
    valid = pd.to_numeric(scores, errors="coerce").dropna()
    stats = {
        "count": int(valid.count()),
        "mean": float(valid.mean()) if len(valid) else 0.0,
        "median": float(valid.median()) if len(valid) else 0.0,
        "std": float(valid.std()) if len(valid) else 0.0,
        "p90": float(valid.quantile(0.90)) if len(valid) else 0.0,
        "max": float(valid.max()) if len(valid) else 0.0,
        "min": float(valid.min()) if len(valid) else 0.0,
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid, bins=30, color="#9467bd", alpha=0.85, edgecolor="white")
    ax.axvline(stats["mean"], color="#d62728", linestyle="--", linewidth=2, label=f"Mean: {stats['mean']:.3f}")
    ax.axvline(stats["median"], color="#1f77b4", linestyle="-.", linewidth=2, label=f"Median: {stats['median']:.3f}")

    ax.set_title("Raw Score Histogram", fontsize=14, fontweight="bold")
    ax.set_xlabel("Raw Score")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return stats


def generate_report(csv_path: Path, project_root: Path) -> Path:
    reports_dir = project_root / "data" / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    damage_col = "損傷種別"
    severity_col = "重症度"
    description_col = "損傷説明"
    score_col = "生スコア"

    for required in [damage_col, severity_col, description_col, score_col]:
        if required not in df.columns:
            raise KeyError(f"Required column not found: {required}")

    total_records = len(df)

    damage_counts = _normalize_series(df[damage_col]).value_counts().sort_values(ascending=False)
    severity_counts = _normalize_series(df[severity_col]).value_counts().sort_values(ascending=False)
    description_lengths = df[description_col].fillna("").astype(str).str.len()
    raw_scores = df[score_col]

    damage_plot = reports_dir / "inference_n10789_damage_type_distribution.png"
    severity_plot = reports_dir / "inference_n10789_severity_distribution.png"
    length_plot = reports_dir / "inference_n10789_description_length_distribution.png"
    score_plot = reports_dir / "inference_n10789_score_histogram.png"

    _make_bar_plot(
        damage_counts,
        title="Damage Type Distribution",
        xlabel="Damage Type",
        ylabel="Count",
        out_path=damage_plot,
        color="#1f77b4",
    )

    _make_bar_plot(
        severity_counts,
        title="Severity Distribution",
        xlabel="Severity",
        ylabel="Count",
        out_path=severity_plot,
        color="#ff7f0e",
    )

    length_stats = _make_description_length_plot(description_lengths, length_plot)
    score_stats = _make_score_histogram(raw_scores, score_plot)

    damage_md = "\n".join(
        [f"- {k}: {v} ({(v/total_records)*100:.2f}%)" for k, v in damage_counts.items()]
    )
    severity_md = "\n".join(
        [f"- {k}: {v} ({(v/total_records)*100:.2f}%)" for k, v in severity_counts.items()]
    )

    report_md = project_root / "Results_inference_n10789.md"

    markdown = f"""# Results: Inference n=10789

## Source
- Merged inference CSV: {csv_path.as_posix()}
- Total records: {total_records}

## 1. Damage Type Distribution

![Damage Type Distribution](data/outputs/reports/{damage_plot.name})

{damage_md}

## 2. Severity Distribution

![Severity Distribution](data/outputs/reports/{severity_plot.name})

{severity_md}

## 3. Damage Description Length

![Damage Description Length Distribution](data/outputs/reports/{length_plot.name})

- Count: {length_stats['count']}
- Mean length: {length_stats['mean']:.2f} chars
- Median length: {length_stats['median']:.2f} chars
- Std: {length_stats['std']:.2f}
- 90th percentile: {length_stats['p90']:.2f} chars
- Min: {length_stats['min']} chars
- Max: {length_stats['max']} chars

## 4. Raw Score Histogram

![Raw Score Histogram](data/outputs/reports/{score_plot.name})

- Count: {score_stats['count']}
- Mean score: {score_stats['mean']:.4f}
- Median score: {score_stats['median']:.4f}
- Std: {score_stats['std']:.4f}
- 90th percentile: {score_stats['p90']:.4f}
- Min: {score_stats['min']:.4f}
- Max: {score_stats['max']:.4f}

## Notes
- This report summarizes base-model inference outputs after merged deduplication.
- Description length is computed from the `損傷説明` column.
"""

    report_md.write_text(markdown, encoding="utf-8")
    return report_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots and markdown for n10789 inference results")
    parser.add_argument("--csv", type=str, default="", help="Path to merged final CSV. If omitted, latest merged CSV is used.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    outputs_dir = project_root / "data" / "outputs"

    csv_path = Path(args.csv) if args.csv else _find_latest_merged_csv(outputs_dir)

    report_path = generate_report(csv_path, project_root)
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
