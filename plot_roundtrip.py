import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_roundtrip(csv_path):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pairs = [
        ("roundtrip_bleu", "bleu", "BLEU"),
        ("roundtrip_chrf", "chrf", "chrF"),
    ]

    for ax, (x_col, y_col, label) in zip(axes, pairs):
        x = df[x_col].values
        y = df[y_col].values

        mask = np.isfinite(x) & np.isfinite(y)
        x_clean, y_clean = x[mask], y[mask]

        slope, intercept, r, p, se = stats.linregress(x_clean, y_clean)
        r2 = r ** 2

        x_line = np.linspace(x_clean.min(), x_clean.max(), 200)
        y_line = slope * x_line + intercept

        ax.scatter(x_clean, y_clean, alpha=0.4, edgecolors="none", s=4)
        ax.plot(x_line, y_line, color="red", linewidth=1.5,
                label=f"y = {slope:.3f}x + {intercept:.3e}\n$R^2$ = {r2:.4f}")

        ax.set_xlabel(f"Roundtrip {label} (normalized)")
        ax.set_ylabel(f"{label} (normalized)")
        ax.set_title(f"Roundtrip {label} vs {label}")
        ax.legend(fontsize=9)

    fig.tight_layout()
    out_path = csv_path.rsplit(".", 1)[0] + "_roundtrip_plots.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_roundtrip.py <path/to/data.csv>")
        sys.exit(1)
    plot_roundtrip(sys.argv[1])
