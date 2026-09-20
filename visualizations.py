"""
=============================================================
TASK 3 - Data Visualization
Dataset : Amazon Product Reviews (amazon_reviews.csv)
Tools   : Matplotlib, Seaborn
=============================================================
Generates 12 publication-quality charts saved to ./plots/
Run: python visualizations.py
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "amazon_reviews.csv")
OUT        = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(OUT, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
MAIN_COLOR   = "#3b82d4"
ACCENT       = "#7c5cd8"
DANGER       = "#e5534b"
SUCCESS      = "#2da44e"
MUTED        = "#57606a"
BG           = "#ffffff"
SURFACE      = "#f7f8fa"
BORDER       = "#e5e7eb"

PALETTE5 = [SUCCESS, MAIN_COLOR, ACCENT, "#f59e0b", DANGER]  # 1-star to 5-star
PALETTE3 = [MAIN_COLOR, ACCENT, DANGER]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   "#1f2328",
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "font.family":       "sans-serif",
})

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print(f"  Saved: {name}")

# ── Load & prep ───────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_PATH)
df["reviewTime"]    = pd.to_datetime(df["reviewTime"])
df["year"]          = df["reviewTime"].dt.year
df["month"]         = df["reviewTime"].dt.month
df["year_month"]    = df["reviewTime"].dt.to_period("M")
df["review_length"] = df["reviewText"].fillna("").apply(len)
df["has_vote"]      = (df["total_vote"] > 0).astype(int)
df["word_count"]    = df["reviewText"].fillna("").apply(lambda x: len(x.split()))

df_voted = df[df["total_vote"] > 0].copy()
df_voted["helpful_ratio"] = df_voted["helpful_yes"] / df_voted["total_vote"]

# rating labels
rating_map = {1.0: "1-star", 2.0: "2-star", 3.0: "3-star",
              4.0: "4-star", 5.0: "5-star"}
df["rating_label"] = df["overall"].map(rating_map)

print("Generating charts ...\n")

# =============================================================================
# CHART 01 — Rating Distribution (Bar + annotation)
# =============================================================================
rating_counts = df["overall"].value_counts().sort_index()
pct           = (rating_counts / len(df) * 100).round(1)

fig, ax = plt.subplots(figsize=(8, 4.5))
colors  = [DANGER, "#f59e0b", MUTED, MAIN_COLOR, SUCCESS]
bars    = ax.bar([str(int(r)) for r in rating_counts.index],
                 rating_counts.values, color=colors,
                 edgecolor=BG, linewidth=1.2, width=0.65)
for bar, p in zip(bars, pct.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{p}%", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#1f2328")
ax.set_xlabel("Star Rating")
ax.set_ylabel("Number of Reviews")
ax.set_title("Chart 1 - Rating Distribution")
ax.yaxis.grid(True); ax.set_axisbelow(True)
save("01_rating_distribution.png")

# =============================================================================
# CHART 02 — Pie chart of rating proportions
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 5))
explode = [0.04, 0.04, 0.04, 0.04, 0.04]
wedges, texts, autotexts = ax.pie(
    rating_counts.values,
    labels=[f"{int(r)}-star" for r in rating_counts.index],
    colors=colors, autopct="%1.1f%%",
    startangle=140, explode=explode,
    pctdistance=0.78,
    wedgeprops=dict(edgecolor=BG, linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(8)
ax.set_title("Chart 2 - Review Share by Star Rating")
save("02_rating_pie.png")

# =============================================================================
# CHART 03 — Monthly review volume (line)
# =============================================================================
monthly = df.groupby("year_month").size().reset_index(name="count")
monthly["label"] = monthly["year_month"].astype(str)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.fill_between(range(len(monthly)), monthly["count"],
                alpha=0.15, color=MAIN_COLOR)
ax.plot(range(len(monthly)), monthly["count"],
        color=MAIN_COLOR, linewidth=2, marker="o",
        markersize=4, markerfacecolor=BG, markeredgecolor=MAIN_COLOR)
step = max(1, len(monthly) // 12)
ax.set_xticks(range(0, len(monthly), step))
ax.set_xticklabels(monthly["label"].iloc[::step], rotation=45, ha="right")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Reviews")
ax.set_title("Chart 3 - Monthly Review Volume (2012-2014)")
ax.yaxis.grid(True); ax.set_axisbelow(True)
save("03_monthly_volume.png")

# =============================================================================
# CHART 04 — Yearly review volume (grouped bar by rating)
# =============================================================================
yearly_rating = (df.groupby(["year", "overall"])
                   .size().reset_index(name="count"))
pivot = yearly_rating.pivot(index="year", columns="overall", values="count").fillna(0)

fig, ax = plt.subplots(figsize=(9, 4.5))
x      = np.arange(len(pivot))
width  = 0.15
for i, (col, color) in enumerate(zip(pivot.columns, colors)):
    offset = (i - 2) * width
    ax.bar(x + offset, pivot[col], width, label=f"{int(col)}-star",
           color=color, edgecolor=BG, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(pivot.index.astype(str))
ax.set_xlabel("Year")
ax.set_ylabel("Number of Reviews")
ax.set_title("Chart 4 - Yearly Review Volume by Star Rating")
ax.legend(title="Rating", ncol=5)
ax.yaxis.grid(True); ax.set_axisbelow(True)
save("04_yearly_by_rating.png")

# =============================================================================
# CHART 05 — Box plot: review length by rating
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
order   = [1.0, 2.0, 3.0, 4.0, 5.0]
bp = ax.boxplot(
    [df[df["overall"] == r]["review_length"].clip(upper=1500).values for r in order],
    patch_artist=True, notch=True, showfliers=False,
    medianprops=dict(color="#1f2328", linewidth=2),
    boxprops=dict(linewidth=1),
    whiskerprops=dict(linewidth=1, linestyle="--"),
    capprops=dict(linewidth=1.5),
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
means = [df[df["overall"] == r]["review_length"].clip(upper=1500).mean() for r in order]
ax.plot(range(1, 6), means, "D--", color="#1f2328",
        markersize=6, markerfacecolor=BG, label="Mean", zorder=5)
ax.set_xticklabels([f"{int(r)}-star" for r in order])
ax.set_xlabel("Star Rating")
ax.set_ylabel("Review Length (chars, capped 1500)")
ax.set_title("Chart 5 - Review Length Distribution by Star Rating")
ax.legend(); ax.yaxis.grid(True); ax.set_axisbelow(True)
save("05_length_boxplot.png")

# =============================================================================
# CHART 06 — Avg review length per rating (horizontal bar)
# =============================================================================
avg_len = df.groupby("overall")["review_length"].mean().sort_index(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
y_pos   = range(len(avg_len))
bar_colors = list(reversed(colors))
h_bars  = ax.barh(y_pos, avg_len.values, color=bar_colors,
                  edgecolor=BG, linewidth=1, height=0.55)
for bar, val in zip(h_bars, avg_len.values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}", va="center", fontsize=9, color="#1f2328")
ax.set_yticks(list(y_pos))
ax.set_yticklabels([f"{int(r)}-star" for r in avg_len.index])
ax.set_xlabel("Average Review Length (chars)")
ax.set_title("Chart 6 - Average Review Length by Star Rating")
ax.xaxis.grid(True); ax.set_axisbelow(True)
save("06_avg_length_hbar.png")

# =============================================================================
# CHART 07 — Helpfulness vote distribution (histogram + KDE)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].bar(["No Votes", "Has Votes"],
            [df["has_vote"].value_counts()[0], df["has_vote"].value_counts()[1]],
            color=[MUTED, MAIN_COLOR], edgecolor=BG, linewidth=1.2, width=0.5)
for bar, val in zip(axes[0].patches,
                    [df["has_vote"].value_counts()[0],
                     df["has_vote"].value_counts()[1]]):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 20,
                 f"{val:,}\n({val/len(df)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].set_title("Vote Receipt: Voted vs Not Voted")
axes[0].set_ylabel("Number of Reviews")
axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

axes[1].hist(df_voted["helpful_ratio"], bins=25,
             color=ACCENT, edgecolor=BG, linewidth=0.8, alpha=0.85)
axes[1].set_title("Helpful Ratio Distribution (voted reviews only)")
axes[1].set_xlabel("Helpful Ratio (helpful_yes / total_vote)")
axes[1].set_ylabel("Count")
axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)

fig.suptitle("Chart 7 - Helpfulness Analysis", fontsize=13, fontweight="bold", y=1.01)
save("07_helpfulness.png")

# =============================================================================
# CHART 08 — Scatter: review length vs total_vote (log scale)
# =============================================================================
sample = df[df["total_vote"] > 0].copy()
sample["log_votes"] = np.log1p(sample["total_vote"])

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(sample["review_length"], sample["total_vote"],
                c=sample["overall"], cmap="RdYlGn",
                alpha=0.55, s=18, edgecolors="none")
m, b, r, p, _ = stats.linregress(np.log1p(sample["review_length"]),
                                 np.log1p(sample["total_vote"]))
x_range = np.linspace(sample["review_length"].min(),
                      sample["review_length"].max(), 300)
ax.plot(x_range,
        np.expm1(m * np.log1p(x_range) + b),
        color=DANGER, linewidth=1.8, linestyle="--",
        label=f"Power-law trend  r={r:.2f}")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Star Rating", fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("Review Length (chars)")
ax.set_ylabel("Total Votes (log scale)")
ax.set_title("Chart 8 - Review Length vs Helpfulness Votes")
ax.legend(); ax.yaxis.grid(True); ax.set_axisbelow(True)
save("08_length_vs_votes.png")

# =============================================================================
# CHART 09 — Correlation heatmap
# =============================================================================
cols  = ["overall", "review_length", "word_count",
         "total_vote", "helpful_yes", "day_diff"]
corr  = df[cols].corr()

fig, ax = plt.subplots(figsize=(7, 5.5))
mask  = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = False  # show full matrix
sns.heatmap(corr, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            center=0, linewidths=0.5, linecolor=BG,
            annot_kws={"size": 9}, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Chart 9 - Correlation Heatmap of Numeric Features")
plt.xticks(rotation=30, ha="right")
save("09_correlation_heatmap.png")

# =============================================================================
# CHART 10 — Violin: word count by rating
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
vp = ax.violinplot(
    [df[df["overall"] == r]["word_count"].clip(upper=300).values for r in order],
    positions=range(1, 6), widths=0.7,
    showmedians=True, showextrema=False
)
for i, (body, color) in enumerate(zip(vp["bodies"], colors)):
    body.set_facecolor(color)
    body.set_alpha(0.7)
    body.set_edgecolor(BORDER)
vp["cmedians"].set_color("#1f2328")
vp["cmedians"].set_linewidth(2)
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f"{int(r)}-star" for r in order])
ax.set_xlabel("Star Rating")
ax.set_ylabel("Word Count (capped at 300)")
ax.set_title("Chart 10 - Word Count Distribution by Star Rating (Violin)")
ax.yaxis.grid(True); ax.set_axisbelow(True)
save("10_wordcount_violin.png")

# =============================================================================
# CHART 11 — Day_diff vs rating (scatter + regression + quartile avg)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sample2 = df.sample(min(1000, len(df)), random_state=42)
sc2 = axes[0].scatter(sample2["day_diff"], sample2["overall"],
                      c=sample2["overall"], cmap="RdYlGn",
                      alpha=0.3, s=12, edgecolors="none")
m2, b2, r2, p2, _ = stats.linregress(df["day_diff"], df["overall"])
x2 = np.linspace(df["day_diff"].min(), df["day_diff"].max(), 300)
axes[0].plot(x2, m2 * x2 + b2, color=DANGER, linewidth=2,
             label=f"Trend  r={r2:.3f}")
axes[0].set_xlabel("day_diff"); axes[0].set_ylabel("Star Rating")
axes[0].set_title("Review Age vs Star Rating")
axes[0].legend(fontsize=8); axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

df["day_diff_q"] = pd.qcut(df["day_diff"], 4,
                           labels=["Q1\n(newest)", "Q2", "Q3", "Q4\n(oldest)"])
q_avg = df.groupby("day_diff_q", observed=True)["overall"].mean()
axes[1].bar(q_avg.index.astype(str), q_avg.values,
            color=[SUCCESS, MAIN_COLOR, ACCENT, DANGER],
            edgecolor=BG, linewidth=1, width=0.55)
axes[1].set_ylim(4.3, 4.8)
for bar, val in zip(axes[1].patches, q_avg.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].set_xlabel("day_diff Quartile"); axes[1].set_ylabel("Avg Star Rating")
axes[1].set_title("Avg Rating by Review Age Quartile")
axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)

fig.suptitle("Chart 11 - Review Age Analysis", fontsize=13, fontweight="bold")
save("11_daydiff_analysis.png")

# =============================================================================
# CHART 12 — Summary story chart (4-panel narrative)
# =============================================================================
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Chart 12 - Data Story: Amazon Review Insights",
             fontsize=14, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# Panel A: donut chart
ax_a = fig.add_subplot(gs[0, 0])
size   = rating_counts.values
wedges_a, _, auto_a = ax_a.pie(
    size, colors=colors, startangle=90,
    autopct="%1.0f%%", pctdistance=0.78,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=1.5))
for at in auto_a:
    at.set_fontsize(8)
ax_a.set_title("A) Rating Share")
centre_circle = plt.Circle((0, 0), 0.45, color=BG)
ax_a.add_patch(centre_circle)
ax_a.text(0, 0, "4,915\nReviews", ha="center", va="center",
          fontsize=9, fontweight="bold", color="#1f2328")

# Panel B: avg length per rating
ax_b = fig.add_subplot(gs[0, 1])
avg_len2 = df.groupby("overall")["review_length"].mean()
ax_b.bar([str(int(r)) for r in avg_len2.index],
         avg_len2.values, color=list(reversed(colors)),
         edgecolor=BG, linewidth=1, width=0.65)
ax_b.set_xlabel("Star Rating"); ax_b.set_ylabel("Avg Chars")
ax_b.set_title("B) Avg Review Length per Rating")
ax_b.yaxis.grid(True); ax_b.set_axisbelow(True)

# Panel C: monthly volume
ax_c = fig.add_subplot(gs[1, 0])
ax_c.fill_between(range(len(monthly)), monthly["count"],
                  alpha=0.15, color=MAIN_COLOR)
ax_c.plot(range(len(monthly)), monthly["count"],
          color=MAIN_COLOR, linewidth=1.8)
step3 = max(1, len(monthly) // 8)
ax_c.set_xticks(range(0, len(monthly), step3))
ax_c.set_xticklabels(monthly["label"].iloc[::step3],
                     rotation=40, ha="right", fontsize=8)
ax_c.set_ylabel("Reviews / Month")
ax_c.set_title("C) Monthly Review Volume")
ax_c.yaxis.grid(True); ax_c.set_axisbelow(True)

# Panel D: vote quartile rating
ax_d = fig.add_subplot(gs[1, 1])
ax_d.bar(q_avg.index.astype(str), q_avg.values,
         color=[SUCCESS, MAIN_COLOR, ACCENT, DANGER],
         edgecolor=BG, linewidth=1, width=0.55)
ax_d.set_ylim(4.3, 4.8)
ax_d.set_xlabel("day_diff Quartile"); ax_d.set_ylabel("Avg Rating")
ax_d.set_title("D) Rating Declines with Review Age")
ax_d.yaxis.grid(True); ax_d.set_axisbelow(True)

save("12_story_dashboard.png")

print("\nAll 12 charts saved to:", OUT)
