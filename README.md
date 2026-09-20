# Task 3 — Data Visualization

## Overview
This task transforms raw Amazon Product Review data into a comprehensive set of **12 publication-quality visualizations** using **Matplotlib** and **Seaborn**. The charts are organized into a single-page HTML dashboard (`dashboard.html`) that tells a cohesive data story.

---

## Dataset
| Property | Value |
|---|---|
| File | `amazon_reviews.csv` (one level up) |
| Rows | 4,915 reviews |
| Period | January 2012 – December 2014 |
| Product | ASIN B007WTAJTO (single product) |

---

## Visualizations Produced

| Chart | Type | Insight |
|---|---|---|
| 01 | Bar chart | Rating distribution — 79.8% are 5-star |
| 02 | Pie / Donut | Rating share proportions |
| 03 | Line + area fill | Monthly review volume over time |
| 04 | Grouped bar | Yearly volume split by star rating |
| 05 | Box plot (notched) | Review length spread per rating |
| 06 | Horizontal bar | Average review length per rating |
| 07 | Histogram + bar | Helpfulness vote distribution |
| 08 | Scatter + trend | Review length vs total votes (log scale) |
| 09 | Heatmap | Correlation matrix of numeric features |
| 10 | Violin plot | Word count distribution per rating |
| 11 | Scatter + grouped bar | Review age vs rating |
| 12 | 4-panel story | Combined narrative dashboard |

---

## Key Visual Insights

- **Ratings are dominated by 5-star** (79.8%) — extreme positive skew visible in both bar and pie charts.
- **Low-star reviews are longer** — 1-star averages 559 chars, 5-star only 234. Clearly visible in box and violin plots.
- **Review volume peaked in 2013** — monthly line chart shows clear growth → peak → decline lifecycle.
- **Only 11.3% of reviews received votes** — most helpfulness mass is concentrated in a small minority.
- **Longer reviews attract more votes** — scatter chart with log-scale y-axis reveals a power-law relationship.
- **Ratings decline with review age** — Q1 (newest) averages 4.70 vs Q4 (oldest) at 4.45.
- **review_length and word_count highly correlated (r=0.97)** — heatmap confirms redundancy.

---

## Project Structure
```
Task_3_DataViz/
├── visualizations.py    <- Main script generating all 12 charts
├── requirements.txt     <- Python dependencies
├── README.md            <- This file
├── dashboard.html       <- Single-page visual dashboard (all charts embedded)
└── plots/               <- Generated PNG files (auto-created on run)
    ├── 01_rating_distribution.png
    ├── 02_rating_pie.png
    ├── 03_monthly_volume.png
    ├── 04_yearly_by_rating.png
    ├── 05_length_boxplot.png
    ├── 06_avg_length_hbar.png
    ├── 07_helpfulness.png
    ├── 08_length_vs_votes.png
    ├── 09_correlation_heatmap.png
    ├── 10_wordcount_violin.png
    ├── 11_daydiff_analysis.png
    └── 12_story_dashboard.png
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate all charts
```bash
# From inside Task_3_DataViz/
python visualizations.py
```

### 3. View the dashboard
Open `dashboard.html` in any modern browser — all charts are embedded inline, no server needed.

---

## Tools Used
| Tool | Version | Purpose |
|---|---|---|
| Matplotlib | >= 3.6 | Base charts, layout, styling |
| Seaborn | >= 0.12 | Heatmap, statistical charts |
| Pandas | >= 1.5 | Data wrangling, groupby |
| NumPy | >= 1.23 | Numeric operations, regression |
| SciPy | >= 1.9 | Linear regression for trend lines |
