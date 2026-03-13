# IQ & GPA Statistical Analysis

A comprehensive exploratory and inferential statistical analysis investigating the relationship between cognitive ability (IQ) and academic performance (GPA).

## Dataset Overview

| Column | Description |
|--------|-------------|
| obs | Observation ID |
| gpa | Grade Point Average (0–4.5 scale) |

[rest of your table...]

---

## Visualizations

<p align="center">
  <img src="Combined_graphs .png" alt="EDA Dashboard" width="900">
</p>

### Analysis Breakdown

**Panel 1 - GPA Distribution** (Top Left)
- Normal distribution with mean ≈ 3.0
- KDE overlay confirms bell curve shape

**Panel 2 - IQ vs GPA** (Top Right)  
- Clear positive correlation (r = 0.65)
- Gender-coded scatter shows consistent pattern

**Panel 3 - Correlation Matrix** (Bottom Left)
- Heatmap reveals IQ as dominant predictor
- Gender shows minimal correlation (-0.10)

**Panel 4 - Gender Comparison** (Bottom Right)
- Overlapping box plots indicate similar performance
- No statistically significant gender difference

---

## Key Results

[rest of your README...]```bash
# Clone the repository
git clone https://github.com/21hanny/EDA_Statistical_analysis.git

# Install dependencies
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn

# Run the analysis
jupyter notebook analysis.ipynb
```

---

## Files

- `analysis.ipynb` — Main analysis notebook
- `gpa_iq_cleaned.csv` — Dataset
- `images/Combined_SS.png` — Exploratory data analysis visualization

---

