# Statistical_analysis}
# IQ & GPA Statistical Analysis

A comprehensive exploratory and inferential statistical analysis investigating the relationship between cognitive ability (IQ) and academic performance (GPA).

---

## Dataset Overview

| Column | Description |
|--------|-------------|
| `obs` | Observation ID |
| `gpa` | Grade Point Average (0–4.5 scale) |
| `iq` | IQ score (range: ~70–135) |
| `gender` | Gender (1 = male, 2 = female) |
| `concept` | Conceptual understanding score |

**Sample size:** 78 students

---

## Exploratory Data Analysis

![EDA Dashboard](images/combined-graphs.png)

Key patterns identified:
- GPA shows approximate normality with μ ≈ 3.0
- Strong positive IQ-GPA correlation (r = 0.6483)
- Gender displays minimal correlation with GPA (r = -0.10)

---

## Linear Regression: IQ → GPA

![Regression Analysis](images/iq-gpa-regression-transparent.png)

**Model Results:**
- **R² = 0.4203** — IQ explains ~42% of GPA variance
- **p-value < 0.001** — Statistically significant
- **Equation:** GPA = β₀ + β₁(IQ) + ε

The 95% confidence interval (shaded band) indicates reliable prediction within this range.

---

## Residual Analysis

![Residuals](images/refined-residuals-plot.png)

**Assumptions validated:**
- Residuals approximately normal (μ = 0, σ = 0.68)
- No systematic patterns detected
- Model assumptions satisfied

---

## Homoscedasticity Check

![Homoscedasticity](images/task4-homoscedasticity.png)

Residual spread remains consistent across IQ values, confirming constant variance assumption for linear regression.

---

## Key Results

| Metric | Value |
|--------|-------|
| Pearson r (IQ–GPA) | 0.6483 |
| R² | 0.4203 |
| p-value | < 0.001 |
| Sample size | 78 |
| Residual std. dev. | 0.68 |

---

## Conclusions

The analysis provides strong statistical evidence that **IQ is a significant positive predictor of GPA**. However, with R² ≈ 0.42, approximately **58% of GPA variance remains unexplained**, suggesting other factors (study habits, motivation, conceptual understanding) play important roles.

The linear model is valid:

✅ Residuals are approximately normal  
✅ Residuals centered at zero  
✅ No strong heteroscedasticity detected

---

## How to Reproduce
```bash
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
- `gpa_iq_cleaned.csv` — Cleaned dataset
- `images/` — All visualizations

---

