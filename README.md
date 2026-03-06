# Statistical_analysis}
IQ & GPA Statistical Analysis
A full exploratory and inferential statistical analysis investigating the relationship between cognitive ability (IQ) and academic performance (GPA), with supporting analyses on gender differences and conceptual understanding scores.

Dataset
File: gpa_iq_cleaned.csv / gpa_iq_fixed.csv
Sample size: 78 students
Variables:
ColumnDescriptionobsObservation IDgpaGrade Point Average (0–4.5 scale)iqIQ score (range: ~70–135)genderGender (1 = male, 2 = female)conceptConceptual understanding score

Analyses Performed
1. Exploratory Data Analysis (EDA)

Distribution of GPA (histogram + KDE)
IQ vs. GPA scatter plot coloured by gender
GPA spread by gender (box plot)
Variable correlation heatmap across all four variables

2. Correlation Analysis

Pearson correlation between all variable pairs
Key finding: IQ–GPA correlation r = 0.6483 (strong positive)
concept also shows meaningful correlation with both GPA (r = 0.55) and IQ (r = 0.49)
gender shows near-zero correlation with GPA (r = −0.10)

3. Linear Regression — IQ → GPA

Simple OLS regression: GPA ~ IQ
R² = 0.4203 — IQ explains ~42% of variance in GPA
p-value < 0.001 — statistically significant
95% confidence interval band plotted around regression line

4. Residual Analysis

Histogram of residuals with KDE overlay
Density plot vs. theoretical normal curve (μ = −0.00, σ = 0.68)
Residuals are approximately normally distributed and centred at zero

5. Homoscedasticity Check

Residuals vs. IQ (predictor) scatter plot
Dotted trend line to detect systematic patterns
Residual spread is reasonably consistent across IQ values, supporting the linear model assumption


Key Results
MetricValuePearson r (IQ–GPA)0.6483R²0.4203p-value< 0.001Sample size78Residual std. dev.0.68

Visualisations
FileDescriptioncombined__graphs_.pngEDA dashboard: GPA distribution, IQ-GPA scatter, correlation heatmap, GPA boxplot by genderiq_gpa_regression_plot.pngRegression plot (dark background)iq_gpa_regression_plot_transparent.pngRegression plot (light/transparent background)Relation_IQ___performace.pngIQ vs. GPA regression — white background versionresiduals_plot.pngResidual histogram + density vs. normal curverefined_residuals_plot.pngRefined residual frequency & probability density plotstask4_homoscedasticity.pngResiduals vs. IQ predictor — homoscedasticity check

How to Reproduce

Clone the repo and place the CSV in the working directory
Install dependencies:

bash   pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn

Run the analysis notebook/script — all plots will be regenerated from the cleaned CSV


Conclusions
The analysis provides strong statistical evidence that IQ is a significant positive predictor of GPA. The linear model is valid — residuals are approximately normal, centred at zero, and show no strong heteroscedasticity. However, with R² ≈ 0.42, roughly 58% of GPA variance remains unexplained by IQ alone, suggesting other factors (study habits, motivation, conceptual understanding, etc.) play an important role.
