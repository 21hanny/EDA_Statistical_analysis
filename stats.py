"""
IQ & GPA Statistical Analysis
Recreates all plots: EDA, Correlation, Regression, Residuals, Homoscedasticity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("gpa_iq_cleaned.csv", index_col=0)
print(df.shape)
print(df.describe())

# ── Helper: save fig ───────────────────────────────────────────────────────────
def save(name):
    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — EDA Dashboard (combined_graphs.png)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# 1a. GPA Distribution
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(df["gpa"], bins=15, color="#aad4e8", kde=True, ax=ax1)
ax1.lines[0].set_color("#4ab3d4")
ax1.set_title("Distribution of GPA")
ax1.set_xlabel("gpa")
ax1.set_ylabel("Count")
ax1.grid(True, linestyle="--", alpha=0.4)

# 1b. IQ vs GPA scatter coloured by gender
ax2 = fig.add_subplot(gs[0, 1])
palette = {1: "#3b1f5e", 2: "#f0d040"}
for g, grp in df.groupby("gender"):
    ax2.scatter(grp["iq"], grp["gpa"],
                color=palette[g], label=str(g), alpha=0.85, edgecolors="none", s=60)
ax2.set_title("Relationship between IQ and GPA")
ax2.set_xlabel("iq")
ax2.set_ylabel("gpa")
ax2.legend(title="gender")
ax2.grid(True, linestyle="--", alpha=0.4)

# 1c. Correlation Heatmap
ax3 = fig.add_subplot(gs[1, 0])
corr = df[["gpa", "iq", "gender", "concept"]].corr()
mask = np.zeros_like(corr, dtype=bool)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, ax=ax3, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax3.set_title("Variable Correlation Heatmap")

# 1d. GPA Spread by Gender (box plot)
ax4 = fig.add_subplot(gs[1, 1])
sns.boxplot(x="gender", y="gpa", data=df,
            palette={1: "#a8c8e8", 2: "#e8c4a8"}, ax=ax4,
            hue="gender", legend=False,
            flierprops=dict(marker="o", markerfacecolor="gray", markersize=5))
ax4.set_title("GPA Spread by Gender")
ax4.set_xlabel("gender")
ax4.set_ylabel("gpa")
ax4.grid(True, linestyle="--", alpha=0.4)

save("combined__graphs_.png")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Regression Plot (IQ → GPA)  [light version]
# ══════════════════════════════════════════════════════════════════════════════
X = df["iq"].values.reshape(-1, 1)
y = df["gpa"].values

model = LinearRegression().fit(X, y)
y_pred_line = model.predict(X)

# Stats
r, p = stats.pearsonr(df["iq"], df["gpa"])
r2    = r2_score(y, model.predict(X))
n     = len(df)

# Confidence interval band
x_range = np.linspace(df["iq"].min(), df["iq"].max(), 200).reshape(-1, 1)
y_range = model.predict(x_range)

# Manual 95% CI for the regression line
slope, intercept, r_val, p_val, se = stats.linregress(df["iq"], df["gpa"])
n_pts   = len(df)
x_mean  = df["iq"].mean()
ss_x    = np.sum((df["iq"] - x_mean) ** 2)
y_hat   = slope * df["iq"] + intercept
mse     = np.sum((df["gpa"] - y_hat) ** 2) / (n_pts - 2)
t_crit  = stats.t.ppf(0.975, df=n_pts - 2)
x_flat  = x_range.flatten()
se_line = np.sqrt(mse * (1/n_pts + (x_flat - x_mean)**2 / ss_x))
ci      = np.column_stack([y_range - t_crit * se_line,
                           y_range + t_crit * se_line])

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# CI band
ax.fill_between(x_range.flatten(), ci[:, 0], ci[:, 1],
                alpha=0.25, color="#e07070", label="95% Confidence Interval")

# Scatter coloured by IQ
sc = ax.scatter(df["iq"], df["gpa"], c=df["iq"], cmap="viridis",
                s=60, zorder=3, edgecolors="none")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("IQ Score")

# Regression line
ax.plot(x_range, y_range, color="red", linewidth=2.5, label="Regression Line")

# Stats box
stats_text = (f"Correlation (r): {r:.4f}\n"
              f"R² Value: {r2:.4f}\n"
              f"Sample Size: {n}\n"
              f"p-value: <0.001")
ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
        fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#4a9fd4", linewidth=2))

ax.set_title("The Relationship Between Cognitive Ability and Academic Performance",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Intelligence Quotient (IQ)", fontsize=12)
ax.set_ylabel("Grade Point Average (GPA)", fontsize=12, color="#4a7abf")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xlim(68, 138)
ax.set_ylim(-0.2, 5.5)

save("iq_gpa_regression_plot.png")

# Also save transparent version
fig2, ax2 = plt.subplots(figsize=(10, 7))
fig2.patch.set_alpha(0)
ax2.set_facecolor((0, 0, 0, 0))
ax2.fill_between(x_range.flatten(), ci[:, 0], ci[:, 1],
                 alpha=0.25, color="#e07070", label="95% Confidence Interval")
sc2 = ax2.scatter(df["iq"], df["gpa"], c=df["iq"], cmap="viridis",
                  s=60, zorder=3, edgecolors="none")
cbar2 = plt.colorbar(sc2, ax=ax2)
cbar2.set_label("IQ Score")
ax2.plot(x_range, y_range, color="red", linewidth=2.5, label="Regression Line")
ax2.text(0.03, 0.97, stats_text, transform=ax2.transAxes,
         fontsize=11, verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                   edgecolor="#4a9fd4", linewidth=2))
ax2.set_title("The Relationship Between Cognitive Ability and Academic Performance",
              fontsize=14, fontweight="bold", pad=15)
ax2.set_xlabel("Intelligence Quotient (IQ)", fontsize=12)
ax2.set_ylabel("Grade Point Average (GPA)", fontsize=12, color="#4a7abf")
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.set_xlim(68, 138)
ax2.set_ylim(-0.2, 5.5)
plt.tight_layout()
plt.savefig("iq_gpa_regression_plot_transparent.png", dpi=150,
            bbox_inches="tight", transparent=True)
plt.show(); plt.close()
print("Saved: iq_gpa_regression_plot_transparent.png")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Residual Analysis  (residuals_plot + refined_residuals_plot)
# ══════════════════════════════════════════════════════════════════════════════
residuals = y - model.predict(X)
res_mean  = residuals.mean()
res_std   = residuals.std()

# ── Version A: refined_residuals_plot (dark steel style) ──────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Residual Analysis: Evaluating GPA Predictions from IQ",
             fontsize=14, fontweight="bold")

steel = "#6b8fa8"
ax1.hist(residuals, bins=14, color=steel, edgecolor="white", alpha=0.85)
ax1.plot(np.linspace(residuals.min(), residuals.max(), 200),
         stats.norm.pdf(np.linspace(residuals.min(), residuals.max(), 200),
                        res_mean, res_std) * len(residuals) *
         (residuals.max() - residuals.min()) / 14,
         color="#2c4a5e", linewidth=2.5)
ax1.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Mean = 0")
ax1.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\n$N$ = {n}",
         transform=ax1.transAxes, fontsize=11, va="top")
ax1.set_title("Frequency of Residuals")
ax1.set_xlabel("Prediction Error (Residual)")
ax1.set_ylabel("Number of Students")
ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.3)

x_res = np.linspace(residuals.min() - 0.5, residuals.max() + 0.5, 300)
ax2.hist(residuals, bins=14, density=True, color=steel, edgecolor="white", alpha=0.85,
         label="Actual Residuals")
ax2.plot(x_res, stats.norm.pdf(x_res, res_mean, res_std),
         "r--", linewidth=2, label="Theoretical Normal")
ax2.axvline(0, color="red", linestyle="--", linewidth=1)
ax2.set_title("Probability Density vs. Normal Curve")
ax2.set_xlabel("Prediction Error (Residual)")
ax2.set_ylabel("Density")
ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.3)

save("refined_residuals_plot.png")

# ── Version B: residuals_plot (blue/pink style) ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Distribution of Residuals from Linear Regression Model",
             fontsize=14, fontweight="bold")

ax1.hist(residuals, bins=14, color="#aad4e8", edgecolor="white")
ax1.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
ax1.set_title("Histogram of Residuals\n(GPA ~ IQ)")
ax1.set_xlabel("Residuals"); ax1.set_ylabel("Frequency")
ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.4)

ax2.hist(residuals, bins=14, density=True, color="#e8a8a8",
         edgecolor="white", alpha=0.85, label="Histogram")
ax2.plot(x_res, stats.norm.pdf(x_res, res_mean, res_std),
         "b-", linewidth=2,
         label=f"Normal(μ={res_mean:.2f}, σ={res_std:.2f})")
ax2.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
ax2.set_title("Density Plot of Residuals\n(GPA ~ IQ)")
ax2.set_xlabel("Residuals"); ax2.set_ylabel("Density")
ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.4)

save("residuals_plot.png")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — Homoscedasticity (Residuals vs Predictor)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_facecolor("white")

ax.scatter(df["iq"], residuals, color="#5a5a7a", s=60, zorder=3, label="Residuals")
ax.axhline(0, color="red", linestyle="--", linewidth=2, label="Zero Error Line")

# Smoothed trend line (lowess-style via polynomial)
z   = np.polyfit(df["iq"], residuals, 2)
p_  = np.poly1d(z)
xp  = np.linspace(df["iq"].min(), df["iq"].max(), 300)
ax.plot(xp, p_(xp), ":", color="gray", linewidth=1.8)

ax.set_title("Task 4: Homoscedasticity Analysis\n(Residuals vs. Predictor)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("IQ Score (Predictor)", fontsize=12)
ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)

save("task4_homoscedasticity.png")


# ══════════════════════════════════════════════════════════════════════════════
# BONUS — Print regression summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Regression Summary")
print("="*60)
print(f"Pearson r  : {r:.4f}")
print(f"R²         : {r2:.4f}")
print(f"p-value    : {p:.6f}")
print(f"Intercept  : {model.intercept_:.4f}")
print(f"Slope (IQ) : {model.coef_[0]:.4f}")
print(f"Residual μ : {res_mean:.4f}")
print(f"Residual σ : {res_std:.4f}")