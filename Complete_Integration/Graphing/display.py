import matplotlib
# Force headless rendering to bypass frozen local GUI windows
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# ==========================================
# CONFIGURATION VARIABLES (Tweak these!)
# ==========================================
MAX_ENTITIES_K = 20000       # Max X-axis projection (in thousands)
MAX_TIME_SEC = 400           # Max Y-axis viewport limit (in seconds)

FIG_WIDTH = 14               # Width of the output image
FIG_HEIGHT = 9               # Height of the output image
FIG_DPI = 200                # Resolution/quality of the output image

LEGEND_LOC = 'upper left'    # Location of the legend (e.g., 'upper left', 'best', 'center right')
TEXTBOX_X = 0.97             # X-coordinate of the equation box (0.0 to 1.0, where 1.0 is far right)
TEXTBOX_Y = 0.04             # Y-coordinate of the equation box (0.0 to 1.0, where 0.0 is absolute bottom)

GRAPH_TITLE = 'Empirical Big-O Curve Fitting on NVIDIA T4'
OUTPUT_FILENAME = 'curve_fitted_performance_full.png'

# Convert to Thousands of Entities (K) and Seconds
entities_k = np.array([100000, 500000, 1000000, 5000000, 10000000]) / 1000.0

# DNF is replaced with np.nan so Matplotlib ignores it safely
naive_sec = np.array([130.25, 3059.95, 11815.10, 264285.20, np.nan]) / 1000.0
tree_sec = np.array([185.21, 848.86, 1783.50, 27271.80, 96985.15]) / 1000.0
mesh_sec = np.array([158.87, 708.16, 1474.25, 19512.37, 63649.05]) / 1000.0
hybrid_sec = np.array([164.54, 868.93, 2264.61, 50441.12, 209969.00]) / 1000.0

# ==========================================
# MATHEMATICAL CURVE FITTING DEFINITIONS
# ==========================================
def fit_linear(n, a, b):
    """ Mathematical model for O(N) """
    return a * n + b

def fit_quadratic(n, a, b):
    """ Mathematical model for O(N^2) """
    return a * (n**2) + b

def fit_n_log_n(n, a, b):
    """ Mathematical model for O(N log N) """
    return a * n * np.log2(n) + b

def calculate_r2(y_true, y_pred):
    """ Calculates the R-squared Goodness of Fit (1.0 is a perfect match) """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Create the FIRST distinct figure
fig1 = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)

# Generate a high-resolution X-axis projected out to MAX_ENTITIES_K
smooth_n = np.linspace(min(entities_k), MAX_ENTITIES_K, 1000)

# --- 1. NAIVE O(N^2) ---
# Filter out the NaN so scipy can fit the curve to the 4 valid points (up to 5M)
valid_naive_mask = ~np.isnan(naive_sec)
valid_entities_naive = entities_k[valid_naive_mask]
valid_naive = naive_sec[valid_naive_mask]

popt_naive, _ = curve_fit(fit_quadratic, valid_entities_naive, valid_naive)
r2_naive = calculate_r2(valid_naive, fit_quadratic(valid_entities_naive, *popt_naive))

plt.plot(smooth_n, fit_quadratic(smooth_n, *popt_naive), color='#dc2626', linestyle='--', linewidth=2, alpha=0.5)
plt.plot(valid_entities_naive, valid_naive, marker='o', linestyle='', markersize=10, color='#dc2626', label=r'Raw Data (Naive $O(N^2)$)')

# --- 2. HYBRID O(N log N) ---
# The Ablation Study proves this is O(N log N)
popt_hybrid, _ = curve_fit(fit_n_log_n, entities_k, hybrid_sec)
r2_hybrid = calculate_r2(hybrid_sec, fit_n_log_n(entities_k, *popt_hybrid))

plt.plot(smooth_n, fit_n_log_n(smooth_n, *popt_hybrid), color='#16a34a', linestyle='--', linewidth=2, alpha=0.5)
plt.plot(entities_k, hybrid_sec, marker='^', linestyle='', markersize=11, color='#16a34a', label=r'Raw Data (Hybrid TreePM)')

# --- 3. TREE O(N log N) ---
popt_tree, _ = curve_fit(fit_n_log_n, entities_k, tree_sec)
r2_tree = calculate_r2(tree_sec, fit_n_log_n(entities_k, *popt_tree))
plt.plot(smooth_n, fit_n_log_n(smooth_n, *popt_tree), color='#2563eb', linestyle='--', linewidth=2, alpha=0.5)
plt.plot(entities_k, tree_sec, marker='s', linestyle='', markersize=10, color='#2563eb', label=r'Raw Data (Tree)')

# --- 4. PURE MESH O(N log N) ---
popt_mesh, _ = curve_fit(fit_n_log_n, entities_k, mesh_sec)
r2_mesh = calculate_r2(mesh_sec, fit_n_log_n(entities_k, *popt_mesh))
plt.plot(smooth_n, fit_n_log_n(smooth_n, *popt_mesh), color='#94a3b8', linestyle='--', linewidth=2, alpha=0.5)
plt.plot(entities_k, mesh_sec, marker='D', linestyle='', markersize=8, color='#94a3b8', label=r'Raw Data (Pure Mesh)')

# Aesthetic formatting
plt.title(GRAPH_TITLE, fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Number of Entities (in Thousands)', fontsize=16, fontweight='bold')
plt.ylabel('Execution Time per Frame (Seconds)', fontsize=16, fontweight='bold')

# Cap the viewport so the scalable algorithms are visible, letting Naive shoot off the top
plt.ylim(0, MAX_TIME_SEC)
plt.xlim(0, MAX_ENTITIES_K)

plt.grid(True, which="major", ls="-", alpha=0.8)
plt.grid(True, which="minor", ls="--", alpha=0.4)

# --- EQUATION TEXTBOX ---
def f_eq(popt, is_n2=False):
    """ Converts Scipy's raw math output into beautiful LaTeX equations """
    a, b = popt
    sign = "+" if b >= 0 else "-"
    var = "N^2" if is_n2 else "N \\log_2(N)"

    # Convert Python's '1.05e-05' to LaTeX '1.05 \times 10^{-5}' for maximum visual oomph
    a_str = f"{a:.2e}".split('e')
    a_base = a_str[0]
    a_exp = int(a_str[1])
    a_tex = f"{a_base} \\times 10^{{{a_exp}}}"

    return f"$T = {a_tex} \\cdot {var} {sign} {abs(b):.2f}$"

eq_text = (
    "$\\bf{Empirical\\ Mathematical\\ Models\\ (Scipy\\ Fit)}$\n\n"
    f"Naive ($O(N^2)$):       {f_eq(popt_naive, True)}    $(R^2 = {r2_naive:.4f})$\n"
    f"Hybrid ($O(N\\log N)$): {f_eq(popt_hybrid, False)}    $(R^2 = {r2_hybrid:.4f})$\n"
    f"Tree ($O(N\\log N)$):   {f_eq(popt_tree, False)}    $(R^2 = {r2_tree:.4f})$\n"
    f"Mesh ($O(N\\log N)$):   {f_eq(popt_mesh, False)}    $(R^2 = {r2_mesh:.4f})$\n\n"
    "*(N = Entities in Thousands, T = Execution Time in Seconds)*"
)

# Render the text box in the empty bottom-right corner
props = dict(boxstyle='round,pad=0.8', facecolor='#f8fafc', alpha=0.9, edgecolor='#cbd5e1')
plt.gca().text(TEXTBOX_X, TEXTBOX_Y, eq_text, transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

import matplotlib.lines as mlines
dashed_line = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=2, label='Scipy Theoretical Curve Fit')
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(dashed_line)
labels.append('Scipy Theoretical Curve Fit')

plt.legend(handles=handles, labels=labels, fontsize=14, loc=LEGEND_LOC)

plt.tight_layout()
plt.savefig(OUTPUT_FILENAME, bbox_inches='tight')
print(f"✅ Saved Graph 1: '{OUTPUT_FILENAME}'")


# ==========================================
# GRAPH 2: MEAN SQUARED ERROR (MSE) ACCURACY
# ==========================================
print("\nGenerating MSE Accuracy Graph...")

# Create the SECOND distinct figure
fig2 = plt.figure(figsize=(10, 6), dpi=FIG_DPI)

# MSE data extracted from the benchmark table (stops at 1M because Naive timed out)
mse_entities_k = np.array([100, 500, 1000])
tree_mse = np.array([31.590673, 29.862026, 27.985409])
mesh_mse = np.array([31.419094, 29.362982, 27.250221])
hybrid_mse = np.array([31.386168, 27.809664, 26.109495])

# Plot the approximation methods
plt.plot(mse_entities_k, tree_mse, marker='s', linestyle='-', linewidth=3, markersize=10, color='#2563eb', label='Tree Method')
plt.plot(mse_entities_k, mesh_mse, marker='D', linestyle='-', linewidth=3, markersize=8, color='#94a3b8', label='Pure Mesh')
plt.plot(mse_entities_k, hybrid_mse, marker='^', linestyle='-', linewidth=3, markersize=11, color='#16a34a', label='Hybrid TreePM')

# Add a flat baseline for Naive (Perfect Accuracy = 0 Error)
plt.axhline(0, color='#dc2626', linestyle='--', linewidth=3, alpha=0.7, label='Naive Ground Truth (0 Error)')

# Aesthetic formatting
plt.title('Approximation Error vs. Naive Baseline', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of Entities (in Thousands)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Squared Error (Positional Drift)', fontsize=14, fontweight='bold')

plt.grid(True, which="major", ls="-", alpha=0.8)
plt.grid(True, which="minor", ls="--", alpha=0.4)

# Force X-axis to label the exact benchmark checkpoints
plt.xticks([100, 500, 1000], fontsize=12)
plt.yticks(fontsize=12)

# Ensure the Y-axis drops down to 0 so the baseline is visible
plt.ylim(-2, 35)
plt.legend(fontsize=12, loc='upper right')

plt.tight_layout()
MSE_FILENAME = 'mse_accuracy_graph.png'
plt.savefig(MSE_FILENAME, bbox_inches='tight')

print(f"✅ Saved Graph 2: '{MSE_FILENAME}'")
