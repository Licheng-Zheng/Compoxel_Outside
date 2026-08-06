import matplotlib.pyplot as plt
import numpy as np

# --- 1. THE RAW BENCHMARK DATA ---
entities = np.array([10_000, 50_000, 100_000, 500_000, 1_000_000])
entities_labels = ["10k", "50k", "100k", "500k", "1M"]

# Time Data (ms)
naive_ms = np.array([9.28, 37.54, 121.16, 2783.10, 10527.18])
tree_ms = np.array([18.09, 36.65, 78.18, 1348.89, 5283.92])
mesh_ms = np.array([14.93, 25.20, 39.66, 755.26, 2682.13])
hybrid_ms = np.array([16.33, 27.51, 60.60, 1512.16, 6209.75])

# MSE Data
tree_mse = np.array([49.47, 48.76, 47.86, 41.86, 45.13])
mesh_mse = np.array([49.93, 48.86, 47.95, 41.57, 44.95])
hybrid_mse = np.array([49.72, 48.79, 47.80, 41.58, 44.99])

# Standard Colors
COLOR_NAIVE = 'tab:red'
COLOR_TREE = 'tab:purple'
COLOR_MESH = 'tab:green'
COLOR_HYBRID = 'tab:blue'

# --- 2. FIGURE SETUP ---
fig = plt.figure(figsize=(18, 12), dpi=200)

def format_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ==========================================
# SUBPLOT 1: SCALABILITY (Log-Log Scale)
# ==========================================
ax1 = plt.subplot(2, 2, 1)
format_axis(ax1, "Algorithm Compute Time Scaling", "Entity Count (Stars)", "Compute Time per Frame (ms)")
ax1.plot(entities, naive_ms, marker='o', markersize=8, linewidth=3, color=COLOR_NAIVE, label='Naive O(N²)')
ax1.plot(entities, tree_ms, marker='s', markersize=6, linewidth=2, color=COLOR_TREE, linestyle='--', label='Tree (Hierarchical)')
ax1.plot(entities, mesh_ms, marker='^', markersize=6, linewidth=2, color=COLOR_MESH, linestyle='--', label='Mesh (Grid)')
ax1.plot(entities, hybrid_ms, marker='D', markersize=8, linewidth=4, color=COLOR_HYBRID, label='Hybrid (P³M)')

ax1.fill_between(entities, hybrid_ms, color=COLOR_HYBRID, alpha=0.1)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(loc='upper left', frameon=True, fontsize=10)

# ==========================================
# SUBPLOT 2: SPEEDUP FACTOR
# ==========================================
ax2 = plt.subplot(2, 2, 2)
format_axis(ax2, "Hybrid P³M Speedup vs. Naive", "Entity Count (Stars)", "Speedup Multiplier (x Faster)")
speedup_factors = naive_ms / hybrid_ms

bars = ax2.bar(entities_labels, speedup_factors, color=COLOR_HYBRID, width=0.5)
for bar in bars:
    height = bar.get_height()
    ax2.annotate(f'{height:.1f}x', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

# ==========================================
# SUBPLOT 3: MSE ACCURACY LOSS
# ==========================================
ax3 = plt.subplot(2, 2, 3)
format_axis(ax3, "Mean Squared Error (vs. Ground Truth)", "Entity Count (Stars)", "MSE Magnitude")

x_indexes = np.arange(len(entities))
bar_width = 0.25

ax3.bar(x_indexes - bar_width, tree_mse, width=bar_width, color=COLOR_TREE, label='Tree')
ax3.bar(x_indexes, mesh_mse, width=bar_width, color=COLOR_MESH, label='Mesh')
ax3.bar(x_indexes + bar_width, hybrid_mse, width=bar_width, color=COLOR_HYBRID, label='Hybrid')

ax3.set_xticks(x_indexes)
ax3.set_xticklabels(entities_labels)
# Set Y-axis limits slightly above the min/max to show the nuanced differences
ax3.set_ylim(40, 52)
ax3.legend(loc='upper right', frameon=True, fontsize=10)

# ==========================================
# SUBPLOT 4: PERFORMANCE VS. ACCURACY (Pareto)
# ==========================================
ax4 = plt.subplot(2, 2, 4)
format_axis(ax4, "Performance vs. Accuracy Trade-off (N=1,000,000)", "Mean Squared Error (Lower = More Accurate)", "Compute Time (Lower = Faster)")

# Plotting the 1 Million Entity data points
ax4.scatter([45.13], [5283.92], color=COLOR_TREE, s=200, marker='s', edgecolors='black', zorder=5, label="Tree")
ax4.scatter([44.95], [2682.13], color=COLOR_MESH, s=200, marker='^', edgecolors='black', zorder=5, label="Mesh")
ax4.scatter([44.99], [6209.75], color=COLOR_HYBRID, s=300, marker='D', edgecolors='black', zorder=5, label="Hybrid")
ax4.scatter([0.0], [10527.18], color=COLOR_NAIVE, s=200, marker='o', edgecolors='black', zorder=5, label="Naive (Ground Truth)")

# Add labels next to the points
ax4.annotate('Tree', xy=(45.13, 5283.92), xytext=(10, 0), textcoords='offset points', color=COLOR_TREE, fontweight='bold')
ax4.annotate('Mesh', xy=(44.95, 2682.13), xytext=(10, 0), textcoords='offset points', color=COLOR_MESH, fontweight='bold')
ax4.annotate('Hybrid', xy=(44.99, 6209.75), xytext=(15, 0), textcoords='offset points', color=COLOR_HYBRID, fontweight='bold')
ax4.annotate('Naive (0 MSE)', xy=(0.0, 10527.18), xytext=(15, -5), textcoords='offset points', color=COLOR_NAIVE, fontweight='bold')

# Limit X-axis to show the grouping, but include 0
ax4.set_xlim(-5, 50)
ax4.invert_xaxis() # Invert so that "Better" (Lower MSE) is to the right!
ax4.set_xlabel("Mean Squared Error ➔ (Lower is Better)", fontsize=11)
ax4.set_ylabel("Compute Time (ms) ➔ (Lower is Better)", fontsize=11)

# --- 3. SAVE THE FIGURE ---
plt.tight_layout(pad=3.0)
output_file = "compoxel_benchmark_infographic.png"
plt.savefig(output_file, transparent=False)
print(f"✅ Infographic generated and saved as {output_file}")