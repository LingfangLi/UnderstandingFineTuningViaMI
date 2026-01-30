import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the score file
path = 'D:\\textcomplexity\\squad_output\\all_results.tsv'
# Initialize empty DataFrame
df = pd.DataFrame()
if path.endswith('.tsv'):
    df = pd.read_csv(path, sep='\t')
if path.endswith('.txt'):
    df = pd.read_csv(path,sep='\t')
    # TXT files may have repeated header rows; deduplicate
    df.drop_duplicates(keep='first', inplace=True)


print(f"File loaded successfully! Shape: {df.shape}")
print(f"Total samples: {len(df)}")

# Define columns to extract

columns_to_extract = {
    'type-token ratio': 'type-token ratio (disjoint windows)',
    'average token length': 'average token length (disjoint windows)',
    'lexical density': 'lexical density',
    'rarity': 'rarity'
}

# Create arrays for each metric
arrays = {}
for name, col in columns_to_extract.items():
    if col in df.columns:
        # Convert to numeric, handling any non-numeric values
        arrays[name] = pd.to_numeric(df[col], errors='coerce').dropna().values
        print(f"\n{name}: {len(arrays[name])} valid values")
        print(f"  Min: {arrays[name].min():.4f}")
        print(f"  Max: {arrays[name].max():.4f}")
        print(f"  Mean: {arrays[name].mean():.4f}")
        print(f"  Std: {arrays[name].std():.4f}")
    else:
        print(f"\nWarning: Column '{col}' not found in the data")

# Create bar charts
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

# Plot histograms (bar charts) for each metric
for idx, (name, data) in enumerate(arrays.items()):
    if idx < 4:  # Only plot first 4 metrics
        ax = axes[idx]

        # Create histogram with 50 bins
        n, bins, patches = ax.hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

        # Add mean line
        mean_val = np.mean(data)
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.3f}')

        # Customize the plot
        ax.set_title(f'Distribution of {name}', fontsize=14, fontweight='bold')
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics text
        stats_text = f'n={len(data)}\nμ={mean_val:.3f}\nσ={np.std(data):.3f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', horizontalalignment='right')

plt.tight_layout()
plt.savefig('squad_score_distributions.png', dpi=300, bbox_inches='tight')
#plt.show()

# Alternative: If you want actual bar charts (not histograms) for specific samples
# Let's create bar charts for the first 20 samples
# fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
# axes2 = axes2.ravel()
#
# sample_indices = range(max(20, len(df)))  # First 20 samples
#
# for idx, (name, col) in enumerate(columns_to_extract.items()):
#     if idx < 4 and col in df.columns:
#         ax = axes2[idx]
#
#         # Get values for first 20 samples
#         values = pd.to_numeric(df[col].iloc[sample_indices], errors='coerce')
#
#         # Create bar chart
#         x_pos = np.arange(len(values))
#         bars = ax.bar(x_pos, values, color='lightcoral', edgecolor='black')
#
#         # Customize the plot
#         ax.set_title(f'{name} for {len(values)} Samples', fontsize=14, fontweight='bold')
#         ax.set_xlabel('Sample Index', fontsize=12)
#         ax.set_ylabel(name, fontsize=12)
#         ax.set_xticks(x_pos)
#         ax.set_xticklabels(sample_indices, rotation=45)
#         ax.grid(True, alpha=0.3, axis='y')
#
#         # Color bars based on value (high values in darker color)
#         norm = plt.Normalize(vmin=values.min(), vmax=values.max())
#         for bar, val in zip(bars, values):
#             bar.set_color(plt.cm.Reds(norm(val)))

# plt.tight_layout()
# plt.savefig('twitter_score_bar_charts.png', dpi=300, bbox_inches='tight')
# plt.show()

# Print summary statistics for all four metrics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS FOR ALL METRICS")
print("=" * 60)
for name, data in arrays.items():
    print(f"\n{name}:")
    print(f"  Count: {len(data)}")
    print(f"  Mean: {np.mean(data):.4f}")
    print(f"  Median: {np.median(data):.4f}")
    print(f"  Std Dev: {np.std(data):.4f}")
    print(f"  Min: {np.min(data):.4f}")
    print(f"  Max: {np.max(data):.4f}")
    print(f"  25th percentile: {np.percentile(data, 25):.4f}")
    print(f"  75th percentile: {np.percentile(data, 75):.4f}")