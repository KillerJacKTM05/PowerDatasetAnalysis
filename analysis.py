import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("solar_panel.csv")

# Set up the plot aesthetics
sns.set(style="whitegrid")
df.hist(bins=20, figsize=(16, 12), layout=(5, 4))
plt.tight_layout()
plt.suptitle("Histograms of Each Column", y=1.02)
plt.show()

# Aggregate power generation per day
daily_power = df.groupby(["Year", "Month", "Day"])["Power Generated"].sum().reset_index()

# Create a 'date' column for easier plotting
daily_power["Date"] = pd.to_datetime(daily_power[["Year", "Month", "Day"]])

# Plot daily power generation
plt.figure(figsize=(14, 6))
sns.lineplot(x="Date", y="Power Generated", data=daily_power)
plt.title("Daily Power Generation Over the Year")
plt.xlabel("Date")
plt.ylabel("Power Generated")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Day with maximum power
max_day = daily_power.loc[daily_power["Power Generated"].idxmax()]
print("Day with maximum power generated:", max_day["Date"].date(), "with", max_day["Power Generated"], "units")

# Check unique values in Sky Cover
print("Unique Sky Cover values:", df["Sky Cover"].unique())

# Define Sky Cover categories
sky_cover_labels = {
    0: "Clear",
    1: "Mostly Clear",
    2: "Partly Cloudy",
    3: "Mostly Cloudy",
    4: "Overcast"
}

# Apply label mapping to create clusters
df["Sky Cover Cluster"] = df["Sky Cover"].map(sky_cover_labels)

# Aggregate per day and sky cover cluster
daily_clustered = df.groupby(["Year", "Month", "Day", "Sky Cover Cluster"])["Power Generated"].sum().reset_index()
daily_clustered["Date"] = pd.to_datetime(daily_clustered[["Year", "Month", "Day"]])

# Plot power generation by multi-level Sky Cover Cluster
plt.figure(figsize=(14, 6))
sns.lineplot(x="Date", y="Power Generated", hue="Sky Cover Cluster", data=daily_clustered, palette="viridis")
plt.title("Power Generation Over Time by Sky Cover Cluster")
plt.xlabel("Date")
plt.ylabel("Power Generated")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Select numeric columns for correlation analysis (excluding Sky Cover itself)
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop("Sky Cover")

# Generate and plot correlation heatmaps for each Sky Cover Cluster
unique_clusters = df["Sky Cover Cluster"].unique()
n_clusters = len(unique_clusters)

fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 7), sharey=True)

for i, cluster in enumerate(sorted(unique_clusters)):
    cluster_df = df[df["Sky Cover Cluster"] == cluster]
    corr_matrix = cluster_df[numeric_cols].corr()
    sns.heatmap(corr_matrix, ax=axes[i], cmap="coolwarm", annot=True, fmt=".2f")
    axes[i].set_title(f"Correlation: {cluster}")

plt.tight_layout()
plt.show()
