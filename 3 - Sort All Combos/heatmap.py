import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Sample data (replace with your actual data)
data = np.random.rand(3, 3)
print(data)
for i in range(len(data)):
    data[i, i] = 1

# Define custom colormap
colors = ["red", "white", "blue"]  # Define the colors for your gradient
cmap_name = "custom_colormap"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data, cmap=custom_cmap, annot=True, fmt=".2f", linewidths=.5)

# Add title and labels
plt.title("Heatmap with Custom Color Gradient")
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")

# Show the plot
plt.show()