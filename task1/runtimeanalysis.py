#2b
import numpy as np
import matplotlib.pyplot as plt
import time
from combined_convex_hulls import convex_hull_monotone_chain, convex_hull_jarvis_march, graham_scan  # Assuming the convex hull functions are in this file

# Define n values and filenames
n_values = [10, 50, 100, 200, 400, 800, 1000]
filenames = [f"pointcloud{n}" for n in n_values]

# Initialize lists to store runtime for each algorithm
runtime_monotone_chain = []
runtime_jarvis_march = []
runtime_graham_scan = []

# Loop through each n and load the corresponding point cloud data
for filename in filenames:
    # Load point cloud data from .dat file (assuming skiprows=1 for header)
    points = np.loadtxt(f'/root/Desktop/host/SEC2/task1/{filename}.dat', skiprows=1)

    # Run and time convex_hull_monotone_chain
    start_time = time.time()
    convex_hull_monotone_chain(points)
    end_time = time.time()
    runtime_monotone_chain.append(end_time - start_time)

    # Run and time convex_hull_jarvis_march
    start_time = time.time()
    convex_hull_jarvis_march(points)
    end_time = time.time()
    runtime_jarvis_march.append(end_time - start_time)

    # Run and time graham_scan
    start_time = time.time()
    graham_scan(points)
    end_time = time.time()
    runtime_graham_scan.append(end_time - start_time)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(n_values, runtime_monotone_chain, label='Monotone Chain', marker='o', linestyle='-')
plt.plot(n_values, runtime_jarvis_march, label='Jarvis March', marker='s', linestyle='--')
plt.plot(n_values, runtime_graham_scan, label='Graham Scan', marker='^', linestyle='-.')

plt.xlabel('Number of Points (n)')
plt.ylabel('Runtime (seconds)')
plt.title('Convex Hull Algorithm Runtime Comparison')
plt.legend()
plt.grid(True)

# Save the plot to a PNG file
plt.savefig("convex_hull_runtime_comparison.png")

# Show the plot
plt.show()
