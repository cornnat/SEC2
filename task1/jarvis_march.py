import numpy as np
import matplotlib.pyplot as plt

# Read the data file
points = np.loadtxt('/root/Desktop/host/SEC2/mesh.dat', skiprows=1)
x = points[:, 0]  # First column is x
y = points[:, 1]  # Second column is y

# Function to calculate the convex hull using Jarvis March (Gift Wrapping)
def convex_hull_jarvis_march(data):
    # Helper function to determine the orientation (cross product)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Step 1: Find the leftmost point
    leftmost = min(data, key=lambda p: p[0])
    hull = []
    point = leftmost

    # Step 2: Repeat until we return to the starting point
    while True:
        hull.append(point)
        # Step 3: Find the most counterclockwise point
        next_point = data[0]
        for candidate in data:
            if np.array_equal(candidate, point):  # Compare arrays properly
                continue
            if cross(point, next_point, candidate) > 0:  # If candidate is more counterclockwise
                next_point = candidate
        point = next_point
        # Step 4: If we've returned to the leftmost point, stop
        if np.array_equal(point, leftmost):  # Compare arrays properly
            break

    return np.array(hull)

# Call the function to get the convex hull
hull = convex_hull_jarvis_march(points)

# Plot the points and the convex hull
plt.figure(figsize=(6, 6))
plt.plot(points[:, 0], points[:, 1], 'o', label="Points")
plt.plot(hull[:, 0], hull[:, 1], 'r-', label="Convex Hull")

# Close the convex hull polygon
plt.fill(hull[:, 0], hull[:, 1], 'r', alpha=0.3)

# Add labels and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Convex Hull using Jarvis March")
plt.legend()

# Save the plot to the current directory
plt.savefig("convex_hull_jarvis_march_plot.png")

# Show the plot
plt.show()