import numpy as np
import matplotlib.pyplot as plt

#read the data file
points = np.loadtxt('/root/Desktop/host/SEC2/mesh.dat', skiprows = 1)
x = points[:, 0]  # First column is x
y = points[:, 1]  # Second column is y

# Function to calculate the convex hull using monotone chain
def convex_hull_monotone_chain(data):
    # Sort the points by x (and by y if necessary)
    data = data[np.argsort(data[:, 0])]
    
    # Function to check the orientation (cross product)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build the lower hull
    lower = []
    for point in data:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    
    # Build the upper hull
    upper = []
    for point in reversed(data):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    
    # Remove the last point of each half because it is repeated at the beginning of the other half
    return np.array(lower[:-1] + upper[:-1])

# Call the function to get the convex hull
hull = convex_hull_monotone_chain(points)

# Plot the points and the convex hull
plt.figure(figsize=(6, 6))
plt.plot(points[:, 0], points[:, 1], 'o', label="Points")
plt.plot(hull[:, 0], hull[:, 1], 'r-', label="Convex Hull")

# Close the convex hull polygon
plt.fill(hull[:, 0], hull[:, 1], 'r', alpha=0.3)

# Add labels and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Convex Hull using Monotone Chain")
plt.legend()

# Save the plot to the current directory
plt.savefig("monotone_chain_convex_hull.png")

# Show the plot
plt.show()