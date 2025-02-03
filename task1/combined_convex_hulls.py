import numpy as np
import matplotlib.pyplot as plt

# Load the data
points = np.loadtxt('/root/Desktop/host/SEC2/mesh.dat', skiprows=1)

# Define the convex hull algorithms

# 1. Monotone Chain Convex Hull
def convex_hull_monotone_chain(data):
    data = data[np.argsort(data[:, 0])]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for point in data:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    
    upper = []
    for point in reversed(data):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    
    return np.array(lower[:-1] + upper[:-1])

# 2. Jarvis March (Gift Wrapping) Convex Hull
def convex_hull_jarvis_march(data):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    leftmost = min(data, key=lambda p: p[0])
    hull = []
    point = leftmost

    while True:
        hull.append(point)
        next_point = data[0]
        for candidate in data:
            if np.array_equal(candidate, point):
                continue
            if cross(point, next_point, candidate) > 0:
                next_point = candidate
        point = next_point
        if np.array_equal(point, leftmost):
            break

    return np.array(hull)

# 3. Graham Scan Convex Hull
def graham_scan(points):
    def polar_angle(p0, p1=None):
        if p1 is None: p1 = anchor
        y_span = p0[1] - p1[1]
        x_span = p0[0] - p1[0]
        return np.arctan2(y_span, x_span)
    
    def distance(p0, p1=None):
        if p1 is None: p1 = anchor
        y_span = p0[1] - p1[1]
        x_span = p0[0] - p1[0]
        return y_span ** 2 + x_span ** 2
    
    def det(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    min_idx = None
    for i, (x, y) in enumerate(points):
        if min_idx is None or y < points[min_idx][1]:
            min_idx = i
        if y == points[min_idx][1] and x < points[min_idx][0]:
            min_idx = i
    anchor = tuple(points[min_idx])
    sorted_pts = sorted(points, key=lambda p: (polar_angle(p), distance(p)))
    
    # Convert sorted_pts to a numpy array for easier removal
    sorted_pts = np.array(sorted_pts)
    
    # Ensure anchor is removed correctly
    anchor_idx = np.where(np.all(sorted_pts == anchor, axis=1))[0][0]
    sorted_pts = np.delete(sorted_pts, anchor_idx, axis=0)
    
    hull = [anchor, sorted_pts[0]]
    for s in sorted_pts[1:]:
        while det(hull[-2], hull[-1], s) <= 0:
            del hull[-1]
            if len(hull) < 2: break
        hull.append(s)

    return np.array(hull)

# Call each convex hull function
hull_monotone_chain = convex_hull_monotone_chain(points)
hull_jarvis_march = convex_hull_jarvis_march(points)
hull_graham_scan = graham_scan(points)

# Plot the point cloud and the convex hulls
plt.figure(figsize=(8, 8))
plt.scatter(points[:, 0], points[:, 1], label="Point Cloud", color='black', s=10)

# Plot each convex hull in different colors
plt.plot(hull_monotone_chain[:, 0], hull_monotone_chain[:, 1], 'r-', label="Monotone Chain Hull")
plt.fill(hull_monotone_chain[:, 0], hull_monotone_chain[:, 1], 'r', alpha=0.3)

plt.plot(hull_jarvis_march[:, 0], hull_jarvis_march[:, 1], 'g-', label="Jarvis March Hull")
plt.fill(hull_jarvis_march[:, 0], hull_jarvis_march[:, 1], 'g', alpha=0.3)

plt.plot(hull_graham_scan[:, 0], hull_graham_scan[:, 1], 'b-', label="Graham Scan Hull")
plt.fill(hull_graham_scan[:, 0], hull_graham_scan[:, 1], 'b', alpha=0.3)

# Add labels, title, and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Convex Hulls Visualized in Different Colors")
plt.legend()

# Show the plot
plt.savefig("combined_convex_hull_plot.png")
plt.show()
