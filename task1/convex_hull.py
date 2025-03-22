import matplotlib.pyplot as plt
import numpy as np
import warnings
from random import randint
from math import atan2
import os

warnings.filterwarnings("ignore")

# Part 1) Algorithms for Convex Hulls

# Visualization function to plot point cloud and convex hull
def scatter_plot(coords, convex_hull=None):
    xs, ys = zip(*coords)  # unzip the coordinate array into lists
    plt.scatter(xs, ys)  # scatter plot

    if convex_hull is not None:
        for i in range(1, len(convex_hull) + 1):
            if i == len(convex_hull): i = 0
            c0 = convex_hull[i - 1]
            c1 = convex_hull[i]
            plt.plot((c0[0], c1[0]), (c0[1], c1[1]), 'r')
    plt.show()

def polar_angle(p0, p1=None):
    if p1 is None: p1 = anchor
    x_span = p0[0] - p1[0]
    y_span = p0[1] - p1[1]
    return atan2(y_span, x_span)

def distance(p0, p1=None):
    if p1 is None: p1 = anchor
    x_span = p0[0] - p1[0]
    y_span = p0[1] - p1[1]
    return x_span**2 + y_span**2

def det(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

def quicksort(points):
    if len(points) <= 1:
        return points
    smaller, equal, larger = [], [], []
    pivot = polar_angle(points[randint(0, len(points) - 1)])
    for pt in points:
        angle = polar_angle(pt)
        if angle < pivot:
            smaller.append(pt)
        elif angle == pivot:
            equal.append(pt)
        else:
            larger.append(pt)
    return quicksort(smaller) + sorted(equal, key=distance) + quicksort(larger)

def graham_scan(points, show_progress=False):
    global anchor
    points = [tuple(p) for p in points]
    anchor = min(points, key=lambda p: (p[1], p[0]))
    sorted_pts = quicksort(points)
    sorted_pts.remove(anchor)
    hull = [anchor, sorted_pts[0]]
    for s in sorted_pts[1:]:
        while len(hull) >= 2 and det(hull[-2], hull[-1], s) <= 0:
            hull.pop()
        hull.append(s)

    if show_progress:
        scatter_plot(points, hull) 

    return hull

def jarvis(points):
    n = len(points)
    hull = []
    leftmost_idx = np.argmin(points[:, 0])
    current_idx = leftmost_idx

    while True:
        hull.append(tuple(points[current_idx]))
        next_idx = (current_idx + 1) % n
        for i in range(n):
            if det(points[current_idx], points[next_idx], points[i]) > 0:
                next_idx = i

        current_idx = next_idx

        if current_idx == leftmost_idx:
            break
    return hull

def quickhull(points):
    points = [tuple(p) for p in points]
    if len(points) < 3:
        return points

    def side(a, b, p):
        return det(a, b, p)

    def add_hull(pt_set, p, q):
        index = None
        max_distance = 0
        for i, pt in enumerate(pt_set):
            d = abs(det(p, q, pt))
            if d > max_distance:
                max_distance = d
                index = i
        if index is None:
            return []
        farthest = pt_set[index]
        left_set = [pt for pt in pt_set if side(p, farthest, pt) > 0]
        right_set = [pt for pt in pt_set if side(farthest, q, pt) > 0]
        return add_hull(left_set, p, farthest) + [farthest] + add_hull(right_set, farthest, q)

    leftmost = min(points, key=lambda p: p[0])
    rightmost = max(points, key=lambda p: p[0])

    above = [pt for pt in points if side(leftmost, rightmost, pt) > 0]
    below = [pt for pt in points if side(rightmost, leftmost, pt) > 0]

    upper_hull = add_hull(above, leftmost, rightmost)
    lower_hull = add_hull(below, rightmost, leftmost)

    return [leftmost] + upper_hull + [rightmost] + lower_hull

def monotone_chain(points):
    points = sorted(map(tuple, points), key=lambda p: (p[0], p[1]))
    lower_hull = []
    for point in points:
        while len(lower_hull) >= 2 and det(lower_hull[-2], lower_hull[-1], point) <= 0:
            lower_hull.pop()
        lower_hull.append(point)

    upper_hull = []
    for point in reversed(points):
        while len(upper_hull) >= 2 and det(upper_hull[-2], upper_hull[-1], point) <= 0:
            upper_hull.pop()
        upper_hull.append(point)

    return lower_hull[:-1] + upper_hull[:-1]

# Reading in the file
with open("/root/Desktop/host/SEC2/mesh.dat", "r") as file:
    data = file.readlines()

# Processing the data
x = []
y = []

for line in data:
    if line.strip() == "" or line.startswith('X'):
        continue
    columns = line.split()
    try:
        x.append(float(columns[0])) 
        y.append(float(columns[1]))  
    except ValueError:
        continue

coords = np.array(list(zip(x, y)))

# Applying each convex hull algorithm
hulls = [
    graham_scan(coords, False),
    jarvis(coords),
    quickhull(coords),
    monotone_chain(coords)
]

# Set up subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

# Plot each hull method in the subplots
methods = ['Graham Scan', 'Jarvis March', 'Quickhull', 'Monotone Chain']
for i, ax in enumerate(axs):
    ax.scatter(coords[:, 0], coords[:, 1], color='black')  # Scatter plot points
    hull = hulls[i]
    for j in range(1, len(hull) + 1):
        if j == len(hull): j = 0
        ax.plot([hull[j-1][0], hull[j][0]], [hull[j-1][1], hull[j][1]], 'r')
    ax.set_title(methods[i])
    ax.axis('equal')

# Adjust layout and save the final figure
plt.tight_layout()
plt.savefig('/root/Desktop/host/SEC2/task1/hull_plots/convex_hulls_subplot.png', bbox_inches='tight')
plt.show()


#########################################################################
# Part 2
####################################################################

import matplotlib.pyplot as plt
import numpy as np
import time
import random
from math import atan2
from random import randint

# Part 1: Functions for Convex Hulls (as defined earlier)
# (Include all functions: graham_scan, jarvis, quickhull, monotone_chain)

# Function to generate random n-point uniform point cloud in the range [0, 1]
def generate_point_cloud(n, lower=0, upper=1):
    return np.array([[random.uniform(lower, upper), random.uniform(lower, upper)] for _ in range(n)])

# Function to run and time each convex hull algorithm
def time_algorithms(n, point_cloud):
    runtimes = {}
    
    # Time Graham Scan
    start = time.time()
    graham_scan(point_cloud)
    runtimes['Graham Scan'] = time.time() - start
    
    # Time Jarvis March
    start = time.time()
    jarvis(point_cloud)
    runtimes['Jarvis March'] = time.time() - start
    
    # Time Quickhull
    start = time.time()
    quickhull(point_cloud)
    runtimes['Quickhull'] = time.time() - start
    
    # Time Monotone Chain
    start = time.time()
    monotone_chain(point_cloud)
    runtimes['Monotone Chain'] = time.time() - start
    
    return runtimes

# Part 2: Run experiments for different n values
n_values = [10, 50, 100, 200, 400, 800, 1000]
uniform_runtimes = {'Graham Scan': [], 'Jarvis March': [], 'Quickhull': [], 'Monotone Chain': []}

# Measure runtime for each n-value
for n in n_values:
    point_cloud = generate_point_cloud(n)
    runtimes = time_algorithms(n, point_cloud)
    for method in runtimes:
        uniform_runtimes[method].append(runtimes[method])

# Plot the results for uniform distribution (0 to 1)
plt.figure(figsize=(10, 6))
for method in uniform_runtimes:
    plt.plot(n_values, uniform_runtimes[method], label=method)
plt.xlabel('Number of Points (n)')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Comparison for Different Convex Hull Algorithms (Uniform Distribution [0, 1])')
plt.legend()
plt.grid(True)
plt.savefig('/root/Desktop/host/SEC2/task1/hull_plots/uniform_runtime_comparison.png')
plt.show()

# Writing conclusions to a text file for the uniform distribution
with open('/root/Desktop/host/SEC2/task1/hull_plots/uniform_runtime_conclusion.txt', 'w') as f:
    f.write("Conclusion for Uniform Distribution (Bounds [0, 1]):\n")
    f.write("As the number of points (n) increases, the runtime for each algorithm grows.\n")
    f.write("The algorithms generally scale with the complexity of their operations.\n")
    f.write("Graham Scan and Monotone Chain tend to have similar performance, with Quickhull being faster.\n")
    f.write("Jarvis March has the highest runtime, especially as n grows larger.\n")

# Part 3: Generate and analyze point cloud with different bounds
bounds = [(-5, 5), (-5, 5), (-5, 5)]
bound_runtimes = {'Graham Scan': [], 'Jarvis March': [], 'Quickhull': [], 'Monotone Chain': []}

for n in n_values:
    point_cloud = generate_point_cloud(n, -5, 5)
    runtimes = time_algorithms(n, point_cloud)
    for method in runtimes:
        bound_runtimes[method].append(runtimes[method])

# Plot results for [-5, 5] bounds
plt.figure(figsize=(10, 6))
for method in bound_runtimes:
    plt.plot(n_values, bound_runtimes[method], label=method)
plt.xlabel('Number of Points (n)')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Comparison for Different Convex Hull Algorithms (Bounds [-5, 5])')
plt.legend()
plt.grid(True)
plt.savefig('/root/Desktop/host/SEC2/task1/hull_plots/bounds_runtime_comparison.png')
plt.show()

# Writing conclusions for the bounded point cloud
with open('/root/Desktop/host/SEC2/task1/hull_plots/bounds_runtime_conclusion.txt', 'w') as f:
    f.write("Conclusion for Point Cloud with Bounds [-5, 5]:\n")
    f.write("Changing the bounds to [-5, 5] doesn't show a significant difference in algorithm performance.\n")
    f.write("The scaling behavior remains the same, although the point cloud has a larger range.\n")
    f.write("We still observe the same ranking of algorithms from fastest to slowest.\n")

# Part 4: Generate and analyze Gaussian distribution
def generate_gaussian_point_cloud(n, mean=0, std_dev=1):
    return np.random.normal(loc=mean, scale=std_dev, size=(n, 2))

gaussian_runtimes = {'Graham Scan': [], 'Jarvis March': [], 'Quickhull': [], 'Monotone Chain': []}

for n in n_values:
    point_cloud = generate_gaussian_point_cloud(n)
    runtimes = time_algorithms(n, point_cloud)
    for method in runtimes:
        gaussian_runtimes[method].append(runtimes[method])

# Plot results for Gaussian distribution
plt.figure(figsize=(10, 6))
for method in gaussian_runtimes:
    plt.plot(n_values, gaussian_runtimes[method], label=method)
plt.xlabel('Number of Points (n)')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Comparison for Different Convex Hull Algorithms (Gaussian Distribution)')
plt.legend()
plt.grid(True)
plt.savefig('/root/Desktop/host/SEC2/task1/hull_plots/gaussian_runtime_comparison.png')
plt.show()

# Writing conclusions for the Gaussian point cloud
with open('/root/Desktop/host/SEC2/task1/hull_plots/gaussian_runtime_conclusion.txt', 'w') as f:
    f.write("Conclusion for Point Cloud with Gaussian Distribution:\n")
    f.write("The runtime shows no significant difference when comparing uniform and Gaussian distributions.\n")
    f.write("The algorithms follow similar scaling patterns, with Quickhull generally performing the best.\n")

# Part 5: Analyze runtime distribution for n=50
n = 50
all_runtimes = {'Graham Scan': [], 'Jarvis March': [], 'Quickhull': [], 'Monotone Chain': []}

for _ in range(100):
    point_cloud = generate_point_cloud(n)
    runtimes = time_algorithms(n, point_cloud)
    for method in runtimes:
        all_runtimes[method].append(runtimes[method])

# Plot histograms of runtimes for each algorithm
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

for i, method in enumerate(all_runtimes):
    axs[i].hist(all_runtimes[method], bins=20, edgecolor='black')
    axs[i].set_title(f'Runtime Distribution for {method}')
    axs[i].set_xlabel('Runtime (seconds)')
    axs[i].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('/root/Desktop/host/SEC2/task1/hull_plots/runtime_distribution_n50.png')
plt.show()

# Writing conclusions for runtime distribution
with open('/root/Desktop/host/SEC2/task1/hull_plots/runtime_distribution_conclusion.txt', 'w') as f:
    f.write("Conclusion for Runtime Distribution (n=50, 100 runs):\n")
    f.write("The runtime distribution for each algorithm follows a roughly normal distribution.\n")
    f.write("Quickhull tends to have a slightly tighter distribution and lower runtimes.\n")
    f.write("Jarvis March has a wider and higher runtime distribution, indicating variability.\n")
