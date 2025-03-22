#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Create directory to save plots, if it doesn't already exist.
if not os.path.exists("plots"):
    os.makedirs("plots")

# ----------------------------
# Part 1: Define the Surfaces
# ----------------------------

def bottom_surface(x, y):
    """Bottom surface (parabolic)."""
    return 2 * x**2 + 2 * y**2

def top_surface(x, y):
    """Top surface (exponential)."""
    return 2 * np.exp(-x**2 - y**2)

# Function to find boundary radius where the surfaces meet.
def compute_boundary_radius():
    equation = lambda r: r**2 - np.exp(-r**2)
    r_boundary = fsolve(equation, 0.7)[0]
    return r_boundary

# Generate grid points within a disk of radius R.
def generate_disk_points(R, num_points=50):
    x_vals = np.linspace(-R, R, num_points)
    y_vals = np.linspace(-R, R, num_points)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    x_grid, y_grid = x_grid.flatten(), y_grid.flatten()
    mask = x_grid**2 + y_grid**2 <= R**2
    return np.vstack([x_grid[mask], y_grid[mask]]).T

# Generate point clouds for top and bottom surfaces.
def generate_surface_cloud(points):
    z_top = top_surface(points[:, 0], points[:, 1])
    z_bottom = bottom_surface(points[:, 0], points[:, 1])
    return np.hstack([points, z_top[:, None]]), np.hstack([points, z_bottom[:, None]])

# ----------------------------
# Part 2: Delaunay Triangulation
# ----------------------------

def delaunay_triangulation():
    # Compute boundary radius and generate grid points.
    R = compute_boundary_radius()
    grid_points = generate_disk_points(R, num_points=50)
    
    # Create top and bottom surface point clouds.
    top_points, bottom_points = generate_surface_cloud(grid_points)
    n = grid_points.shape[0]  # Number of grid points
    
    # Perform Delaunay triangulation on (x,y) projection.
    triangulation = Delaunay(grid_points)
    
    # Identify boundary points where r is close to R.
    boundary_indices = [i for i, pt in enumerate(grid_points)
                        if np.isclose(np.sqrt(pt[0]**2 + pt[1]**2), R, atol=1e-3)]
    boundary_set = set(boundary_indices)
    
    # Identify interior points (not on boundary).
    interior_indices = [i for i in range(n) if i not in boundary_set]
    
    # Create the combined vertex list: top surface and interior bottom surface.
    vertices_top = top_points
    vertices_bottom_interior = bottom_points[interior_indices]
    combined_vertices = np.vstack((vertices_top, vertices_bottom_interior))
    
    # Map bottom surface vertices: boundary points keep same index, interior points get new index.
    bottom_mapping = {}
    for j, i in enumerate(interior_indices):
        bottom_mapping[i] = n + j
    for i in boundary_indices:
        bottom_mapping[i] = i
    
    # Re-index bottom surface triangles using mapping.
    bottom_triangles = []
    for triangle in triangulation.simplices:
        new_triangle = [bottom_mapping[i] for i in triangle]
        bottom_triangles.append(new_triangle)
    bottom_triangles = np.array(bottom_triangles)
    
    # Top surface triangles use original indices.
    top_triangles = triangulation.simplices
    
    # Combine both surface triangles.
    combined_triangles = np.vstack((top_triangles, bottom_triangles))
    
    # Visualize the closed surface mesh.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(combined_vertices[:, 0], combined_vertices[:, 1],
                    combined_vertices[:, 2],
                    triangles=combined_triangles, cmap='viridis',
                    edgecolor='none', alpha=0.8)
    ax.set_title("Closed Surface Triangulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig("plots/full_triangulation.png", dpi=300)
    plt.show()

# ----------------------------
# Part 3: Volume Mesh Triangulation
# ----------------------------

def volume_mesh_triangulation():
    """
    Generate a 3D volume mesh using Delaunay triangulation.
    Sample points between the bottom and top surfaces for each (x,y) point in a grid.
    Perform 3D triangulation to create tetrahedra and visualize the volume mesh boundary.
    """
    # Get boundary radius.
    R = compute_boundary_radius()
    
    # Generate lower resolution grid for volume mesh visualization.
    num_xy = 15
    xy_points = generate_disk_points(R, num_points=num_xy)
    
    # Build 3D point cloud by sampling along z between bottom and top surfaces.
    volume_points = []
    num_z = 4  # Number of z-levels.
    for pt in xy_points:
        x, y = pt
        z_bottom = bottom_surface(x, y)
        z_top = top_surface(x, y)
        z_values = np.linspace(z_bottom, z_top, num_z)
        for z in z_values:
            volume_points.append([x, y, z])
    volume_points = np.array(volume_points)
    
    # Perform 3D Delaunay triangulation on volume points.
    triangulation_3d = Delaunay(volume_points)
    
    # Extract tetrahedra.
    tetrahedra = triangulation_3d.simplices
    
    # Extract faces from tetrahedra.
    faces = {}
    for tet in tetrahedra:
        # Each tetrahedron has 4 faces.
        for face in [(tet[0], tet[1], tet[2]),
                     (tet[0], tet[1], tet[3]),
                     (tet[0], tet[2], tet[3]),
                     (tet[1], tet[2], tet[3])]:
            face_sorted = tuple(sorted(face))
            faces[face_sorted] = faces.get(face_sorted, 0) + 1
    
    # Boundary faces appear once, interior faces appear twice.
    boundary_faces = [face for face, count in faces.items() if count == 1]
    
    # Visualize the volume mesh boundary surface.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a collection for boundary triangles.
    triangles = [volume_points[list(face)] for face in boundary_faces]
    mesh = Poly3DCollection(triangles, facecolor='cyan', edgecolor='gray', alpha=0.5)
    ax.add_collection3d(mesh)
    
    # Optionally, scatter plot the volume points.
    ax.scatter(volume_points[:, 0], volume_points[:, 1], volume_points[:, 2], color='red', s=10)
    
    ax.set_title("Volume Mesh Triangulation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Set axis limits.
    ax.set_xlim(np.min(volume_points[:, 0]), np.max(volume_points[:, 0]))
    ax.set_ylim(np.min(volume_points[:, 1]), np.max(volume_points[:, 1]))
    ax.set_zlim(np.min(volume_points[:, 2]), np.max(volume_points[:, 2]))
    
    plt.savefig("plots/volume_mesh.png", dpi=300)
    plt.show()

# ----------------------------
# Part 4: Surface Mesh from Volume Mesh
# ----------------------------

def surface_mesh_from_volume_mesh():
    """
    Generate a surface mesh from a volume mesh and visualize it.
    Perform the same steps as Part 3 but focusing on extracting the surface from the volume.
    """
    # Get boundary radius.
    R = compute_boundary_radius()
    
    # Use lower resolution grid for volume point generation.
    num_xy = 15
    xy_points = generate_disk_points(R, num_points=num_xy)
    
    # Build 3D point cloud by sampling along z between the bottom and top surfaces.
    volume_points = []
    num_z = 4
    for pt in xy_points:
        x, y = pt
        z_bottom = bottom_surface(x, y)
        z_top = top_surface(x, y)
        z_values = np.linspace(z_bottom, z_top, num_z)
        for z in z_values:
            volume_points.append([x, y, z])
    volume_points = np.array(volume_points)
    
    # Perform 3D Delaunay triangulation.
    triangulation_3d = Delaunay(volume_points)
    tetrahedra = triangulation_3d.simplices
    
    # Extract faces from the tetrahedra.
    face_count = {}
    for tet in tetrahedra:
        for face in [(tet[0], tet[1], tet[2]),
                     (tet[0], tet[1], tet[3]),
                     (tet[0], tet[2], tet[3]),
                     (tet[1], tet[2], tet[3])]:
            face_sorted = tuple(sorted(face))
            face_count[face_sorted] = face_count.get(face_sorted, 0) + 1
    
    # Boundary faces.
    boundary_faces = [face for face, count in face_count.items() if count == 1]
    boundary_faces = np.array(boundary_faces)
    
    # Visualize surface mesh using plot_trisurf.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_trisurf(volume_points[:, 0], volume_points[:, 1], volume_points[:, 2],
                    triangles=boundary_faces, cmap='plasma', edgecolor='none', alpha=0.8)
    
    ax.set_title("Surface Mesh from Volume Mesh")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.savefig("plots/surface_from_volume_mesh.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    # Execute the steps.
    delaunay_triangulation()
    volume_mesh_triangulation()
    surface_mesh_from_volume_mesh()
