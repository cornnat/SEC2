import numpy as np
import matplotlib.pyplot as plt

#2 part a
# n-point uniform point cloud

def generate_point_cloud(n, filename):
    # Generate random points
    points = np.random.rand(n, 2)
    # Plot the points
    plt.scatter(points[:, 0], points[:, 1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Point Cloud")
    
    # Save the plot as a PNG file with the given filename
    plt.savefig(f"{filename}.png")
    plt.show()
    
    # Save the points to a .dat file with the same filename
    np.savetxt(f"{filename}.dat", points, delimiter=" ", header="X Y", comments="")
    
    print(f"Point cloud saved as {filename}.dat and {filename}.png")

# Call the function
generate_point_cloud(10, "pointcloud10")

generate_point_cloud(50, "pointcloud50")

generate_point_cloud(100, "pointcloud100")

generate_point_cloud(200, "pointcloud200")

generate_point_cloud(400, "pointcloud400")

generate_point_cloud(800, "pointcloud800")

generate_point_cloud(1000, "pointcloud1000")

