import numpy as np
import matplotlib.pyplot as plt

#read the data file
data = np.loadtxt('mesh.dat', skiprows = 1)
x = data[:, 0]  # First column is x
y = data[:, 1]  # Second column is y

#visualizing the data - plotting data and saving fig
plt.scatter(x, y, s=1)  # 's' controls the size of the points
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Point Cloud Visualization')
plt.savefig('mesh_visualized_test.png', format='png') 
plt.show()
