import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.ticker import FuncFormatter

# Function to create rotation matrices
def rotation_matrix_x(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])

def rotation_matrix_y(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])

def rotation_matrix_z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

# Sphere parameters
radius = 2
center = (3, 3, 3)

# Generate the sphere surface
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)
x = center[0] + radius * np.sin(v) * np.cos(u)
y = center[1] + radius * np.sin(v) * np.sin(u)
z = center[2] + radius * np.cos(v)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
sphere = ax.plot_surface(x, y, z, color='lightgreen')

# Adjust the dimensionality of projection surfaces
projection_x_original = np.stack([np.zeros_like(y), y, z])
projection_y_original = np.stack([x, np.zeros_like(x), z])
projection_z_original = np.stack([x, y, np.zeros_like(z)])

# Initialize projection variables
projection_x = None
projection_y = None
projection_z = None

# Plot the projections on the planes x=0, y=0, and z=0
projection_x = ax.plot_surface(np.zeros_like(y), y, z, color='lightblue', alpha=0.6)
projection_y = ax.plot_surface(x, np.zeros_like(x), z, color='lightblue', alpha=0.6)
projection_z = ax.plot_surface(x, y, np.zeros_like(z), color='lightblue', alpha=0.6)

rotation_angles = [0, 0, 0]  # Initialize rotation angles around x, y, and z axes

def update_plot(val):
    # Retrieve the rotation angles from the slider values and convert to radians
    rotation_angles[0] = np.radians(slider_x.val)
    rotation_angles[1] = np.radians(slider_y.val)
    rotation_angles[2] = np.radians(slider_z.val)

    # Apply rotations to the projection surfaces
    rotated_projection_x = np.dot(rotation_matrix_x(rotation_angles[0]), projection_x_original)
    rotated_projection_y = np.dot(rotation_matrix_y(rotation_angles[1]), projection_y_original)
    rotated_projection_z = np.dot(rotation_matrix_z(rotation_angles[2]), projection_z_original)

    # Update the projection surfaces in the plot
    projection_x.set_verts(rotated_projection_x)
    projection_y.set_verts(rotated_projection_y)
    projection_z.set_verts(rotated_projection_z)

    # Redraw the plot
    fig.canvas.draw_idle()

# Set up the sliders
ax_slider_x = plt.axes([0.15, 0.05, 0.65, 0.03])
ax_slider_y = plt.axes([0.15, 0.02, 0.65, 0.03])
ax_slider_z = plt.axes([0.15, 0.09, 0.65, 0.03])

# Set the degree format for the sliders
ax_slider_x.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}°'))
ax_slider_y.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}°'))
ax_slider_z.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}°'))

slider_x = Slider(ax_slider_x, 'Rotation X', -90, 90, valinit=0, valstep=10)
slider_y = Slider(ax_slider_y, 'Rotation Y', -90, 90, valinit=0, valstep=10)
slider_z = Slider(ax_slider_z, 'Rotation Z', -90, 90, valinit=0, valstep=10)

# Connect the sliders to the update function
slider_x.on_changed(update_plot)
slider_y.on_changed(update_plot)
slider_z.on_changed(update_plot)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Visualizing shadows: Loomis-Whitney analogy')

# Set the aspect ratio of the axes
ax.set_box_aspect([1, 1, 1])  # Make axes equal in scale

# Set the limits of the axes
max_radius = max(radius, np.max(np.abs(center)))
ax.set_xlim(center[0] - max_radius, center[0] + max_radius)
ax.set_ylim(center[1] - max_radius, center[1] + max_radius)
ax.set_zlim(center[2] - max_radius, center[2] + max_radius)

# Draw the axes and origin marker
ax.quiver(0, 0, 0, max_radius, 0, 0, color='black', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, max_radius, 0, color='black', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, max_radius, color='black', arrow_length_ratio=0.1)
ax.text(0, 0, 0, 'O', ha='center', va='center')

plt.show()
