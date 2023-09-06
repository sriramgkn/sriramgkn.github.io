import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

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

# Plot the projections on the planes x=0, y=0, and z=0
ax.plot_surface(np.zeros_like(y), y, z, color='lightblue', alpha=0.6)
ax.plot_surface(x, np.zeros_like(x), z, color='lightblue', alpha=0.6)
ax.plot_surface(x, y, np.zeros_like(z), color='lightblue', alpha=0.6)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Sphere with Projections and Interactive Controls')

# Set the aspect ratio of the axes
ax.set_box_aspect([1, 1, 1])  # Make axes equal in scale

# Set the limits of the axes
max_radius = max(radius, np.max(np.abs(center)))
ax.set_xlim(center[0] - max_radius, center[0] + max_radius)
ax.set_ylim(center[1] - max_radius, center[1] + max_radius)
ax.set_zlim(center[2] - max_radius, center[2] + max_radius)

# Create sliders for rotation and deformation
ax_rot_x = plt.axes([0.2, 0.1, 0.6, 0.03])
slider_rot_x = Slider(ax_rot_x, 'Rotation X', -10, 10, valinit=0)

ax_rot_y = plt.axes([0.2, 0.05, 0.6, 0.03])
slider_rot_y = Slider(ax_rot_y, 'Rotation Y', -10, 10, valinit=1)

ax_rot_z = plt.axes([0.2, 0.0, 0.6, 0.03])
slider_rot_z = Slider(ax_rot_z, 'Rotation Z', -10, 10, valinit=1)

ax_deform = plt.axes([0.2, 0.15, 0.6, 0.03])
slider_deform = Slider(ax_deform, 'Deformation', 0.1, 2.0, valinit=1.7)

def update(val):
    rot_x = np.deg2rad(slider_rot_x.val)
    rot_y = np.deg2rad(slider_rot_y.val)
    rot_z = np.deg2rad(slider_rot_z.val)
    deform = slider_deform.val

    # Apply rotation and deformation to the sphere
    x_rot = center[0] + deform * radius * np.sin(v) * np.cos(u + rot_x)
    y_rot = center[1] + deform * radius * np.sin(v) * np.sin(u + rot_y)
    z_rot = center[2] + deform * radius * np.cos(v + rot_z)
    sphere.set_verts([x_rot, y_rot, z_rot])

    fig.canvas.draw_idle()

slider_rot_x.on_changed(update)
slider_rot_y.on_changed(update)
slider_rot_z.on_changed(update)
slider_deform.on_changed(update)

# Draw the axes and origin marker
ax.quiver(0, 0, 0, max_radius, 0, 0, color='black', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, max_radius, 0, color='black', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, max_radius, color='black', arrow_length_ratio=0.1)
ax.text(0, 0, 0, 'O', ha='center', va='center')

plt.show()
