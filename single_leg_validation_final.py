# Perfect single leg simulation in z height sim/ani separated

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import time
import pandas as pd

# ---------------------------
# Simulation Parameters
# ---------------------------
L = 0.190        # Total rod length (meters)
N = 31           # Number of nodes (higher N for smoother curve)
dx = L / (N - 1) # Nominal segment length

# Thickness tapering from fixed end to tip
thickness_fixed = 0.0135   # Thickness at fixed end (meters)
thickness_tip = 0.0035     # Thickness at tip (meters)
thickness = thickness_fixed + (thickness_tip - thickness_fixed) * np.linspace(0, 1, N)

# Initial Shape: Cubic function to curve upwards then downwards
a = 6.356
b = -1.810
c = 0.0
d = 0.02175

x = np.linspace(0, L, N)
z = a * x**3 + b * x**2 + c * x + d  # Height is now z
y = np.zeros_like(x)                 # Rod lies in the XZ plane

# Store the initial configuration
initial_x = np.copy(x)
initial_z = np.copy(z)

# Compute Normals to the Centerline
dxs = np.gradient(x)
dzs = np.gradient(z)
lengths = np.sqrt(dxs**2 + dzs**2)
nx = -dzs / lengths  # Normal vector components (pointing outward)
nz = dxs / lengths

# Compute Top and Bottom Edges of the Rod
top_x = x + (thickness / 2) * nx
top_z = z + (thickness / 2) * nz
bottom_x = x - (thickness / 2) * nx
bottom_z = z - (thickness / 2) * nz

# Create Polygon Points: Top edge followed by reversed bottom edge
rod_x = np.concatenate([top_x, bottom_x[::-1]])
rod_z = np.concatenate([top_z, bottom_z[::-1]])

# Pulley Parameters
theta_deg = 168  # Desired pulling angle in degrees
theta_angle = np.deg2rad(theta_deg)  # Convert to radians
R0 = 0.03  # Initial tendon length (meters)

# Initialize Pulley Position Based on Fixed End and Pulling Angle
x0 = x[0]
z0 = z[0]
px = x0 + R0 * np.cos(theta_angle)  # Pulley x-position
pz = z0 + R0 * np.sin(theta_angle)  # Pulley z-position

# Material Properties
E_default = 1e6          # Young's modulus for flexible TPU (Pascals)
rho = 1200.0             # Density (kg/m³)

# Compute Cross-Sectional Area and Moment of Inertia for each segment
width = 0.02     # Assume constant width of 2 cm
A = thickness * width  # Cross-sectional area (m²)
I = (width * thickness**3) / 12.0  # Second moment of area (m⁴)

# Initialize Young's Modulus array with default E
E = np.full(N-1, E_default)

# Mass per node
m = np.zeros(N)
m[1:-1] = (rho * A[:-2] * dx + rho * A[1:-1] * dx) / 2
m[0] = rho * A[0] * dx / 2
m[-1] = rho * A[-1] * dx / 2

# Gravity
g = 9.81  # m/s²

# Friction at tip
mu = 0.1  # friction coefficient
tip_index = N - 1

# Tendon Parameters
angle_deg = 116.28
damp_c = 0.1

dt = 1e-4  # Time step
simulation_time = 8  # Total simulation time (s)
time_steps = int(simulation_time / dt)

# Contact (Ground Penalty) Parameters
k_contact = 1e5   # Ground stiffness (N/m)
d_contact = 10.0  # Ground damping (Ns/m)

# Initialize State Variables
theta_state = np.zeros(N)  # Angles (radians)
vx = np.zeros(N)  # x-velocities (m/s)
vz = np.zeros(N)  # z-velocities (m/s)
omega = np.zeros(N)  # Angular velocities (rad/s)

reaction_forces = []

# ---------------------------
# Tendon Tension Function
# ---------------------------
def angle_to_tendon_force(angle_deg, max_angle_deg=270, max_torque=1.5, servo_radius=0.01):
    """
    Convert servo angle (degrees) to tendon force using linear mapping.
    - angle_deg: current angle of servo (can be a float or array)
    - max_angle_deg: maximum allowed servo rotation
    - max_torque: maximum torque at max_angle (Nm)
    - servo_radius: pulley radius in meters
    Returns tendon force in Newtons.
    """
    torque = (angle_deg / max_angle_deg) * max_torque
    force = torque / servo_radius
    return force

def tendon_tension(t, z, angle_deg):
    # Timing parameters
    t_start = 1.0239501
    ramp_duration = 2.4
    hold_duration = 0.2
    decay_duration = 2.0

    t_ramp_end = t_start + ramp_duration
    t_hold_end = t_ramp_end + hold_duration
    t_decay_end = t_hold_end + decay_duration

    # Exit if lifted
    # if isinstance(z, np.ndarray) and z[N-1] - thickness[N-1] > 0:
    #     return 0.0
    if t < t_start or t > t_decay_end:
        return 0.0

    # Flexible angle profile
    if t_start <= t < t_ramp_end:
        angle_deg = angle_deg * np.sin(np.pi * (t - t_start) / (2 * ramp_duration))
    elif t_ramp_end <= t < t_hold_end:
        angle_deg = angle_deg
    elif t_hold_end <= t < t_decay_end:
        angle_deg = angle_deg * np.cos(np.pi * (t - t_hold_end) / (2 * decay_duration))
    else:
        angle_deg = 0.0

    return angle_to_tendon_force(angle_deg)

def friction_forces(x, z, vx, vz):
    Fx_f = np.zeros(N)
    Fz_f = np.zeros(N)
    for i in range(N):
        if z[i] <= 0.0:
            N_load = m[i] * g
            if abs(vx[i]) > 1e-12:
                Fx_f[i] = -mu*N_load*np.sign(vx[i])
    return Fx_f, Fz_f

def contact_forces(z, vz):
    Fz_contact = np.zeros(N)
    for i in range(N):
        if z[i] < 0.0:
            penetration = -z[i]
            damping = -vz[i]
            Fz_contact[i] = k_contact * penetration + d_contact * damping
    return Fz_contact

from numba import njit

@njit
def compute_internal_forces(x, z, theta_state):
    Fx = np.zeros(N)
    Fz = np.zeros(N)
    My = np.zeros(N)
    for i in range(N - 1):
        dx_i = x[i + 1] - x[i]
        dz_i = z[i + 1] - z[i]
        l_i = np.sqrt(dx_i**2 + dz_i**2)
        eps = (l_i - dx) / dx
        seg_angle = np.arctan2(dz_i, dx_i)
        if i < N - 2:
            dtheta = theta_state[i + 1] - theta_state[i]
        else:
            dtheta = 0.0
        f_axial = E[i] * A[i] * eps
        m_bend = E[i] * I[i] * dtheta / dx
        Fx[i] += f_axial * np.cos(seg_angle)
        Fz[i] += f_axial * np.sin(seg_angle)
        Fx[i + 1] -= f_axial * np.cos(seg_angle)
        Fz[i + 1] -= f_axial * np.sin(seg_angle)
        My[i] -= m_bend
        My[i + 1] += m_bend
    Fz -= m * g
    return Fx, Fz, My

def apply_uniform_tension(Fx, Fz, My, T, x, z):
    tension_length = int(2 * N / 3)
    for i in range(tension_length):
        Fx[i] += T * nx[i]
        Fz[i] += T * nz[i]
        My[i] -= (thickness[i] / 2) * T * nx[i]
    return Fx, Fz, My

reaction_time = []
reaction_rx = []
reaction_rz = []
reaction_rm = []
height = []
tendon_forces = []
x_history = []
z_history = []
theta_history = []

# ---------------------------
# Vertices and Faces
# ---------------------------
def generate_faces(x, y, z, thickness, width):
    vertices = []
    faces = []

    for i in range(N):
        top_left = (x[i], y[i] + width / 2, z[i] + thickness[i] / 2)
        top_right = (x[i], y[i] - width / 2, z[i] + thickness[i] / 2)
        bottom_left = (x[i], y[i] + width / 2, z[i] - thickness[i] / 2)
        bottom_right = (x[i], y[i] - width / 2, z[i] - thickness[i] / 2)
        vertices.append([top_left, top_right, bottom_right, bottom_left])

    for i in range(N - 1):
        v1 = vertices[i]
        v2 = vertices[i + 1]
        faces.append([v1[0], v1[1], v2[1], v2[0]])  # Top face
        faces.append([v1[3], v1[2], v2[2], v2[3]])  # Bottom face
        faces.append([v1[0], v1[3], v2[3], v2[0]])  # Left face
        faces.append([v1[1], v1[2], v2[2], v2[1]])  # Right face

    faces.append(vertices[0])      # Start face
    faces.append(vertices[-1])     # End face

    return faces

faces = generate_faces(x, y, z, thickness, width)

# Lists to store tracked x and z values for the four key points
tracked_x = {"fixed": [], "one_third": [], "two_third": [], "tip": []}
tracked_z = {"fixed": [], "one_third": [], "two_third": [], "tip": []}

# ---------------------------
# Simulation Function
# ---------------------------
def simulate():
    global x, z
    vx = np.zeros_like(x)
    vz = np.zeros_like(z)
    omega = np.zeros_like(x)
    theta_state = np.zeros_like(x)

    for step in range(time_steps):
        t = step * dt
        # Print intermediate time steps
        if step % (time_steps // 10) == 0:  # Print at approximately 10 intervals
            print(f"Time step: {step}, Time: {t:.6f} s")
        Fx_int, Fz_int, My_int = compute_internal_forces(x, z, theta_state)
        Fx_f, Fz_f = friction_forces(x, z, vx, vz)
        Fx_int, Fz_int, My_int = apply_uniform_tension(Fx_int, Fz_int, My_int, tendon_tension(t, z, angle_deg), x, z)
        T = tendon_tension(t, z, angle_deg)

        if T == 0:
            vx *= 0.5
            vz *= 0.5
            omega *= 0.5
            
        restoring_factor = 8e2
        for i in range(N):
            Fx_int[i] -= restoring_factor * (x[i] - initial_x[i])
            Fz_int[i] -= restoring_factor * (z[i] - initial_z[i])

        Fx_net = Fx_int + Fx_f - damp_c * vx
        Fz_net = Fz_int + Fz_f - damp_c * vz

        Fz_contact = contact_forces(z, vz)
        Fz_net += Fz_contact

        My_net = My_int - damp_c * omega  # Fixed: Changed 'c' to 'damp_c'

        ax = Fx_net / m
        az = Fz_net / m
        alpha = My_net / (0.5 * rho * A * dx**2)

        vx += ax * dt
        vz += az * dt
        omega += alpha * dt

        x += vx * dt
        z += vz * dt
        theta_state += omega * dt

        x[0] = 0
        vx[0] = 0

        # Store simulation results
        x_history.append(np.copy(x))
        z_history.append(np.copy(z))
        theta_history.append(np.copy(theta_state))

        height.append(z[0])
        Rx = -Fx_net[0]
        Rz = -Fz_net[0]
        Rm = -My_net[0]

        reaction_time.append(t)
        reaction_rx.append(Rx)
        reaction_rz.append(Rz)
        reaction_rm.append(Rm)
        tendon_forces.append(T)

        # Track key points
        tracked_x["fixed"].append(x[0])
        tracked_x["one_third"].append(x[N//3])
        tracked_x["two_third"].append(x[2*N//3])
        tracked_x["tip"].append(x[N-1])
        
        tracked_z["fixed"].append(z[0]*1000)
        tracked_z["one_third"].append(z[N//3]*1000)
        tracked_z["two_third"].append(z[2*N//3]*1000)
        tracked_z["tip"].append(z[N-1]*1000)

# ---------------------------
# Animation Function
# ---------------------------
def animate():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Axis limits and ticks
    ax.set_xlim(-0.05, L + 0.05)
    ax.set_ylim(-0.05, 0.05)
    ax.set_zlim(-0.05, 0.2)
    ax.set_yticks(np.linspace(-0.05, 0.05, 3))

    # Axis labels with bold, Times New Roman, and larger font
    ax.set_xlabel('X Position [m]', fontweight='bold', fontsize=16, fontname='Times New Roman')
    ax.set_ylabel('Y Position [m]', fontweight='bold', fontsize=16, fontname='Times New Roman')
    ax.set_zlabel('Z Position [m]', fontweight='bold', fontsize=16, fontname='Times New Roman')

    # Tick labels styling
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontweight('bold')
        tick.set_fontsize(14)
        tick.set_fontname('Times New Roman')

    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontweight('bold')
        tick.set_fontsize(14)
        tick.set_fontname('Times New Roman')

    for tick in ax.zaxis.get_ticklabels():
        tick.set_fontweight('bold')
        tick.set_fontsize(14)
        tick.set_fontname('Times New Roman')

    # Title
    ax.set_title(
        'Cosserat Rod Simulation with Uniform Tendon Tension and Restoring Forces',
        fontweight='bold',
        fontsize=16,
        fontname='Times New Roman'
    )

    ax.grid(True)
    ax.set_aspect('equal', 'box')

    # Set the camera view to focus on the x-z plane
    ax.view_init(elev=0, azim=-40)

    rod_patch = Poly3DCollection(faces, facecolor='lightblue', edgecolor='blue', alpha=0.4)
    ax.add_collection3d(rod_patch)
    centerline, = ax.plot(x, z, y, 'k--', linewidth=2, label='Centerline')
    tendon_line, = ax.plot([], [], [], 'r-', lw=2, label='Tendon')
    pulley_marker = ax.scatter(px, pz, 0, color='orange', s=50, label='Pulley')
    tendon_text = ax.text(0.1, 0.1, 0.3, '', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    reaction_text = ax.text(0.02, 0.4, -0.1, '', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Time annotation
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))

    def update(step):
        global x, z
        t = step * dt

        # Access precomputed data
        current_x = x_history[step]
        current_z = z_history[step]
        current_theta = theta_history[step]

        # Retrieve precomputed values
        Rx = reaction_rx[step]
        Rz = reaction_rz[step]
        Rm = reaction_rm[step]
        T = tendon_forces[step]

        px = current_x[0] + R0 * np.cos(theta_angle)
        pz = current_z[0] + R0 * np.sin(theta_angle)
        pulley_marker._offsets3d = ([px], [0], [pz])

        top_x = current_x + (thickness / 2) * nx
        top_z = current_z + (thickness / 2) * nz
        bottom_x = current_x - (thickness / 2) * nx
        bottom_z = current_z - (thickness / 2) * nz
        rod_x = np.concatenate([top_x, bottom_x[::-1]])
        rod_z = np.concatenate([top_z, bottom_z[::-1]])

        updated_faces = generate_faces(current_x, y, current_z, thickness, width)
        rod_patch.set_verts(updated_faces)

        centerline.set_data(current_x, y)
        centerline.set_3d_properties(current_z)
        two_thirds_index = int(2 * N / 3)
        tendon_x = [px] + list(x_history[step][:two_thirds_index])
        tendon_z = [pz] + list(current_z[:two_thirds_index] - thickness[:two_thirds_index] / 2)
        tendon_y = [0] * len(tendon_x)

        tendon_line.set_data(tendon_x, tendon_y)
        tendon_line.set_3d_properties(tendon_z)

        tendon_text.set_text(f'Tendon Force: {T:.2f} N')
        reaction_text.set_text(f'Reaction at Fixed End:\nRx = {Rx:.2f} N\nRz = {Rz:.2f} N\nRm = {Rm:.2e} Nm')
        time_text.set_text(f'Time: {t:.5f} s')
        return rod_patch, centerline, tendon_line, pulley_marker, tendon_text, reaction_text

    frame_skip = 200
    ani = FuncAnimation(fig, update, frames=range(0, time_steps, frame_skip), interval=30, blit=False, repeat=True)

    legend = plt.legend(
        bbox_to_anchor=(0.75, 0.75),
        loc='center',
        frameon=True
    )
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(14)
        text.set_fontname('Times New Roman')
    plt.show()

# Run simulation
start_time = time.time()
simulate()
end_time = time.time()
elapsed = end_time - start_time
print(f"\n✅ Simulation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")

# Run or skip animation
run_animation = False
if run_animation:
    animate()
else:
    print("Animation is turned off. Only simulation results will be displayed.")

# Plot results after simulation
plt.figure(figsize=(12, 6))

# Plot Rx
plt.subplot(4, 1, 1)
plt.plot(reaction_time, reaction_rx, label='$R_x$', color='b')
plt.xlabel('Time [s]')
plt.ylabel('Reaction Force $R_x$ [N]')
plt.title('Reaction Force $R_x$ vs Time')
plt.grid(True)
plt.legend()

# Plot Rz
plt.subplot(4, 1, 2)
plt.plot(reaction_time, reaction_rz, label='$R_z$', color='g')
plt.xlabel('Time [s]')
plt.ylabel('Reaction Force $R_z$ [N]')
plt.title('Reaction Force $R_z$ vs Time')
plt.grid(True)
plt.legend()

# Plot Rm
plt.subplot(4, 1, 3)
plt.plot(reaction_time, reaction_rm, label='$R_m$', color='r')
plt.xlabel('Time [s]')
plt.ylabel('Reaction Moment $R_m$ [Nm]')
plt.title('Reaction Moment $R_m$ vs Time')
plt.grid(True)
plt.legend()

# Plot height
plt.subplot(4, 1, 4)
plt.plot(reaction_time, height, label='$z[0]$', color='r')
plt.xlabel('Time [s]')
plt.ylabel('Height $h$ [m]')
plt.title('Height $h$ vs Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Plot X and Z positions for the tracked points
plt.figure(figsize=(12, 6))
plt.plot(reaction_time, tracked_x["fixed"], label='Fixed End', color='b')
plt.plot(reaction_time, tracked_x["one_third"], label='1/3rd Length', color='g')
plt.plot(reaction_time, tracked_x["two_third"], label='2/3rd Length', color='r')
plt.plot(reaction_time, tracked_x["tip"], label='Tip', color='m')
plt.xlabel('Time [s]')
plt.ylabel('X Position [m]')
plt.title('X Position Evolution')
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 6))
plt.plot(reaction_time, tracked_z["fixed"], label='Fixed End', color='b')
plt.plot(reaction_time, tracked_z["one_third"], label='1/3rd Length', color='g')
plt.plot(reaction_time, tracked_z["two_third"], label='2/3rd Length', color='r')
plt.plot(reaction_time, tracked_z["tip"], label='Tip', color='m')
plt.xlabel('Time [s]')
plt.ylabel('Z Position [mm]')
plt.title('Z Position Evolution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Comparison with experimental data
excel_file = "single_leg_aruco_val.xlsx"
comparison_config = {
    "id6": "fixed",
    "id2": "one_third",
    "id3": "two_third",
    "id4": "tip",
}

# Plotting settings
fontname = 'Times New Roman'
fontsize = 16
sim_linewidth = 4

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

label_map = {
    'fixed': 'Node 0 (Fixed Base)',
    'one_third': 'Node 10 (~1/3 Length)',
    'two_third': 'Node 20 (~2/3 Length)',
    'tip': 'Node 30 (Tip)'
}

for idx, (sheet_name, sim_key) in enumerate(comparison_config.items()):
    ax = axes[idx]
    df = pd.read_excel(excel_file, sheet_name=sheet_name, usecols=[0, 11], header=None)
    df.columns = ['Time', 'Z']
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Z'] = pd.to_numeric(df['Z'], errors='coerce')
    df.dropna(inplace=True)

    time_exp = df['Time'].to_numpy()
    z_exp = df['Z'].to_numpy()
    time_sim = reaction_time
    z_sim = tracked_z[sim_key]

    ax.plot(time_exp, z_exp, linestyle='--', marker='o', label='Experimental', color='#1E90FF')
    ax.plot(time_sim, z_sim, linewidth=sim_linewidth, label='Simulation', color='tab:red')
    ax.set_title(label_map[sim_key], fontweight='bold', fontsize=fontsize+1, fontname=fontname)
    ax.set_xlabel("Time [s]", fontweight='bold', fontsize=fontsize, fontname=fontname)
    ax.set_ylabel("Z Position [mm]", fontweight='bold', fontsize=fontsize, fontname=fontname)
    ax.tick_params(axis='both', labelsize=fontsize-1, width=1.5)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
        tick.set_fontname(fontname)
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(fontsize)
        text.set_fontname(fontname)
    ax.grid(True)

plt.tight_layout()
plt.savefig("z_position_comparison_single_leg.png", dpi=300, bbox_inches='tight')
plt.show()

# Tip trajectory plot with error handling
x_tip = np.array(tracked_x["tip"]) * 1000  # convert to mm
z_tip = np.array(tracked_z["tip"])         # already in mm
print("x_tip range:", np.min(x_tip), "to", np.max(x_tip))  # Diagnostic

# Filter for X in [150, 190] mm
mask = (x_tip >= 150) & (x_tip <= 190)
x_filtered = x_tip[mask]
z_filtered = z_tip[mask]

plt.figure(figsize=(8, 6))
if len(x_filtered) > 0:
    plt.plot(x_filtered, z_filtered, color='purple', linewidth=2)
    plt.scatter(x_filtered[0], z_filtered[0], color='green', label='Start', s=80)
    plt.scatter(x_filtered[-1], z_filtered[-1], color='red', label='End', s=80)
else:
    print("No data points in the specified X range [150, 190] mm")
plt.title("Filtered Tip Workspace (150 mm ≤ X ≤ 190 mm)")
plt.xlabel("X Position [mm]")
plt.ylabel("Z Position [mm]")
plt.ylim(-1, 10)
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

# Error metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("\n=== Error Metrics for Z Position [mm] ===")
for sheet_name, sim_key in comparison_config.items():
    df = pd.read_excel(excel_file, sheet_name=sheet_name, usecols=[0, 11], header=None)
    df.columns = ['Time', 'Z']
    df.dropna(inplace=True)
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df['Z'] = pd.to_numeric(df['Z'], errors='coerce')
    df.dropna(inplace=True)

    time_exp = df['Time'].to_numpy().astype(float)
    z_exp = df['Z'].to_numpy().astype(float)
    time_sim = np.array(reaction_time).astype(float)
    z_sim = np.array(tracked_z[sim_key]).astype(float)

    z_sim_interp = np.interp(time_exp, time_sim, z_sim)
    rmse = np.sqrt(mean_squared_error(z_exp, z_sim_interp))
    mae = mean_absolute_error(z_exp, z_sim_interp)
    avg_error = np.mean(z_sim_interp - z_exp)
    data_range = np.max(z_exp) - np.min(z_exp)
    nrmse = rmse / data_range * 100
    error_percent = abs(avg_error) / data_range * 100
    accuracy = 100 - error_percent

    print(f"{sim_key:10s}: RMSE = {rmse:.3f} mm, MAE = {mae:.3f} mm, "
          f"NRMSE = {nrmse:.2f}%, Avg Error = {avg_error:.3f} mm, "
          f"Error% = {error_percent:.2f}%, Accuracy = {accuracy:.2f}%")