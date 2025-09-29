import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# Rod + Cuboid Parameters
# ---------------------------
L = 0.190;   N = 31;   dx = L/(N-1)
th_f, th_t = 0.0135, 0.0035
thickness  = th_f + (th_t-th_f)*np.linspace(0,1,N)

# initial rod shape (cubic)
a, b, c0, d0 = 6.356, -1.810, 0.0, 0.02175
x_init = np.linspace(0, L, N)
z_init = a*x_init**3 + b*x_init**2 + c0*x_init + d0
initial_x, initial_z = x_init.copy(), z_init.copy()

rho   = 1200.0;   width = 0.02
A     = thickness * width
I     = width * thickness**3 / 12.0
E     = np.full(N-1, 1e6)

# node masses
m = np.zeros(N)
m[1:-1] = (rho*A[:-2]*dx + rho*A[1:-1]*dx)/2
m[0]    = rho*A[0]*dx/2
m[-1]   = rho*A[-1]*dx/2

mc = 2.0    # cuboid mass (kg)
g  = 9.81

# rod‐ground interaction
mu        = 0.1     # friction
k_contact = 1e6     # contact stiffness (increased)
d_contact = 10.0    # contact damping
damp_c    = 0.5     # rod viscous damping (increased)

# simulation time
dt         = 1e-4
sim_time   = 25.0
time_steps = int(sim_time/dt)

# precompute normals
dxs = np.gradient(x_init)
dzs = np.gradient(z_init)
lens= np.hypot(dxs, dzs)
nx  = -dzs/ lens
nz  =  dxs/ lens

# ---------------------------
# Tendon / Pulley
# ---------------------------
k_tension_scale = 4e-4
def angle_to_force(ang_deg):
    torque = (ang_deg/270)*1.5
    return torque/0.01

def tendon_tension(t, z, ang_deg):
    t0, ramp, hold, decay = 5.0, 2.4, 0.2, 2.0
    t_ramp_end  = t0 + ramp
    t_hold_end  = t_ramp_end + hold
    t_decay_end = t_hold_end + decay

    # no pull if tip lifts or outside window
    if z[-1] - thickness[-1] > 0 or t < t0 or t > t_decay_end:
        return 0.0

    if t < t_ramp_end:
        frac = np.sin(np.pi*(t-t0)/(2*ramp))
    elif t < t_hold_end:
        frac = 1.0
    else:
        frac = np.cos(np.pi*(t-t_hold_end)/(2*decay))

    return angle_to_force(ang_deg) * k_tension_scale * frac

# ---------------------------
# Internal + Friction + Contact
# ---------------------------
def compute_internal_forces(x, z, theta):
    Fx = np.zeros(N); Fz = np.zeros(N); My = np.zeros(N)
    for i in range(N-1):
        dx_i, dz_i = x[i+1]-x[i], z[i+1]-z[i]
        l_i = np.hypot(dx_i, dz_i)
        eps = (l_i-dx)/dx
        ang = np.arctan2(dz_i, dx_i)
        dth= (theta[i+1]-theta[i]) if i<N-2 else 0.0

        f_ax = E[i]*A[i]*eps
        m_bd = E[i]*I[i]*dth/dx

        Fx[i]   += f_ax*np.cos(ang)
        Fz[i]   += f_ax*np.sin(ang)
        Fx[i+1] -= f_ax*np.cos(ang)
        Fz[i+1] -= f_ax*np.sin(ang)
        My[i]   -= m_bd
        My[i+1] += m_bd

    # gravity + static share
    Fz -= m*g
    Fz[0] -= mc*g/4
    return Fx, Fz, My

def friction_forces(x, z, vx, vz):
    Fx_f = np.zeros(N)
    for i in range(N):
        if z[i]<=0 and abs(vx[i])>1e-12:
            Fx_f[i] = -mu*m[i]*g*np.sign(vx[i])
    return Fx_f

def contact_forces(z, vz):
    Fz_ct = np.zeros(N)
    if z[-1] <= 0:
        pen = -z[-1]; vd = -vz[-1]
        Fz_ct[-1] = k_contact*pen + d_contact*vd
    return Fz_ct

def apply_uniform_tension(Fx, Fz, My, T):
    idx = int(2*N/3)
    for i in range(idx):
        Fx[i]   += T*nx[i]
        Fz[i]   += T*nz[i]
        My[i]  -= (thickness[i]/2)*T*nx[i]
    return Fx, Fz, My

# ---------------------------
# Gait Schedule (unchanged)
# ---------------------------
base_neutral = 116.28
amp          = 25.0
def build_gait_schedule(leg):
    delay = int(0.5/dt)
    up, rl = base_neutral+amp, base_neutral-amp
    sched  = np.zeros(time_steps)
    for i in range(0, time_steps, 2*delay):
        if leg in ['BR','BL','FR']:
            sched[i:i+delay] = up
        else:
            sched[i:i+delay] = rl
        if leg in ['BL','FL','BR']:
            sched[i+delay:i+2*delay] = up
        else:
            sched[i+delay:i+2*delay] = rl
    return sched

# ---------------------------
# Single‐Leg Simulation
# ---------------------------
def simulate_leg(angle_sched):
    x, z       = x_init.copy(), z_init.copy()
    vx, vz     = np.zeros(N), np.zeros(N)
    omega, th  = np.zeros(N), np.zeros(N)
    rx, rz     = np.zeros(time_steps), np.zeros(time_steps)

    for k in range(time_steps):
        t = k*dt
        Fx_i, Fz_i, My_i = compute_internal_forces(x, z, th)
        Fx_fr = friction_forces(x, z, vx, vz)
        Fz_ct = contact_forces(z, vz)

        ang = angle_sched[k]
        T   = tendon_tension(t, z, ang)
        Fx_i, Fz_i, My_i = apply_uniform_tension(Fx_i, Fz_i, My_i, T)

        if T==0:
            vx   *= 0.5; vz   *= 0.5; omega*=0.5

        # stronger restoring spring
        Fx_i -= 1e3*(x - initial_x)
        Fz_i -= 1e3*(z - initial_z)

        Fx_n = Fx_i + Fx_fr - damp_c*vx
        Fz_n = Fz_i + Fz_ct - damp_c*vz
        My_n = My_i - damp_c*omega

        ax    = Fx_n / m
        az    = Fz_n / m
        alpha = My_n/(0.5*rho*A*dx*dx)

        vx    += ax*dt;     x+= vx*dt
        vz    += az*dt;     z+= vz*dt
        omega+= alpha*dt;   th+=omega*dt

        vx[0] = 0; x[0]=0
        rx[k] = -Fx_n[0]
        rz[k] = -Fz_n[0]

    return rx, rz

# ---------------------------
# Multi‐Leg + Cuboid Integration
# ---------------------------
leg_labels    = ['BR','BL','FL','FR']
angles_global = [225,135,45,315]

rx_all, rz_all = [], []
for leg in leg_labels:
    sched = build_gait_schedule(leg)
    rxi, rzi = simulate_leg(sched)
    rx_all.append(rxi); rz_all.append(rzi)
rx_all = np.array(rx_all)
rz_all = np.array(rz_all)

# ---------------------------
# Cuboid CoM Integration
# ---------------------------
c_cuboid = 0.1      # horizontal drag N·s/m (unchanged)
k_z      = 1500.0   # vertical spring stiffness N/m
c_z      =  200.0   # vertical damping  N·s/m
z0       = -0.175   # target mean height (m)

p = np.zeros((time_steps,3))
v = np.zeros((time_steps,3))

for k in range(1, time_steps):
    # leg sum → world
    Fx = sum(rx_all[j,k]*np.cos(np.deg2rad(angles_global[j])) for j in range(4))
    Fy = sum(rx_all[j,k]*np.sin(np.deg2rad(angles_global[j])) for j in range(4))
    Fz = sum(rz_all[j,k] for j in range(4)) - mc*g

    # horizontal drag
    Fdx = -c_cuboid * v[k-1,0]
    Fdy = -c_cuboid * v[k-1,1]

    # vertical spring–damper about z0
    dz    = p[k-1,2] - z0
    Fz_sup= -k_z*dz - c_z*v[k-1,2]

    # accelerations
    a_x = (Fx + Fdx)/mc
    a_y = (Fy + Fdy)/mc
    a_z = (Fz + Fz_sup)/mc

    v[k,0] = v[k-1,0] + a_x*dt
    v[k,1] = v[k-1,1] + a_y*dt
    v[k,2] = v[k-1,2] + a_z*dt

    p[k]   = p[k-1] + v[k]*dt

# ---------------------------
# Plot & compare to experiment
# ---------------------------
# --- Styling controls ---
fontsize = 24
fontname = 'Times New Roman'
legend_fontsize = 20
marker_size = 5
sim_linewidth = 7.0

# --- Time vector ---
t = np.arange(time_steps) * dt

# --- Load experimental data ---
df = pd.read_excel('forward_walk/test3arcuvoforwardgait.xlsx', sheet_name='id 2')
time_exp = df['Human'].to_numpy()
x_exp = df['X_com (mm)'].to_numpy()
y_exp = df['Y_com (mm)'].to_numpy()
z_exp = df['Z_rel (mm)'].to_numpy()

# --- Create 1x3 subplot layout ---
fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharex=False)

# ========== X_COM ==========
axes[0].plot(t, p[:, 0]*1e3, label='Sim $X_{\\mathrm{COM}}$', color='#FF5733', linewidth=sim_linewidth)
axes[0].plot(time_exp, x_exp,
        '--',
        label='Exp $X_{\\mathrm{COM}}$',
        color='k',
        marker='o', markersize=marker_size, alpha=0.7)
axes[0].set_xlabel("Time (s)", fontsize=fontsize, fontweight='bold', fontname=fontname)
axes[0].set_ylabel(r"X position (mm)", fontsize=fontsize, fontweight='bold', fontname=fontname)
axes[0].grid(True)
legend = axes[0].legend(frameon=True, fontsize=legend_fontsize)
legend.get_frame().set_edgecolor('black')
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_fontname(fontname)

# ========== Z_COM ==========
axes[1].plot(t, -p[:, 2]*1e3, label='Sim $Z_{\\mathrm{COM}}$', color='tab:green', linewidth=sim_linewidth)
axes[1].plot(time_exp, -z_exp,
        '--',
        label='Exp $Z_{\\mathrm{COM}}$',
        color='k',
        marker='o', markersize=marker_size, alpha=0.7)
axes[1].set_xlabel("Time (s)", fontsize=fontsize, fontweight='bold', fontname=fontname)
axes[1].set_ylabel(r"Z position of torso (mm)", fontsize=fontsize, fontweight='bold', fontname=fontname)
axes[1].grid(True)
legend = axes[1].legend(frameon=True, fontsize=legend_fontsize)
legend.get_frame().set_edgecolor('black')
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_fontname(fontname)

# Apply consistent tick styling
for ax in axes:
    ax.tick_params(axis='both', labelsize=fontsize-1, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontname(fontname)

plt.tight_layout()
plt.savefig("Forward_walk_com_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

