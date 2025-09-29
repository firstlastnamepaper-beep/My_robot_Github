import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# Rod + Cuboid Parameters
# ---------------------------
L = 0.190       # rod length (m)
N = 31          # nodes per rod
dx = L/(N-1)

# linear taper of thickness
th_f, th_t = 0.0135, 0.0035
thickness = th_f + (th_t-th_f)*np.linspace(0,1,N)

# cubic‐spline initial shape
a,b,c0,d0 = 6.356, -1.810, 0.0, 0.02175
x_init = np.linspace(0, L, N)
z_init = a*x_init**3 + b*x_init**2 + c0*x_init + d0
initial_x, initial_z = x_init.copy(), z_init.copy()

rho   = 1200.0     # material density (kg/m³)
width = 0.02       # rod width (m)
A     = thickness * width
I     = width * thickness**3 / 12.0
E     = np.full(N-1, 1e6)  # Young's modulus (Pa)

# nodal masses
m = np.zeros(N)
m[1:-1] = (rho*A[:-2]*dx + rho*A[1:-1]*dx)/2
m[0]    = rho*A[0]*dx/2
m[-1]   = rho*A[-1]*dx/2

# cuboid (body) parameters
mc = 2.0    # mass (kg)
g  = 9.81   # gravity (m/s²)

# rod‐ground interaction
mu        = 0.1     # friction coeff
k_contact = 1e5     # ground stiffness (N/m)
d_contact = 10.0    # ground damping (N·s/m)
damp_c    = 0.5     # rod viscous damping

# precompute rod normals
dxs = np.gradient(x_init)
dzs = np.gradient(z_init)
lens= np.hypot(dxs, dzs)
nx  = -dzs/ lens
nz  =  dxs/ lens

# ---------------------------
# Crawl Gait scheduling
# ---------------------------
dt         = 1e-4
sim_time   = 25.0
steps      = int(sim_time/dt)

# four‐phase crawl (each phase = 1 s)
t0         = 4.0         # initial delay
phase_time = 1.0
cycle_time = 4*phase_time
ds         = int(phase_time/dt)

# pulse shape
ramp, hold, decay = 0.2, 0.6, 0.2

# neutral & deflection angles
def build_crawl_schedule(leg):
    ul, ulc, relc, defc = 180, 220, 60, 100  # your earlier values
    sched = np.zeros(steps)
    for base in range(0, steps, 4*ds):
        # Phase 1
        s,e = base, base+ds
        if   leg in ('FR','BL'): sched[s:e]=defc
        elif leg=='FL':          sched[s:e]=ulc
        else:                     sched[s:e]=defc-relc
        # Phase 2
        s,e = base+ds, base+2*ds
        if   leg=='BR':          sched[s:e]=ulc
        elif leg=='FL':          sched[s:e]=defc-relc
        else:                     sched[s:e]=defc
        # Phase 3
        s,e = base+2*ds, base+3*ds
        if   leg in ('BR','FL'): sched[s:e]=defc
        elif leg=='BL':          sched[s:e]=defc-relc
        else:                     sched[s:e]=ulc
        # Phase 4
        s,e = base+3*ds, base+4*ds
        if   leg=='FR':          sched[s:e]=defc-relc
        elif leg=='BL':          sched[s:e]=ulc
        else:                     sched[s:e]=defc
    return sched

# ---------------------------
# Tendon tension profile
# ---------------------------
k_tension_scale = 1e-4
def angle_to_force(ang_deg):
    torque = (ang_deg/270)*1.5
    return torque/0.01

def tendon_tension(t, z, ang_deg):
    # no pull if tip lifts or before t0
    if (z[-1]-thickness[-1]>0) or (t<t0):
        return 0.0
    t_adj   = (t-t0) % cycle_time
    t_phase = t_adj % phase_time
    if   t_phase< ramp:      frac = np.sin(np.pi*t_phase/(2*ramp))
    elif t_phase< ramp+hold: frac = 1.0
    else:                    frac = np.cos(np.pi*(t_phase-(ramp+hold))/(2*decay))
    return angle_to_force(ang_deg)*k_tension_scale*frac

# ---------------------------
# Internal/bending, friction, contact
# ---------------------------
def compute_internal_forces(x, z, th):
    Fx = np.zeros(N); Fz = np.zeros(N); My = np.zeros(N)
    for i in range(N-1):
        dx_i, dz_i = x[i+1]-x[i], z[i+1]-z[i]
        li = np.hypot(dx_i, dz_i)
        eps    = (li-dx)/dx
        seg_ang= np.arctan2(dz_i,dx_i)
        dth    = (th[i+1]-th[i]) if i<N-2 else 0.0
        f_ax = E[i]*A[i]*eps
        m_bd = E[i]*I[i]*dth/dx
        Fx[i]   += f_ax*np.cos(seg_ang)
        Fz[i]   += f_ax*np.sin(seg_ang)
        Fx[i+1] -= f_ax*np.cos(seg_ang)
        Fz[i+1] -= f_ax*np.sin(seg_ang)
        My[i]   -= m_bd
        My[i+1] += m_bd
    # gravity + share to base
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
    if z[-1]<=0:
        pen, vd = -z[-1], -vz[-1]
        Fz_ct[-1] = k_contact*pen + d_contact*vd
    return Fz_ct

def apply_uniform_tension(Fx, Fz, My, T):
    idx = int(2*N/3)
    for i in range(idx):
        Fx[i] += T*nx[i]
        Fz[i] += T*nz[i]
        My[i] -= (thickness[i]/2)*T*nx[i]
    return Fx, Fz, My

# ---------------------------
# Single‐leg simulation
# ---------------------------
def simulate_leg(angle_sched):
    x,z    = x_init.copy(), z_init.copy()
    vx,vz  = np.zeros(N), np.zeros(N)
    omega,ths = np.zeros(N), np.zeros(N)
    rx,rz  = np.zeros(steps), np.zeros(steps)
    for k in range(steps):
        t = k*dt
        Fx_i, Fz_i, My_i = compute_internal_forces(x,z,ths)
        Fx_fr = friction_forces(x,z,vx,vz)
        Fz_ct = contact_forces(z,vz)
        T     = tendon_tension(t, z, angle_sched[k])
        Fx_i,Fz_i,My_i = apply_uniform_tension(Fx_i,Fz_i,My_i,T)
        # extra damping when tendon slack
        if T==0:
            vx*=0.5; vz*=0.5; omega*=0.5
        # shape‐restoring
        Fx_i -= 1e2*(x-initial_x)
        Fz_i -= 1e2*(z-initial_z)
        Fx_n = Fx_i + Fx_fr - damp_c*vx
        Fz_n = Fz_i + Fz_ct - damp_c*vz
        My_n = My_i - damp_c*omega
        ax    = Fx_n/m
        az    = Fz_n/m
        alpha = My_n/(0.5*rho*A*dx*dx)
        vx   += ax*dt;   x  += vx*dt
        vz   += az*dt;   z  += vz*dt
        omega+=alpha*dt; ths+=omega*dt
        vx[0]=0; x[0]=0
        rx[k] = -Fx_n[0]
        rz[k] = -Fz_n[0]
    return rx, rz

# ---------------------------
# Multi‐leg + cuboid integration
# ---------------------------
leg_labels    = ['BR','BL','FL','FR']
angles_global = [225,135,45,315]

rx_all, rz_all = [], []
for leg in leg_labels:
    sched = build_crawl_schedule(leg)
    rxi, rzi = simulate_leg(sched)
    rx_all.append(rxi); rz_all.append(rzi)
rx_all = np.array(rx_all)
rz_all = np.array(rz_all)

# CoM integration with horizontal drag + vertical spring‐damper (tuned to ~150 mm)
c_cuboid = 0.1
k_z      = 1500.0
c_z      =  200.0
z0       = -0.150

p = np.zeros((steps,3))
v = np.zeros((steps,3))
for k in range(1, steps):
    Fx = sum(rx_all[j,k]*np.cos(np.deg2rad(angles_global[j])) for j in range(4))
    Fy = sum(rx_all[j,k]*np.sin(np.deg2rad(angles_global[j])) for j in range(4))
    Fz = sum(rz_all[j,k] for j in range(4)) - mc*g

    # horizontal drag
    Fdx = -c_cuboid * v[k-1,0]
    Fdy = -c_cuboid * v[k-1,1]
    # vertical spring‐damper about z0
    dz   = p[k-1,2] - z0
    Fz_sd= -k_z*dz - c_z*v[k-1,2]

    ax = -(Fx + Fdx)/mc
    ay = (Fy + Fdy)/mc
    az = (Fz + Fz_sd)/mc

    v[k] = v[k-1] + np.array([ax, ay, az])*dt
    p[k] = p[k-1] + v[k]*dt

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
t = np.arange(steps) * dt

# --- Plot X_com & Z_com vs experiment ---
df = pd.read_excel('crawl/single_leg_aruco_val.xlsx', sheet_name='id 2')
time_exp = df['human'].to_numpy()
x_exp    = df['X_com (mm)'].to_numpy()
y_exp      = df['Y_com (mm)'].to_numpy()
z_exp    = df['Z_rel (mm)'].to_numpy()

# --- Create 1x3 subplot layout ---
fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharex=False)

# X_COM
ax = axes[0]
ax.plot(t, p[:,0]*1e3,
        label='Sim $X_{\\mathrm{COM}}$',
        linewidth=sim_linewidth,
        color='#FF5733')
ax.plot(time_exp, x_exp,
        '--',
        label='Exp $X_{\\mathrm{COM}}$',
        color='k',
        marker='o', markersize=marker_size, alpha=0.7)
ax.set_xlabel('Time (s)',
              fontweight='bold', fontsize=fontsize, fontname=fontname)
ax.set_ylabel('X Position (mm)',
              fontweight='bold', fontsize=fontsize, fontname=fontname)
ax.grid(True)
ax.tick_params(axis='both', labelsize=fontsize-1, width=1.2)
for lbl in ax.get_xticklabels()+ax.get_yticklabels():
    lbl.set_fontweight('bold'); lbl.set_fontname(fontname)
leg = ax.legend(frameon=True, fontsize=legend_fontsize)
leg.get_frame().set_edgecolor('black')
for txt in leg.get_texts():
    txt.set_fontweight('bold'); txt.set_fontname(fontname)

# Z_COM
ax = axes[1]
ax.plot(t, -p[:,2]*1e3,
        label='Sim $Z_{\\mathrm{COM}}$',
        linewidth=sim_linewidth,
        color='tab:green')
ax.plot(time_exp, -z_exp,
        '--',
        label='Exp $Z_{\\mathrm{COM}}$',
        color='k',
        marker='o', markersize=marker_size, alpha=0.7)
ax.set_xlabel('Time (s)',
              fontweight='bold', fontsize=fontsize, fontname=fontname)
ax.set_ylabel('Z Position (mm)',
              fontweight='bold', fontsize=fontsize, fontname=fontname)
ax.grid(True)
ax.tick_params(axis='both', labelsize=fontsize-1, width=1.2)
for lbl in ax.get_xticklabels()+ax.get_yticklabels():
    lbl.set_fontweight('bold'); lbl.set_fontname(fontname)
leg = ax.legend(frameon=True, fontsize=legend_fontsize)
leg.get_frame().set_edgecolor('black')
for txt in leg.get_texts():
    txt.set_fontweight('bold'); txt.set_fontname(fontname)

plt.tight_layout()
plt.savefig('crawl_COM_xyz_comparison.png', dpi=300, bbox_inches='tight')
plt.show()