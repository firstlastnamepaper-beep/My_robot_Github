import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Robot parameters
mass = 2.16  # kg (2.0 kg body + 0.04 kg x 4 legs)
x_len = 0.1255
y_len = 0.0855
z_len = 0.034
Ixx = (1/12)*mass*(y_len**2 + z_len**2)
Iyy = (1/12)*mass*(x_len**2 + z_len**2)
Izz = (1/12)*mass*(x_len**2 + y_len**2)
inertia = np.diag([Ixx, Iyy, Izz])  # Inertia tensor (kg·m²)
g = 9.81  # Gravity (m/s²)
dt = 0.033  # Time step (s)
mpc_horizon = 15  # MPC horizon length
fd_horizon = 5  # Forward dynamics horizon length
f_max = 6.0  # Maximum force per leg (N)
num_legs = 4

leg_positions = np.array([
    [0.1255/2 + 0.19*np.sqrt(2), 0.0855/2+0.19*np.sqrt(2), -(0.16+0.034/2)],
    [0.1255/2 + 0.19*np.sqrt(2), -(0.0855/2+0.19*np.sqrt(2)), -(0.16+0.034/2)],
    [-(0.1255/2 + 0.19*np.sqrt(2)), (0.0855/2+0.19*np.sqrt(2)), -(0.16+0.034/2)],
    [-(0.1255/2 + 0.19*np.sqrt(2)), -(0.0855/2+0.19*np.sqrt(2)), -(0.16+0.034/2)]
])

# Leg frame rotation matrices (legs at 45 degrees outward)
sqrt2 = np.sqrt(2)/2
R_leg = [
    np.array([[sqrt2, sqrt2, 0], [-sqrt2, sqrt2, 0], [0, 0, 1]]),  # FL: 45 deg
    np.array([[sqrt2, -sqrt2, 0], [sqrt2, sqrt2, 0], [0, 0, 1]]),  # FR: 135 deg
    np.array([[-sqrt2, sqrt2, 0], [-sqrt2, -sqrt2, 0], [0, 0, 1]]), # BL: -45 deg
    np.array([[-sqrt2, -sqrt2, 0], [sqrt2, -sqrt2, 0], [0, 0, 1]])  # BR: -135 deg
]

# Continuous-time state-space matrices
A_c = np.zeros((13, 13))
A_c[0:3, 6:9] = np.eye(3)  # Angular velocities to angles
A_c[3:6, 9:12] = np.eye(3)  # Velocities to positions
A_c[9:12, 12] = np.array([0, 0, 1])  # Gravity affects z-acceleration
A_c[9, 9] = -0.1  # Damping on v_x
A_c[10, 10] = -0.1  # Damping on v_y
A_c[11, 11] = -0.5  # Increased damping on v_z

# Discretize dynamics
A_d = np.eye(13) + dt * A_c
B_c = np.zeros((13, 12))

for i in range(num_legs):
    B_c[9:12, 3*i:3*i+3] = (1/mass) * np.eye(3)  # Linear force to acceleration
    r = leg_positions[i]
    Rx = np.array([[0, 0, 0], [0, 0, -r[0]], [0, r[0], 0]])
    Ry = np.array([[0, 0, r[1]], [0, 0, 0], [-r[1], 0, 0]])
    Rz = np.array([[0, -r[2], 0], [r[2], 0, 0], [0, 0, 0]])
    R = Rx + Ry + Rz
    B_c[6:9, 3*i:3*i+3] = np.linalg.inv(inertia) @ R  # Torque to angular acceleration

B_d = dt * B_c

# Cost function weights
weights = np.array([
    0.1,      # roll
    0.1,      # pitch
    0.1,      # yaw
    0.0,      # p_x
    0.0,      # p_y
    100000.0, # p_z (increased to prioritize height)
    0.01,     # omega_x
    0.01,     # omega_y
    0.01,     # omega_z
    20.0,     # v_x
    0.1,      # v_y
    0.0,      # v_z (increased to dampen vertical velocity)
    0.0       # g_z
])/1000

# Reference trajectory
def get_reference_trajectory(t, horizon, dt, desired_vx=0.02, desired_pz=0.1):
    x_ref = np.zeros((13, horizon + 1))
    for k in range(horizon + 1):
        x_ref[9, k] = desired_vx
        x_ref[3, k] = desired_vx * (t + k * dt)
        x_ref[5, k] = desired_pz
        x_ref[12, k] = -g
    return x_ref

# MPC controller
def mpc_control(state, t):
    x0 = state.copy()
    x_ref = get_reference_trajectory(t, mpc_horizon, dt)

    # Gait sequence
    gait_period = 5
    phase = int((t % gait_period) >= 2.5)
    mu = [0.0] * num_legs
    if phase == 0:
        mu[0] = 0.6  # FL contracting
        mu[1] = 0.2  # FR relaxing
    else:
        mu[0] = 0.2  # FL relaxing
        mu[1] = 0.6  # FR contracting
    mu[2] = 0.1  # BL contracted
    mu[3] = 0.1  # BR contracted

    X = cp.Variable((13, mpc_horizon + 1))
    U = cp.Variable((12, mpc_horizon))

    # Cost function
    cost = 0
    desired_fx = 0.6  # Desired forward force in leg frame for contracting leg (N)
    for k in range(mpc_horizon):
        state_error = X[:, k] - x_ref[:, k]
        cost += cp.quad_form(state_error, np.diag(weights))
        # Add cost to encourage desired fx for contracting leg
        for leg in range(num_legs):
            f_vec = U[3*leg:3*leg+3, k]
            f_leg = R_leg[leg].T @ f_vec
            if leg == 0 or leg == 1:  # FL or FR
                if (leg == 0 and phase == 0) or (leg == 1 and phase == 1):
                    # Contracting: encourage fx to be close to desired_fx
                    cost += 0.01 * cp.sum_squares(f_leg[0] - desired_fx)
                else:
                    # Relaxing: encourage fx to be close to 0
                    cost += 0.01 * cp.sum_squares(f_leg[0])
    cost += 1e-8 * cp.sum_squares(U)  # Reduced control effort penalty

    # Constraints
    constraints = [X[:, 0] == x0]
    for k in range(mpc_horizon):
        constraints.append(X[:, k+1] == A_d @ X[:, k] + B_d @ U[:, k])
        total_fz = 0
        for leg in range(num_legs):
            f_vec = U[3*leg:3*leg+3, k]
            f_leg = R_leg[leg].T @ f_vec
            mu_leg = mu[leg]
            if leg == 0 or leg == 1:  # FL or FR
                if (leg == 0 and phase == 0) or (leg == 1 and phase == 1):
                    # Contracting: enforce minimum fx
                    constraints.append(f_leg[0] >= 0.2)  # Minimum forward force
                    constraints.append(f_leg[0] <= mu_leg * f_leg[2])
                else:
                    # Relaxing: enforce minimum fz to prevent collapse
                    constraints.append(f_leg[2] >= 4.0)  # Minimum normal force cap of 4 N
                    constraints.append(f_leg[0] >= -mu_leg * f_leg[2])
                    constraints.append(f_leg[0] <= mu_leg * f_leg[2])
                constraints.append(cp.abs(f_leg[1]) <= mu_leg * f_leg[2])
            else:  # BL or BR
                constraints.append(cp.norm(f_leg[:2], p=2) <= mu_leg * f_leg[2])
                constraints.append(f_leg[2] >= 4.0)  # Minimum normal force cap of 4 N
            constraints.append(f_vec[2] >= 0)
            constraints.append(f_vec[2] <= f_max)
            total_fz += f_vec[2]
        constraints.append(total_fz >= mass * g - 0.01)  # Tightened lower bound
        constraints.append(total_fz <= mass * g + 0.01)  # Tightened upper bound

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status != cp.OPTIMAL:
        print(f"Solver failed with status: {prob.status}")
        return np.zeros(12)
    return U.value[:, 0]

# Low-Level Control (LLC) with Original Soft Leg Model
class LowLevelControl:
    def __init__(self):
        # Parameters for theta control dynamics
        self.tau = 0.005  # Even faster time constant (seconds)
        self.alpha = 1 - np.exp(-dt / self.tau)
        self.beta = 1 - self.alpha
        self.theta_desired_history = [np.zeros(num_legs)]
        self.theta_actual_history = [113.0 * np.ones(num_legs)]  # Initial theta for ~5.3 N
        self.steady_state_factor_theta = np.zeros(num_legs)  # No steady-state error

    def pre_init(self, u_desired):
        # Pre-compute initial theta and force
        theta_desired = np.zeros(num_legs)
        for leg in range(num_legs):
            f_vec_desired = u_desired[3*leg:3*leg+3]
            f_leg_desired = R_leg[leg].T @ f_vec_desired
            fz_desired = f_leg_desired[2]
            if fz_desired > 0:
                theta_desired[leg] = (fz_desired + 6.91) / 0.107
            else:
                theta_desired[leg] = 0.0
        self.theta_actual_history[0] = theta_desired.copy()

    def apply(self, u_desired, t):
        u_actual = np.zeros(12)
        
        # Step 1: Extract desired fz (N_desired) for each leg
        theta_desired = np.zeros(num_legs)
        for leg in range(num_legs):
            f_vec_desired = u_desired[3*leg:3*leg+3]
            f_leg_desired = R_leg[leg].T @ f_vec_desired
            fz_desired = f_leg_desired[2]  # Desired normal force
            
            # Step 2: Compute desired theta using adjusted inverse model
            if fz_desired > 0:
                theta_desired[leg] = (fz_desired + 6.91) / 0.107  # Adjusted for fz = 0.107 * theta - 6.91
            else:
                theta_desired[leg] = 0.0  # Leg not in contact

        # Step 3: Apply dynamics
        self.theta_desired_history.append(theta_desired.copy())
        if len(self.theta_desired_history) > 2:
            self.theta_desired_history.pop(0)
        theta_prev = self.theta_actual_history[-1]
        theta_current = self.beta * theta_prev + self.alpha * theta_desired
        self.theta_actual_history.append(theta_current)
        if len(self.theta_actual_history) > 2:
            self.theta_actual_history.pop(0)
        
        theta_actual = theta_current  # No steady-state error

        # Step 4: Compute actual normal force (fz_actual) using the adjusted model
        for leg in range(num_legs):
            if theta_actual[leg] >= 0:
                fz_actual = max(0.107 * theta_actual[leg] - 6.91, 0)  # Original model
            else:
                fz_actual = 0.0

            # Step 5: Estimate actual fx and fy using friction model
            f_vec_desired = u_desired[3*leg:3*leg+3]
            f_leg_desired = R_leg[leg].T @ f_vec_desired
            phase = int((t % 5) >= 2.5)
            mu_leg = 0.6 if (leg == 0 and phase == 0) or (leg == 1 and phase == 1) else 0.2 if leg < 2 else 0.1

            if fz_actual > 0:
                f_tangential_desired = np.array([f_leg_desired[0], f_leg_desired[1]])
                norm_tangential_desired = np.linalg.norm(f_tangential_desired)
                f_tangential_max = mu_leg * fz_actual
                if norm_tangential_desired <= f_tangential_max:
                    f_leg_actual = f_leg_desired.copy()
                else:
                    if norm_tangential_desired > 0:
                        scale = f_tangential_max / norm_tangential_desired
                        f_leg_actual = f_leg_desired * scale
                    else:
                        f_leg_actual = np.zeros(3)
                f_leg_actual[2] = fz_actual
            else:
                f_leg_actual = np.zeros(3)

            # Transform back to body frame
            f_vec_actual = R_leg[leg] @ f_leg_actual
            u_actual[3*leg:3*leg+3] = f_vec_actual

        return u_actual

# Forward Dynamics (FD) and State Estimator
def forward_dynamics(state, u_actual, fd_horizon):
    current_state = state.copy()
    for _ in range(fd_horizon):
        current_state = A_d @ current_state + B_d @ u_actual
    return current_state

# Combined Simulation
def simulate():
    llc = LowLevelControl()
    state = np.zeros(13)
    state[5] = 0.0  # Initial height matches desired_pz
    state[12] = -g   # Gravity term
    t = 0.0
    sim_time = 40.0
    mpc_steps_per_run = 20
    fd_steps_per_run = fd_horizon
    total_steps_per_run = mpc_steps_per_run + fd_steps_per_run
    time_per_run = total_steps_per_run * dt
    total_runs = int(sim_time / time_per_run)

    # Initial MPC control to pre-compute forces
    initial_u_desired = mpc_control(state, t)
    llc.pre_init(initial_u_desired)

    times = []
    pz_values = []
    vx_values = []
    vy_values = []
    vz_values = []
    roll_values = []
    pitch_values = []
    yaw_values = []
    fz_values = [[] for _ in range(num_legs)]
    fx_leg_values = [[] for _ in range(num_legs)]
    fy_leg_values = [[] for _ in range(num_legs)]
    theta_desired_values = [[] for _ in range(num_legs)]
    theta_actual_values = [[] for _ in range(num_legs)]
    mu_values = []

    initial_force = np.zeros(12)
    initial_force[2] = initial_force[5] = initial_force[8] = initial_force[11] = mass * g / num_legs + 0.2  # Overforce for first 3 steps

    for run in range(total_runs):
        print(f"\n--- Run {run} starting at t={t:.3f} ---")
        current_state = state.copy()
        last_u_actual = initial_force if run == 0 and t < 3 * dt else np.zeros(12)
        for mpc_step in range(mpc_steps_per_run):
            print(f"Run {run}, MPC Step {mpc_step}, t={t:.3f}")
            u_desired = mpc_control(current_state, t)
            print(f"MPC Desired Forces: {u_desired}")
            u_actual = llc.apply(u_desired, t)
            # Override for first 3 steps
            if run == 0 and t < 3 * dt:
                for leg in range(num_legs):
                    u_actual[3*leg+2] = max(u_actual[3*leg+2], mass * g / num_legs + 0.2)
            last_u_actual = u_actual.copy()
            print(f"LLC Actual Forces: {u_actual}")
            current_state = A_d @ current_state + B_d @ u_actual
            print(f"State after MPC step: pz={current_state[5]:.3f}, vx={current_state[9]:.3f}, vy={current_state[10]:.3f}, vz={current_state[11]:.3f}, roll={current_state[0]:.3f}, pitch={current_state[1]:.3f}, yaw={current_state[2]:.3f}")
            phase = int((t % 5) >= 2.5)
            mu_current = [0.6 if phase == 0 else 0.2, 0.2 if phase == 0 else 0.6, 0.1, 0.1]
            mu_values.append(mu_current[0])
            times.append(t)
            pz_values.append(current_state[5])
            vx_values.append(current_state[9])
            vy_values.append(current_state[10])
            vz_values.append(current_state[11])
            roll_values.append(current_state[0])
            pitch_values.append(current_state[1])
            yaw_values.append(current_state[2])
            for leg in range(num_legs):
                f_vec = u_actual[3*leg:3*leg+3]
                f_leg = R_leg[leg].T @ f_vec
                fz_values[leg].append(f_vec[2])
                fx_leg_values[leg].append(f_leg[0])
                fy_leg_values[leg].append(f_leg[1])
                theta_desired_values[leg].append(llc.theta_desired_history[-1][leg])
                theta_actual_values[leg].append(llc.theta_actual_history[-1][leg])
            print(f"Run {run}, MPC Step {mpc_step}: t={t:.3f}, pz={current_state[5]:.3f}, vx={current_state[9]:.3f}, roll={current_state[0]:.3f}, mu_FL={mu_current[0]}")
            print(f"Leg0: fz={u_actual[2]:.2f}, fy={u_actual[1]:.2f}, fx={u_actual[0]:.3f}, "
                  f"Leg1: fz={u_actual[5]:.2f}, fy={u_actual[4]:.2f}, fx={u_actual[3]:.3f}, "
                  f"Leg2: fz={u_actual[8]:.2f}, fy={u_actual[7]:.2f}, fx={u_actual[6]:.3f}, "
                  f"Leg3: fz={u_actual[11]:.2f}, fy={u_actual[10]:.2f}, fx={u_actual[9]:.3f}")
            t += dt
        print(f"Run {run}, Running Forward Dynamics for {fd_steps_per_run} steps...")
        state = forward_dynamics(current_state, last_u_actual, fd_horizon)
        print(f"State after FD: pz={state[5]:.3f}, vx={state[9]:.3f}, vy={state[10]:.3f}, vz={state[11]:.3f}, roll={state[0]:.3f}, pitch={state[1]:.3f}, yaw={state[2]:.3f}")
        for fd_step in range(fd_steps_per_run):
            phase = int((t % 5) >= 2.5)
            mu_current = [0.6 if phase == 0 else 0.2, 0.2 if phase == 0 else 0.6, 0.1, 0.1]
            mu_values.append(mu_current[0])
            times.append(t)
            pz_values.append(state[5])
            vx_values.append(state[9])
            vy_values.append(state[10])
            vz_values.append(state[11])
            roll_values.append(state[0])
            pitch_values.append(state[1])
            yaw_values.append(state[2])
            for leg in range(num_legs):
                f_vec = last_u_actual[3*leg:3*leg+3]
                f_leg = R_leg[leg].T @ f_vec
                fz_values[leg].append(f_vec[2])
                fx_leg_values[leg].append(f_leg[0])
                fy_leg_values[leg].append(f_leg[1])
                theta_desired_values[leg].append(llc.theta_desired_history[-1][leg])
                theta_actual_values[leg].append(llc.theta_actual_history[-1][leg])
            print(f"Run {run}, FD Step {fd_step}: t={t:.3f}, pz={state[5]:.3f}, vx={state[9]:.3f}, roll={state[0]:.3f}, mu_FL={mu_current[0]}")
            print(f"Leg0: fz={last_u_actual[2]:.2f}, fy={last_u_actual[1]:.2f}, fx={last_u_actual[0]:.3f}, "
                  f"Leg1: fz={last_u_actual[5]:.2f}, fy={last_u_actual[4]:.2f}, fx={last_u_actual[3]:.3f}, "
                  f"Leg2: fz={last_u_actual[8]:.2f}, fy={last_u_actual[7]:.2f}, fx={last_u_actual[6]:.3f}, "
                  f"Leg3: fz={last_u_actual[11]:.2f}, fy={last_u_actual[10]:.2f}, fx={last_u_actual[9]:.3f}")
            t += dt

    # Plotting the four requested plots
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = 'black'

    # Plot 1: vx vs time
    plt.figure(figsize=(8, 5))
    plt.plot(times, vx_values, 'b-', linewidth=3, label='vx')
    plt.axhline(y=0.02, color='r', linestyle='--', linewidth=3, label='Desired vx = 0.02 m/s')
    plt.xlabel('Time (s)', fontweight='bold')
    plt.ylabel('vx (m/s)', fontweight='bold')
    plt.title('Forward Velocity (vx) vs Time', fontweight='bold')
    plt.grid(True, linewidth=1.5)
    plt.legend(loc='best', prop={'weight': 'bold'})
    plt.ylim(0, 0.06)  # Set y-axis limits for vx
    plt.yticks(np.arange(0, 0.06, 0.02))  # Ticks every 0.02 m/s
    plt.tight_layout()

    # Plot 2: Height (pz) vs time
    plt.figure(figsize=(8, 5))
    plt.plot(times, pz_values, 'r-', linewidth=3, label='Height (pz)')
    plt.axhline(y=0.1, color='b', linestyle='--', linewidth=3, label='Desired pz = 0.1 m')
    plt.xlabel('Time (s)', fontweight='bold')
    plt.ylabel('Height (m)', fontweight='bold')
    plt.title('Height (pz) vs Time', fontweight='bold')
    plt.grid(True, linewidth=1.5)
    plt.legend(loc='best', prop={'weight': 'bold'})
    plt.ylim(0, 0.15)  # Set y-axis limits for height
    plt.yticks(np.arange(0, 0.15, 0.05))  # Ticks every 0.05 m
    plt.tight_layout()

    # Plot 3: vy vs time
    plt.figure(figsize=(8, 5))
    plt.plot(times, vy_values, 'g-', linewidth=3, label='vy')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=3, label='Desired vy = 0 m/s')
    plt.xlabel('Time (s)', fontweight='bold')
    plt.ylabel('vy (m/s)', fontweight='bold')
    plt.title('Lateral Velocity (vy) vs Time', fontweight='bold')
    plt.grid(True, linewidth=1.5)
    plt.legend(loc='best', prop={'weight': 'bold'})
    plt.ylim(-0.2, 0.2)  # Set y-axis limits for vy
    plt.yticks(np.arange(-0.2, 0.2, 0.05))  # Ticks every 0.02 m/s
    plt.tight_layout()

    # Plot 4: Desired angles for 4 legs vs time
    plt.figure(figsize=(8, 5))
    for leg in range(num_legs):
        plt.plot(times, theta_desired_values[leg], linewidth=3,
                label=f'Leg {leg} Theta Desired',
                color=['red', 'blue', 'green', 'orange'][leg])
    plt.xlabel('Time (s)', fontweight='bold')
    plt.ylabel('Desired Angle (degrees)', fontweight='bold')
    plt.title('Desired Angles for Legs vs Time', fontweight='bold')
    plt.grid(True, linewidth=1.5)
    plt.legend(loc='best', prop={'weight': 'bold'})
    plt.ylim(110, 120)  # Set y-axis limits for angles
    plt.yticks(np.arange(110, 120, 2))  # Ticks every 2 degrees
    plt.tight_layout()

    # Save all plots
    plt.savefig('quadruped_mpc_plots.png')
    print("Plots saved as 'quadruped_mpc_plots.png'")

    # Show plots
    plt.show()

if __name__ == "__main__":
    simulate()
