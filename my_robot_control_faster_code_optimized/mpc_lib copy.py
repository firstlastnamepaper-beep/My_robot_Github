#!/usr/bin/env python3
import numpy as np
import cvxpy as cp

# ─── Robot parameters ─────────────────────────────────
mass = 2.0         # kg
g = 9.81           # gravity
dt = 0.033         # control timestep
mpc_horizon = 15
f_max = 6.0
num_legs = 4

# Inertia tensor
x_len, y_len, z_len = 0.1255, 0.0855, 0.034
Ixx = (1/12)*mass*(y_len**2 + z_len**2)
Iyy = (1/12)*mass*(x_len**2 + z_len**2)
Izz = (1/12)*mass*(x_len**2 + y_len**2)
inertia = np.diag([Ixx, Iyy, Izz])

# Leg positions in body frame
leg_positions = np.array([
    [ x_len/2+0.19*np.sqrt(2),  y_len/2+0.19*np.sqrt(2), -(0.16+z_len/2)],
    [ x_len/2+0.19*np.sqrt(2), -y_len/2-0.19*np.sqrt(2), -(0.16+z_len/2)],
    [-x_len/2-0.19*np.sqrt(2),  y_len/2+0.19*np.sqrt(2), -(0.16+z_len/2)],
    [-x_len/2-0.19*np.sqrt(2), -y_len/2-0.19*np.sqrt(2), -(0.16+z_len/2)]
])

# Leg frame rotations (45°)
s2 = np.sqrt(2)/2
R_leg = [
    np.array([[ s2,  s2, 0], [-s2,  s2, 0], [0,0,1]]),
    np.array([[ s2, -s2, 0], [ s2,  s2, 0], [0,0,1]]),
    np.array([[-s2,  s2, 0], [-s2, -s2, 0], [0,0,1]]),
    np.array([[-s2, -s2, 0], [ s2, -s2, 0], [0,0,1]])
]

# Continuous → Discrete dynamics
A_c = np.zeros((13,13))
A_c[0:3,6:9] = np.eye(3)
A_c[3:6,9:12] = np.eye(3)
A_c[9:12,12] = [0,0,1]
A_c[9,9], A_c[10,10], A_c[11,11] = -0.1, -0.1, -0.5
A_d = np.eye(13) + dt*A_c

B_c = np.zeros((13,3*num_legs))
for i, r in enumerate(leg_positions):
    B_c[9:12,3*i:3*i+3] = (1/mass)*np.eye(3)
    Rx = np.array([[0,0,0],[0,0,-r[0]],[0,r[0],0]])
    Ry = np.array([[0,0,r[1]],[0,0,0],[-r[1],0,0]])
    Rz = np.array([[0,-r[2],0],[r[2],0,0],[0,0,0]])
    B_c[6:9,3*i:3*i+3] = np.linalg.inv(inertia) @ (Rx+Ry+Rz)
B_d = dt * B_c

# Cost weights
weights = np.array([
    0.1, 0.1, 0.1, 0.0, 0.0, 100000.0,
    0.01,0.01,0.01,20.0, 0.1, 0.0, 0.0
]) / 1000

# Reference trajectory
def get_reference_trajectory(t, horizon, dt, desired_vx=0.02, desired_pz=0.1):
    x_ref = np.zeros((13, horizon+1))
    for k in range(horizon+1):
        x_ref[9,k]  = desired_vx
        x_ref[3,k]  = desired_vx*(t + k*dt)
        x_ref[5,k]  = desired_pz
        x_ref[12,k] = -g
    return x_ref

def mpc_control(state, t):
    x0    = state.copy()
    x_ref = get_reference_trajectory(t, mpc_horizon, dt)
    gait_period = 5.0
    phase = int((t % gait_period) >= gait_period/2)
    mu    = [0.6 if i==phase else 0.2 if i<2 else 0.1
             for i in range(num_legs)]

    X = cp.Variable((13, mpc_horizon+1))
    U = cp.Variable((3*num_legs, mpc_horizon))
    cost = 0

    # build cost
    desired_fx = 0.6
    for k in range(mpc_horizon):
        e = X[:,k] - x_ref[:,k]
        cost += cp.quad_form(e, np.diag(weights))
        for leg in range(num_legs):
            f_vec = U[3*leg:3*leg+3, k]
            f_leg = R_leg[leg].T @ f_vec
            if leg < 2:
                if leg == phase:
                    cost += 0.01*cp.sum_squares(f_leg[0] - desired_fx)
                else:
                    cost += 0.01*cp.sum_squares(f_leg[0])
    cost += 1e-8*cp.sum_squares(U)

    # constraints
    cons = [X[:,0] == x0]
    for k in range(mpc_horizon):
        cons.append(X[:,k+1] == A_d@X[:,k] + B_d@U[:,k])
        total_fz = 0
        for leg in range(num_legs):
            f = U[3*leg:3*leg+3, k]
            f_leg = R_leg[leg].T @ f
            cons += [f[2] >= 0, f[2] <= f_max]
            if leg<2:
                if leg == phase:
                    cons += [f_leg[0] >= 0.2,
                             f_leg[0] <= mu[leg]*f_leg[2]]
                else:
                    cons += [
                        f_leg[2] >= 4.0,
                        f_leg[0] >= -mu[leg]*f_leg[2],
                        f_leg[0] <=  mu[leg]*f_leg[2]
                    ]
                cons.append(cp.abs(f_leg[1]) <= mu[leg]*f_leg[2])
            else:
                cons.append(cp.norm(f_leg[:2],2) <= mu[leg]*f_leg[2])
                cons.append(f_leg[2] >= 4.0)
            total_fz += f[2]
        cons += [
            total_fz >= mass*g - 0.01,
            total_fz <= mass*g + 0.01
        ]

    prob = cp.Problem(cp.Minimize(cost), cons)
    prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
    if prob.status != cp.OPTIMAL:
        return np.zeros(3*num_legs)
    return U.value[:,0]

class LowLevelControl:
    def __init__(self):
        self.tau   = 0.005
        self.alpha = 1 - np.exp(-dt/self.tau)
        self.beta  = 1 - self.alpha
        self.theta_actual_history = [113.0*np.ones(num_legs)]

    def pre_init(self, u_des):
        theta = np.zeros(num_legs)
        for i in range(num_legs):
            fz = max((R_leg[i].T @ u_des[3*i:3*i+3])[2], 0)
            theta[i] = (fz+6.91)/0.107 if fz>0 else 0.0
        self.theta_actual_history[0] = theta.copy()

    def apply(self, u_des, t):
        # 1) compute the “theoretical” θₜ in [0…116.28°]
        theta_d = np.zeros(num_legs)
        for i in range(num_legs):
            # vertical force in leg frame
            fz = max((R_leg[i].T @ u_des[3*i:3*i+3])[2], 0)
            theta_d[i] = (fz + 6.91)/0.107 if fz > 0 else 0.0

        # first‐order filter
        prev = self.theta_actual_history[-1]
        curr = self.beta * prev + self.alpha * theta_d
        self.theta_actual_history.append(curr.copy())
        if len(self.theta_actual_history) > 2:
            self.theta_actual_history.pop(0)

        # 2) remap [0…116.28] → [130…250] (your hardware stroke)
        theta_min_hw   = 130.0
        theta_max_hw   = 250.0
        theta_max_theo = 116.28
        clamped = np.clip(curr, 0.0, theta_max_theo)
        base_hw = theta_min_hw + (clamped/theta_max_theo)*(theta_max_hw - theta_min_hw)

        # 3) rebuild µ exactly as in mpc_control()
        gait_period = 0.5/7
        phase = int((t % gait_period) >= (gait_period/2))
        mu = [0.0]*num_legs
        # front‐left contracts in phase==0
        mu[0] = 0.6 if phase == 0 else 0.2
        # front‐right contracts in phase==1
        mu[1] = 0.6 if phase == 1 else 0.2
        # back legs always fully contract
        mu[2] = mu[3] = 0.6

        # 4) scale each leg’s excursion by µ/max(µ)=µ/0.6
        max_mu = 0.6
        swing = base_hw - theta_min_hw
        final_hw = theta_min_hw + swing * (np.array(mu) / max_mu)

        # 5) swap front <-> back so output is [BL, BR, FL, FR]
        swapped_hw = np.array([final_hw[2], final_hw[3],
                               final_hw[0], final_hw[1]])
        return swapped_hw

        # 5) return the 4 hardware‐scaled angles
        return final_hw
