"""
Drone liftoff simulation + altitude estimation using a Kalman Filter (KF).
Cases:
(a) Truth (sim ground truth)
(b) Sensor only (no KF)
(c) KF with matched model (m_est = m_true)
(d) KF with mismatched model (m_est = 1.10 * m_true)

Values given from question prompt:
- mass (true) = 0.25 kg
- sampling rate = 200 Hz
- constant thrust mean = 2.7 N with thrust noise variance 0.25 N^2
- measurement variance per-sample ~ Uniform[0.01, 0.5] m^2 (sensor reports it)
- simulate 5 secondslik
- Discrete-time model:
    x_k = A x_{k-1} + B u_k
    h_k = C x_k + v_k
  where x = [z, v]^T, C=[1,0], A=[[1, dt],[0,1]], B=[[0.5 dt^2],[dt]], 
  and u_k = T_k/m_est - g for the estimator. 
  The ground truth uses a_true = (T_k + tau_T)/m_true - g, where tau_T ~ N(0, 0.25).
- KF process noise uses the input-noise view: Sigma_u = (sigma_T / m_est)^2, Q = B Sigma_u B^T.
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_and_estimate(
    m_true=0.25,
    fs=200.0,
    duration_s=5.0,
    thrust_mean=2.7,
    thrust_var=0.25,
    meas_var_min=0.01,
    meas_var_max=0.5,
    g=9.81,
    mass_mismatch_factor=1.10,
    seed=None
):
    ''' Simulate drone liftoff and estimate altitude using Kalman Filter. '''
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / fs
    N = int(duration_s * fs)

    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt],
                  [dt]])
    C = np.array([[1.0, 0.0]])

    sigma_T = np.sqrt(thrust_var)
    T_seq = np.full(N, thrust_mean)
    time = np.arange(N) * dt
    x_true = np.zeros((2, N))   
    z_meas = np.zeros(N)
    R_seq = np.zeros(N)

    #simulate ground truth
    z, v = 0.0, 0.0
    for k in range(N):
        tau_T = np.random.normal(0.0, sigma_T)
        a_true = (T_seq[k] + tau_T) / m_true - g
        x = np.array([[z],[v]])
        x = A @ x + B * a_true
        z, v = float(x[0,0]), float(x[1,0])
        x_true[:, k] = [z, v]
        R_k = np.random.uniform(meas_var_min, meas_var_max)
        R_seq[k] = R_k
        z_meas[k] = z + np.random.normal(0.0, np.sqrt(R_k))

    def run_kf(m_est):
        ''' Run Kalman Filter with given mass estimate. '''
        Sigma_u = (sigma_T / m_est) ** 2
        Q = (B @ B.T) * Sigma_u
        x_est = np.zeros((2, N))
        P = np.eye(2)
        x = np.array([[0.0],[0.0]])

        for k in range(N):
            u_k = T_seq[k] / m_est - g
            x_pred = A @ x + B * u_k
            P_pred = A @ P @ A.T + Q

            R_k = R_seq[k]
            S = C @ P_pred @ C.T + R_k
            K = (P_pred @ C.T) / S
            innov = z_meas[k] - float(C @ x_pred)
            x = x_pred + K * innov
            P = (np.eye(2) - K @ C) @ P_pred
            x_est[:, k] = x.flatten()
        return x_est
    x_est_matched = run_kf(m_true)
    x_est_mismatch = run_kf(m_true * mass_mismatch_factor)
    return {
        "time": time,
        "z_truth": x_true[0, :],
        "z_sensor": z_meas,
        "z_kf_matched": x_est_matched[0, :],
        "z_kf_mismatch": x_est_mismatch[0, :],
    }

def main():
    res = simulate_and_estimate(seed=42)
    t = res["time"]

    # Plot altitudes
    plt.figure(figsize=(10, 5.5))
    plt.plot(t, res["z_truth"], color="black", linewidth=2.0, label="Truth (ground truth)")
    plt.plot(t, res["z_sensor"], color="deepskyblue", linewidth=0.5, label="Sensor (face value, no KF)")
    plt.plot(t, res["z_kf_matched"], color="orange", linewidth=1.0, label="KF (matched model)")
    plt.plot(t, res["z_kf_mismatch"], color="red", linewidth=1.0, label="KF (mass +10%)")
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [m]")
    plt.title("Drone Liftoff Altitudes")
    plt.legend()
    plt.grid(True)
    plt.xlim(0,5)
    plt.ylim(-0.5,res["z_truth"][-1]+1)
    plt.tight_layout()

    # Plot errors
    plt.figure(figsize=(10, 5.5))
    plt.plot(t, res["z_sensor"] - res["z_truth"], label="Sensor error")
    plt.plot(t, res["z_kf_matched"] - res["z_truth"], label="KF matched error")
    plt.plot(t, res["z_kf_mismatch"] - res["z_truth"], label="KF mass +10% error")
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude Error [m]")
    plt.title("Altitude Estimation Errors (relative to truth)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0,5)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()