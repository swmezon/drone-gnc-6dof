import numpy as np


def _crossing_frequency(w, y, target=0.0):
    idx = np.where(np.diff(np.sign(y - target)) != 0)[0]
    if not len(idx):
        return float("nan"), None
    i = int(idx[0])
    # log-frequency interpolation is more appropriate for Bode data.
    x0, x1 = np.log(w[i]), np.log(w[i + 1])
    y0, y1 = y[i], y[i + 1]
    frac = 0.0 if y1 == y0 else (target - y0) / (y1 - y0)
    return float(np.exp(x0 + frac * (x1 - x0))), i


def local_rate_loop_analysis(a_rate, b_surface, controller_gain, actuator_tau=0.05, sensor_tau=0.02):
    """Classical SISO loop metrics for a local rate loop around trim.

    Local plant: G_rate(s) = b / (s - a)
    Actuator:     G_act(s)  = 1 / (tau_a s + 1)
    Sensor:       H(s)      = 1 / (tau_s s + 1)
    Controller:   C(s)      = K

    The open-loop transfer function is L = C * G_rate * G_act * H.
    """
    w = np.logspace(-3, 4, 200000)
    s = 1j * w
    L = (controller_gain * b_surface) / (
        (s - a_rate) * (1 + actuator_tau * s) * (1 + sensor_tau * s)
    )
    mag = np.abs(L)
    phase_deg = np.unwrap(np.angle(L)) * 180.0 / np.pi
    db = 20.0 * np.log10(np.maximum(mag, 1e-15))

    wgc, i_gc = _crossing_frequency(w, db, 0.0)
    pm = float("nan")
    if i_gc is not None:
        # interpolate the unwrapped phase at gain crossover
        pm = float(180.0 + np.interp(np.log(wgc), np.log(w), phase_deg))

    wpc, i_pc = _crossing_frequency(w, phase_deg, -180.0)
    gm_db = float("inf")
    if i_pc is not None:
        db_at_pc = float(np.interp(np.log(wpc), np.log(w), db))
        gm_db = -db_at_pc

    # Closed-loop complementary sensitivity T=L/(1+L); bandwidth is the first
    # -3 dB drop relative to its DC magnitude.
    T = L / (1.0 + L)
    t_db = 20.0 * np.log10(np.maximum(np.abs(T), 1e-15))
    dc_db = float(t_db[0])
    wbw, _ = _crossing_frequency(w, t_db, dc_db - 3.0)

    return {
        "gain_margin_db": float(gm_db),
        "phase_margin_deg": float(pm),
        "gain_cross_rad_s": float(wgc),
        "phase_cross_rad_s": float(wpc),
        "closed_loop_bandwidth_rad_s": float(wbw),
    }


def local_rate_loop_margins(a_rate, b_surface, controller_gain, actuator_tau=0.05, sensor_tau=0.02):
    """Backward-compatible tuple interface."""
    out = local_rate_loop_analysis(a_rate, b_surface, controller_gain, actuator_tau, sensor_tau)
    return (
        out["gain_margin_db"],
        out["phase_margin_deg"],
        out["gain_cross_rad_s"],
        out["phase_cross_rad_s"],
    )
