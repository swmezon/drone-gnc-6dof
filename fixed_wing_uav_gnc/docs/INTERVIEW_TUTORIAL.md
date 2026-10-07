# Fixed-Wing GNC Interview Tutorial

This guide is intentionally written as if you are learning the system for the first time. The goal is not to memorize code. The goal is to be able to look at each equation or line and explain **what physical problem it solves, why it is there, and what would go wrong without it**.

---

# 1. The complete story

Imagine a small airplane flying without a human pilot.

1. The **6-DOF plant** is the real airplane in our virtual world.
2. The **IMU and GPS** are the airplane's senses.
3. The **EKF** is the airplane's best guess of where it is and how it is moving.
4. **Guidance** is the navigator saying, "fly toward that waypoint."
5. The **autopilot** decides how much aileron, elevator, rudder, and throttle to command.
6. **Linearization and margins** ask whether small disturbances around cruise are robustly controlled.
7. **Monte Carlo** asks whether the design still works when the airplane, sensors, and atmosphere are not exactly nominal.

A recruiter sees one GitHub repository. A GNC engineer should see a complete estimation/control/verification chain.

---

# 2. Coordinate frames: why we need two worlds

We use an inertial North-East-Down (NED) frame for position and a body frame fixed to the airplane for aerodynamic forces and angular rates.

The airplane's body axes are approximately:

- body x: nose forward
- body y: right wing
- body z: down

The state is

\[
\mathbf{x}
=
\begin{bmatrix}
p_N&p_E&p_D&u&v&w&\phi&\theta&\psi&p&q&r\end{bmatrix}^{T}.
\]

The first three numbers tell us **where** the airplane is. The next three tell us how fast it moves along its own body axes. The Euler angles tell us how the body frame is rotated relative to NED. The final three are body angular rates.

Analogy: NED is the map on the wall. The body frame is a coordinate system painted on the airplane. If the airplane rolls 90 degrees, "up on the map" and "up relative to the airplane" are no longer the same direction. Rotation matrices translate between those viewpoints.

---

# 3. Translational equations: deriving `vel_dot`

Newton's second law in an inertial frame is

\[
\mathbf{F}=m\mathbf{a}.
\]

But our velocity vector is expressed in the rotating body frame. A derivative observed in a rotating frame needs an extra transport term:

\[
\left(\frac{d\mathbf{v}}{dt}\right)_I
=
\left(\frac{d\mathbf{v}}{dt}\right)_B
+
\boldsymbol{\omega}\times\mathbf{v}.
\]

Substitute that into Newton's law:

\[
\mathbf{F}_B
=
m
\left(
\dot{\mathbf{v}}_B
+
\boldsymbol{\omega}_B\times\mathbf{v}_B
\right).
\]

Solve for the body-frame velocity derivative:

\[
\dot{\mathbf{v}}_B
=
\frac{\mathbf{F}_B}{m}
-
\boldsymbol{\omega}_B\times\mathbf{v}_B.
\]

That is exactly why `dynamics.py` contains:

```python
vel_dot = force_b / p.mass - np.cross(omega, vel_b)
```

Analogy: walk forward inside a rotating merry-go-round. Even if you walk straight relative to the platform, an observer on the ground sees your direction continuously changing. The cross-product term accounts for the fact that the coordinate system itself rotates.

---

# 4. Rotational equations: deriving `omega_dot`

Angular momentum is

\[
\mathbf{H}=\mathbf{I}\boldsymbol{\omega}.
\]

The rotational form of Newton's second law is

\[
\mathbf{M}
=
\left(\frac{d\mathbf{H}}{dt}\right)_I.
\]

Again, the body frame rotates, so

\[
\mathbf{M}
=
\mathbf{I}\dot{\boldsymbol{\omega}}
+
\boldsymbol{\omega}\times
\left(\mathbf{I}\boldsymbol{\omega}\right).
\]

Solve for angular acceleration:

\[
\dot{\boldsymbol{\omega}}
=
\mathbf{I}^{-1}
\left[
\mathbf{M}
-
\boldsymbol{\omega}\times
\left(\mathbf{I}\boldsymbol{\omega}\right)
\right].
\]

That becomes:

```python
omega_dot = np.linalg.solve(
    inertia,
    moment_b - np.cross(omega, inertia @ omega)
)
```

`np.linalg.solve(I, rhs)` means "find the vector x that satisfies I x = rhs." It is numerically preferable to explicitly computing `inv(I) @ rhs`.

---

# 5. Aerodynamics: from air motion to forces

The airplane reacts to **air-relative** velocity, not merely ground-relative velocity. If the aircraft moves north at 25 m/s while the wind moves north at 5 m/s, the wing feels roughly 20 m/s of airspeed.

Therefore

\[
\mathbf{v}_{air,B}
=
\mathbf{v}_{ground,B}
-
\mathbf{R}_{N}^{B}\mathbf{v}_{wind,N}.
\]

Airspeed is

\[
V_a
=
\sqrt{u_a^2+v_a^2+w_a^2}.
\]

Angle of attack is

\[
\alpha
=
\tan^{-1}\left(\frac{w_a}{u_a}\right).
\]

Sideslip is

\[
\beta
=
\sin^{-1}\left(\frac{v_a}{V_a}\right).
\]

Dynamic pressure follows from fluid mechanics:

\[
\bar q
=
\frac{1}{2}\rho V_a^2.
\]

Lift, drag, and moments are dynamic pressure times geometry times nondimensional coefficients:

\[
L=\bar q S C_L,
\qquad
D=\bar q S C_D,
\qquad
M=\bar q S c C_m.
\]

Why use coefficients? Because they separate the airplane's shape/aerodynamic behavior from the current air density and speed.

---

# 6. `aircraft.py`: what the numbers mean

`AircraftParams` is the airplane's specification sheet.

```python
mass = 13.5
```

The vehicle weighs 13.5 kg in the model. Increasing mass makes the same force produce less acceleration.

```python
S = 0.55
b = 2.90
c = 0.19
```

`S` is wing reference area, `b` is wingspan, and `c` is reference chord. Aerodynamic forces scale with area. Roll/yaw moments scale with span, and pitch moment scales with chord.

```python
Jx, Jy, Jz, Jxz
```

These are moments/products of inertia. They are the rotational equivalent of mass. A large `Jy` means it takes more pitching moment to create the same pitch acceleration.

The `CL_*`, `Cm_*`, `Cl_*`, and `Cn_*` values are stability/control derivatives. For example,

\[
C_L
=
C_{L0}
+C_{L_\alpha}\alpha
+C_{L_q}\frac{c}{2V_a}q
+C_{L_{\delta_e}}\delta_e.
\]

Read this verbally as: "lift coefficient changes because of baseline lift, angle of attack, pitch rate, and elevator deflection."

---

# 7. Trim: why we solve before controlling

Trim means an equilibrium flight condition. For straight-and-level cruise, we want approximately zero acceleration and zero pitch acceleration.

The trim solver chooses

\[
\mathbf{z}
=
\begin{bmatrix}
\alpha & \theta & \delta_e & \delta_t
\end{bmatrix}^{T}
\]

and minimizes residuals such as

\[
\dot u\approx0,
\qquad
\dot w\approx0,
\qquad
\dot q\approx0,
\qquad
\theta-\alpha\approx0.
\]

The final residual in `trim.py` is

```python
return np.array([dx[3], dx[5], dx[10], z[1] - z[0]])
```

The first three values ask for no longitudinal acceleration and no pitch acceleration. The last enforces approximately level flight-path angle: if pitch angle equals angle of attack, the velocity vector is approximately horizontal.

`least_squares` is like repeatedly turning four knobs until the four error meters are as close to zero as possible.

---

# 8. Numerical linearization: where A and B come from

The nonlinear model is

\[
\dot{\mathbf{x}}
=
\mathbf{f}(\mathbf{x},\mathbf{u}).
\]

Near trim, write

\[
\mathbf{x}=\mathbf{x}_0+\delta\mathbf{x},
\qquad
\mathbf{u}=\mathbf{u}_0+\delta\mathbf{u}.
\]

Use a first-order Taylor expansion:

\[
\delta\dot{\mathbf{x}}
\approx
\mathbf{A}\delta\mathbf{x}
+
\mathbf{B}\delta\mathbf{u},
\]

where

\[
\mathbf{A}
=
\left.\frac{\partial\mathbf{f}}{\partial\mathbf{x}}\right|_0,
\qquad
\mathbf{B}
=
\left.\frac{\partial\mathbf{f}}{\partial\mathbf{u}}\right|_0.
\]

The code does not symbolically differentiate a giant equation. It nudges one variable positively and negatively and measures the slope:

\[
\frac{\partial f}{\partial x_i}
\approx
\frac{f(x_i+h)-f(x_i-h)}{2h}.
\]

Analogy: to estimate the slope of a hill where you stand, take one tiny step uphill and one tiny step downhill, compare elevations, and divide by the distance between the two samples.

---

# 9. Eigenvalues and flying qualities

For the unforced linearized dynamics

\[
\delta\dot{\mathbf{x}}
=
\mathbf{A}\delta\mathbf{x},
\]

solutions contain terms like

\[
e^{\lambda t}.
\]

If the real part of an eigenvalue is negative, the exponential shrinks. If positive, it grows.

For a complex pair

\[
\lambda
=
\sigma\pm j\omega_d,
\]

natural frequency is

\[
\omega_n
=
\sqrt{\sigma^2+\omega_d^2},
\]

and damping ratio is

\[
\zeta
=
-\frac{\sigma}{\omega_n}.
\]

The familiar fixed-wing modes are short-period, phugoid, Dutch roll, roll subsidence, and spiral. This proof-of-concept currently reports eigenvalue properties; a next enhancement is automatic modal labeling.

Interview-safe statement: "I linearized the nonlinear rigid-body model around straight-and-level trim and examined the local eigenstructure."

---

# 10. Cascaded autopilot: why there are loops inside loops

Guidance produces a desired bank angle and pitch angle. The autopilot then converts angle error into desired angular rate, and angular-rate error into control-surface deflection.

For roll:

\[
p_c
=
K_{\phi}
(\phi_c-\phi).
\]

Then

\[
e_p=p_c-p.
\]

Then the aileron command is roughly

\[
\delta_a
=
K_p e_p
+K_i\int e_pdt
-K_d p.
\]

Why cascade? The body rate responds much faster than vehicle position. The inner loop acts like a fast stabilizing reflex; the outer loop gives slower strategic instructions.

Analogy: when driving, "stay in your lane" is an outer objective. Your hands rapidly adjust steering angle in an inner loop.

---

# 11. Actuator magnitude and rate limits

Real control surfaces cannot rotate infinitely far or teleport from one angle to another.

Magnitude limit:

\[
|\delta_a|\le\delta_{a,max}.
\]

Rate limit:

\[
|\dot{\delta}_a|\le\dot{\delta}_{a,max}.
\]

The code first clips the requested target to the angle limit, then clips the change per time step to the rate limit.

This is nonlinear behavior. Therefore it is evaluated in the time-domain nonlinear simulation, **not folded into the classical gain/phase-margin claim**.

---

# 12. Guidance: turning a waypoint into a bank command

For a waypoint with north/east coordinates, desired course is

\[
\chi_d
=
\tan^{-1}\left(
\frac{E_{wp}-E}{N_{wp}-N}
\right).
\]

In code we use `np.arctan2(y, x)` rather than simple `atan(y/x)` because `arctan2` knows the quadrant and safely handles zero denominators.

Course error is wrapped into \([-\pi,\pi]\), then multiplied by a gain to produce desired bank angle.

Analogy: a navigation app says the destination is to your right. It does not directly turn the steering wheel; it tells the low-level controller what direction it wants.

---

# 13. Sensor models: truth is hidden from the controller

A gyro measurement is modeled as

\[
\tilde{\boldsymbol{\omega}}
=
\boldsymbol{\omega}
+
\mathbf{b}_g
+
\mathbf{n}_g.
\]

The bias is a persistent offset. Noise changes randomly from sample to sample.

An accelerometer measures specific force, not simply inertial acceleration:

\[
\mathbf{f}_B
=
\dot{\mathbf{v}}_B
+
\boldsymbol{\omega}\times\mathbf{v}_B
-
\mathbf{R}_{N}^{B}\mathbf{g}_N.
\]

GPS supplies noisy NED position and velocity at a much slower rate.

The simulation propagates at 50 Hz and GPS updates approximately at 5 Hz. This creates the core sensor-fusion problem: fast drifting inertial data plus slower absolute corrections.

---

# 14. Why the EKF has 15 states

The navigation estimator uses

\[
\mathbf{x}_{nav}
=
\begin{bmatrix}
\mathbf{p}^{T}&
\mathbf{v}^{T}&
\boldsymbol{\eta}^{T}&
\mathbf{b}_{g}^{T}&
\mathbf{b}_{a}^{T}
\end{bmatrix}^{T}.
\]

Count them:

- position: 3
- velocity: 3
- attitude: 3
- gyro bias: 3
- accelerometer bias: 3

Total: 15.

The filter is not magical. It carries two things:

1. `x`: its best estimate.
2. `P`: how uncertain it is about that estimate.

---

# 15. EKF prediction

The nonlinear propagation model uses bias-corrected gyro and accelerometer measurements:

\[
\boldsymbol{\omega}
=
\tilde{\boldsymbol{\omega}}-\hat{\mathbf{b}}_g,
\]

\[
\mathbf{a}_N
=
\mathbf{R}_{B}^{N}
\left(
\tilde{\mathbf{f}}_B-\hat{\mathbf{b}}_a
\right)
+
\mathbf{g}_N.
\]

Then

\[
\dot{\mathbf{p}}=\mathbf{v},
\qquad
\dot{\mathbf{v}}=\mathbf{a}_N.
\]

Covariance prediction is

\[
\mathbf{P}_{k}^{-}
=
\mathbf{F}_{k}
\mathbf{P}_{k-1}^{+}
\mathbf{F}_{k}^{T}
+
\mathbf{Q}.
\]

`Q` means uncertainty introduced by the process/model between measurements.

Analogy: every minute you walk with your eyes closed, your uncertainty about your location grows.

---

# 16. EKF GPS correction

GPS measures position and velocity, so

\[
\mathbf{z}
=
\mathbf{H}\mathbf{x}
+
\mathbf{v}.
\]

Innovation:

\[
\mathbf{y}
=
\mathbf{z}
-
\mathbf{H}\hat{\mathbf{x}}^{-}.
\]

This is simply "what GPS said minus what I predicted."

Innovation covariance:

\[
\mathbf{S}
=
\mathbf{H}\mathbf{P}^{-}\mathbf{H}^{T}
+
\mathbf{R}.
\]

Kalman gain:

\[
\mathbf{K}
=
\mathbf{P}^{-}\mathbf{H}^{T}\mathbf{S}^{-1}.
\]

Correction:

\[
\hat{\mathbf{x}}^{+}
=
\hat{\mathbf{x}}^{-}
+
\mathbf{K}\mathbf{y}.
\]

`R` represents measurement uncertainty. If `R` is huge, the filter distrusts GPS. If `R` is tiny, the filter follows GPS aggressively.

---

# 17. RMSE and convergence

At every time step, position error magnitude is

\[
e_p(k)
=
\left\|
\mathbf{p}_{true}(k)-\hat{\mathbf{p}}(k)
\right\|_2.
\]

Position RMSE is

\[
RMSE_p
=
\sqrt{
\frac{1}{N}
\sum_{k=1}^{N}e_p^2(k)
}.
\]

Why square? Negative and positive component errors cannot cancel, and large errors are penalized more strongly.

The nominal convergence requirement is stricter than simply crossing 1 m once: the position error must remain below 1 m for three continuous seconds. That prevents a noisy accidental threshold crossing from being called convergence.

---

# 18. Gain margin, phase margin, and crossover

For one local loop,

\[
L(s)=C(s)G(s)G_{act}(s)H(s).
\]

Gain crossover is the frequency where

\[
|L(j\omega_{gc})|=1.
\]

At that frequency, phase margin is

\[
PM
=
180^{\circ}
+
\angle L(j\omega_{gc}).
\]

Phase crossover is where loop phase reaches \(-180^{\circ}\). Gain margin is the reciprocal of loop magnitude there, commonly expressed in dB.

The current nominal local roll/pitch rate loops cross unity at approximately 3.94–4.77 rad/s. That is useful context because an enormous margin at an extremely tiny crossover could simply mean the loop is too slow.

Interview question to expect: "What does that crossover mean physically?" Answer: it is roughly the frequency where the loop transitions from strongly correcting lower-frequency disturbances to having much less authority over higher-frequency motion.

---

# 19. Monte Carlo: why one successful flight proves very little

A deterministic simulation asks, "does this one perfectly specified airplane work?"

Monte Carlo asks, "does a population of slightly different airplanes work?"

For each case this project redraws sensor noise/bias, wind, mass, inertia, and selected aerodynamic derivatives.

A parameter is scaled approximately as

\[
p_i^{MC}
=
p_i^{nom}
\left(1+\epsilon_i\right),
\]

with

\[
\epsilon_i\sim\mathcal{N}(0,0.05^2)
\]

and clipping at ±15%.

The pass requirements are intentionally written before reading the result:

- position RMSE < 5 m
- convergence < 8 s
- surface saturation fraction < 10%
- no numerical divergence

This prevents moving the goalposts after seeing the data.

---

# 20. `scripts/run_all.py` from top to bottom

The imports are the toolbox:

```python
from pathlib import Path
import argparse
import csv
import json
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
```

- `Path`: reliable file/folder paths.
- `argparse`: lets the user type `--mc-cases 500` and `--jobs 8`.
- `csv`/`json`: save machine-readable evidence.
- `multiprocessing`: run independent Monte Carlo cases on several CPU cores.
- `numpy`: vectors/matrices/numerical calculations.
- `matplotlib`: figures.

Next, imports from `src` pull in the engineering blocks. This separation is important: `src` contains reusable engineering logic; `scripts` orchestrates experiments.

```python
ROOT = Path(__file__).resolve().parents[1]
```

`__file__` is this script's location. `resolve()` makes it absolute. `parents[1]` moves up from `scripts/run_all.py` to the project root.

```python
FIG = ROOT / "results" / "figures"
DATA = ROOT / "results" / "data"
```

These are output folders. The `/` operator on a `Path` joins path pieces; it is not numerical division here.

```python
FIG.mkdir(parents=True, exist_ok=True)
```

Create the folder if it does not exist. `parents=True` creates missing parent folders. `exist_ok=True` means rerunning the script is not an error.

`save_trim_validation(...)` simulates the open-loop airplane from trim for 20 seconds. If altitude immediately runs away, the "equilibrium" is not trustworthy.

`save_eigen_analysis(A)` computes eigenvalues of the trim-linearized state matrix and saves them to CSV and a complex-plane figure.

`analyze_rate_loops(A, B)` extracts the local body-rate dynamics from the linearized matrices. `A[9,9]` is the sensitivity of roll-rate derivative to roll rate. `B[9,0]` is the sensitivity of roll-rate derivative to aileron. The same logic is used for pitch/elevator.

`save_mission_plots(...)` creates human-readable evidence. A recruiter can see waypoint following and estimator convergence without reading code.

`monte_carlo_case(i)` creates one independent uncertain universe. The seed `1000+i` makes the result reproducible: case 23 is always the same random case when code/settings are unchanged.

`run_monte_carlo(N, jobs)` repeats those universes, collects metrics, saves every trial to CSV, and generates the RMSE distribution plot.

Inside `main()`:

```python
parser.add_argument("--mc-cases", type=int, default=500)
```

The default final validation is 500 cases, but you can use 20 or 50 while debugging.

```python
p = nominal_aircraft()
x0, u0, sol = trim_state_control(p)
```

Create the baseline airplane and find its equilibrium state/control.

```python
A, B = linearize(x0, u0, p)
```

Create the local linear model.

```python
mission = run_mission(seed=7)
```

Run the nominal nonlinear closed-loop navigation demonstration.

```python
arr = run_monte_carlo(args.mc_cases, jobs=args.jobs)
```

Run robustness verification.

Finally `summary` stores the exact values that may be copied into a résumé. `json.dumps(..., indent=2)` makes them human-readable.

The final guard

```python
if __name__ == "__main__":
    main()
```

means: run `main()` when this file is executed as a program, but do not automatically launch the experiment when another Python module imports this file. This guard is especially important for multiprocessing on Windows.

---

# 21. Questions you should be able to answer after studying the repo

1. Why are aerodynamic forces based on air-relative rather than ground-relative velocity?
2. Why does the translational equation contain \(-\boldsymbol{\omega}\times\mathbf{v}\)?
3. What does a moment of inertia mean physically?
4. What makes an aircraft "trimmed"?
5. Why linearize about trim instead of using the nonlinear model directly for Bode margins?
6. What information is contained in A and B?
7. Why is the autopilot cascaded?
8. What is the difference between sensor bias and white noise?
9. Why does an IMU solution drift without absolute aiding?
10. Why does GPS help position/velocity but not directly measure attitude?
11. What are Q and R in an EKF?
12. What is an innovation?
13. Why is the Kalman gain not a manually fixed controller gain?
14. What does gain crossover mean?
15. Why must actuator saturation be evaluated separately from linear gain/phase margins?
16. Why define Monte Carlo pass criteria before running the trials?
17. Why are 500 trials more informative than 50 for percentile reporting?
18. What uncertainty did you actually vary, and what did you *not* vary?

If you can answer these in your own words, the GitHub project becomes evidence of understanding rather than merely evidence that code exists.
