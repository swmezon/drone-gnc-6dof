# Vehicle Dynamics Notes

The vehicle layer converts a state and a control vector into a state derivative. The current quadrotor uses a 12-state Newton-Euler model in a z-up inertial frame.

The key design choice is separation through `VehicleModel`: guidance and evaluation code should not need to know whether the plant is a quadrotor, fixed-wing aircraft, VTOL, or another vehicle.

State ordering for the quadrotor:

`[x, y, z, u, v, w, phi, theta, psi, p, q, r]`

Control ordering:

`[T, tau_x, tau_y, tau_z]`

`T` is collective thrust in newtons. `tau_x`, `tau_y`, and `tau_z` are body moments in N m.
