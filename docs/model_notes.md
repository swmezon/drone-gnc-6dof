# Model Notes

This project uses a classical 12-state Euler-angle rigid-body model for educational GNC development.

## Coordinate convention

- Inertial frame: right-handed, +z upward.
- Body frame: right-handed, attached to the vehicle.
- Positive collective thrust: body +z.
- Gravity: inertial -z.
- Orientation: Z-Y-X yaw-pitch-roll.

## Important limitation

Euler-angle kinematics become singular at pitch = +/- 90 degrees. This project intentionally retains Euler angles to make the classical 12-state formulation explicit. A quaternion attitude representation is a logical extension for aggressive flight.
