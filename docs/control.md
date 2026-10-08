# Control Architecture Notes

The controller is cascaded because translational motion and rotational motion occur at different conceptual layers.

1. Position/velocity error creates desired inertial acceleration.
2. Desired horizontal acceleration maps to desired roll and pitch.
3. Attitude error maps to desired body rates.
4. Body-rate PID maps rate error to body torque.

This mirrors the hierarchy used in many multirotor control stacks: an outer loop asks *where should the vehicle go?*, and a faster inner loop asks *how should the vehicle rotate to make that happen?*
