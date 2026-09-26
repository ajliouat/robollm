"""Shared reset configuration for the custom simulated arm.

The pose was selected from forward kinematics and collider clearance on the
development scenes, not from held-out task success. It is not a calibrated
Franka Panda configuration. Keep the XML home keyframe synchronized with it.

The scene provides ideal body gravity compensation to the robot only. This
models an ideal support term, not actuator estimation or hardware validation;
the objects retain normal gravity and all contact handling remains enabled.
"""

import numpy as np


HOME_QPOS = np.array(
    [-np.pi / 2, -0.5, 0.0, -1.8, 0.0, 1.5, 0.785, 0.02, 0.02],
    dtype=np.float64,
)
HOME_QPOS.setflags(write=False)
