"""Real-hardware placeholder for Marvin Pro.

Simulation replay is supported.  A real controller must implement the same
``Dual_arm_controller`` API as the other robot adapters before RUN_MODE_REAL
can be enabled safely.
"""


class Dual_arm_controller:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Marvin Pro real-robot control is not configured; use RUN_MODE_REAL=0"
        )
