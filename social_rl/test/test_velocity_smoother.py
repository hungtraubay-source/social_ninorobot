"""Pure regression tests for the train/deployment velocity limiter."""

import unittest

from social_rl.velocity_smoother import VelocitySmoother


class VelocitySmootherTest(unittest.TestCase):

    def test_limits_linear_and_angular_change_per_control_tick(self):
        smoother = VelocitySmoother(0.5, 1.25)

        linear, angular = smoother.step(0.5, 1.0, 0.2)

        self.assertAlmostEqual(linear, 0.1)
        self.assertAlmostEqual(angular, 0.25)
        linear, angular = smoother.step(-0.5, -1.0, 0.2)
        self.assertAlmostEqual(linear, 0.0)
        self.assertAlmostEqual(angular, 0.0)

    def test_reset_makes_the_next_command_ramp_from_rest(self):
        smoother = VelocitySmoother(0.5, 1.25)
        smoother.step(0.5, 1.0, 0.2)
        smoother.reset()

        linear, angular = smoother.step(0.5, 1.0, 0.2)

        self.assertAlmostEqual(linear, 0.1)
        self.assertAlmostEqual(angular, 0.25)

    def test_disabled_smoother_preserves_legacy_checkpoint_commands(self):
        smoother = VelocitySmoother(0.5, 1.25, enabled=False)

        self.assertEqual(smoother.step(0.5, -1.0, 0.2), (0.5, -1.0))

    def test_rejects_invalid_period_and_limits(self):
        with self.assertRaises(ValueError):
            VelocitySmoother(0.0, 1.25)
        smoother = VelocitySmoother(0.5, 1.25)
        with self.assertRaises(ValueError):
            smoother.step(0.5, 1.0, 0.0)


if __name__ == '__main__':
    unittest.main()
