"""Regression coverage for loading NumPy 2 checkpoints on NumPy 1.x."""

import importlib
import unittest

import numpy as np

from social_rl.agent_node import _install_numpy_checkpoint_compat


class NumpyCheckpointCompatTest(unittest.TestCase):

    def test_numpy_2_private_modules_are_importable(self):
        installed = _install_numpy_checkpoint_compat()

        numpy_private_core = importlib.import_module('numpy._core')
        numpy_private_multiarray = importlib.import_module(
            'numpy._core.multiarray')

        if int(np.__version__.split('.', maxsplit=1)[0]) < 2:
            self.assertTrue(installed)
            self.assertIs(numpy_private_core,
                          importlib.import_module('numpy.core'))
            self.assertIs(numpy_private_multiarray,
                          importlib.import_module('numpy.core.multiarray'))
        else:
            self.assertFalse(installed)


if __name__ == '__main__':
    unittest.main()
