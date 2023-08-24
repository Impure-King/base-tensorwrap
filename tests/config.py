"""Tests the module ``tensorwrap.config```."""

import jax
import unittest
from ..tensorwrap import config
class test_is_device_available(unittest.TestCase):
    def test_is_device_available(self):
        self.assertEqual(config.is_device_available("cuda"), jax.devices("gpu"), f"Should be {jax.devices('gpu')}")
        self.assertEqual(config.is_device_available("cpu"), jax.devices("cpu"), f"Should be {jax.devices('cpu')}")
        self.assertEqual(config.is_device_available("tpu"), False, f"Should be {jax.devices('tpu')}")
        print("Test 1 passed.")


if __name__ == "__main__":
    unittest.main()