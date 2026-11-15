import unittest

from utils.device import gpu_is_visible, use_cpu_only


class UseCpuOnlyTests(unittest.TestCase):
    def test_no_gpu_is_visible_after_use_cpu_only(self) -> None:
        use_cpu_only()
        self.assertFalse(gpu_is_visible())

    def test_use_cpu_only_can_be_called_twice(self) -> None:
        use_cpu_only()
        use_cpu_only()
        self.assertFalse(gpu_is_visible())


if __name__ == "__main__":
    unittest.main()
