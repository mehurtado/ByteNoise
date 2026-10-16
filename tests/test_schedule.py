import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train as T


class Schedule(unittest.TestCase):
    def test_warmup_then_cosine_by_updates(self):
        f = T.lr_lambda(4, 104, 1.0, 0.1)
        self.assertAlmostEqual(f(0), 0.25)
        self.assertAlmostEqual(f(3), 1.0)
        self.assertAlmostEqual(f(4), 1.0)
        self.assertAlmostEqual(f(104), 0.1)
        self.assertAlmostEqual(f(54), 0.55)

    def test_time_budget_drives_decay(self):
        f = T.lr_lambda(0, 10**6, 1.0, 0.0, lambda: 50.0, 100.0)
        self.assertAlmostEqual(f(10), 0.5)
        f2 = T.lr_lambda(0, 10**6, 1.0, 0.0, lambda: 100.0, 100.0)
        self.assertAlmostEqual(f2(10), 0.0)


if __name__ == "__main__":
    unittest.main()
