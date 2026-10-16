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


class SetOverrides(unittest.TestCase):
    def test_diffusion_and_train_keys(self):
        from config import get_preset
        cfg, tov = T.apply_set_items(get_preset("compact"), ["diffusion.ce_weight=0.0", "train.batch_size=7", "learning_rate=0.5"])
        self.assertEqual(cfg.diffusion.ce_weight, 0.0)
        self.assertEqual(tov, {"batch_size": 7, "learning_rate": 0.5})
        with self.assertRaises(ValueError):
            T.apply_set_items(get_preset("compact"), ["diffusion.nope=1"])


if __name__ == "__main__":
    unittest.main()
