import unittest
import numpy as np
import data

NAME = "mix-pilot-v1"


@unittest.skipUnless(data.is_mix(NAME), "mix-pilot-v1 not built")
class MixTests(unittest.TestCase):
    def test_quota_and_bounds(self):
        st, src, info = data.mix_window_starts(NAME, "train", 10000, seed=1)
        self.assertEqual(len(st), 10000)
        self.assertEqual([info[k]["windows"] for k in info], [4000, 3500, 1500, 1000])
        n = data.load_split(NAME, "train").numel()
        self.assertTrue((st >= 0).all() and (st + 256 <= n).all())
        st2, _, _ = data.mix_window_starts(NAME, "train", 10000, seed=1)
        self.assertTrue((st == st2).all())  # deterministic

    def test_windows_inside_source_ranges(self):
        import json, os
        mix = json.load(open(os.path.join(data.DATA_ROOT, NAME, "mix.json")))
        for split in ("train", "val"):
            st, src, _ = data.mix_window_starts(NAME, split, 3000)
            for i, s in enumerate(mix["sources"]):
                lo, hi = s["ranges"][split]
                x = st[src == i]
                self.assertTrue((x >= lo).all() and (x + 256 <= hi).all())


if __name__ == "__main__":
    unittest.main()
