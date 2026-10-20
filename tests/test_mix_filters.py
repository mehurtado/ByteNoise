import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import mix_filters as mf


def win(s):
    return np.frombuffer((s.encode() + b"a" * 256)[:256], np.uint8)


class FilterTests(unittest.TestCase):
    def test_flags_junk(self):
        table = "| Pos | Team | Pld |\n| --- | ---- | --- |\n| 1 | Dumbarton | 19 |\n| 2 | Cowlairs | 18 |"
        pad = "Incumbent re-elected." + " " * 40 + "| William Richardson 51.8%"
        r = mf.reasons(np.stack([win(table), win(pad), win("-" * 20 + " Intermediate")]))
        self.assertTrue(r["pipes"][0] and r["ws_run"][1] and r["sym_run"][2])

    def test_keeps_prose_and_dialogue(self):
        prose = ['"Mom, can we play?" she asked.\n\n"Yes," said Mom. "But be home by six; it gets dark early."\n\n',
                 "The 1892 season (Aston Villa, 2–3) began on 10 October. See Smith & Sons, p. 45: 'a fine year'.\n# Heading\n\n",
                 "Once upon a time, there was a little girl named Lily... She loved her dog -- a small, brown pup!"]
        W = np.stack([win(s) for s in prose])
        self.assertFalse(mf.is_junk(W).any())

    def test_sample_metrics_exclude_padding(self):
        s = b"The cat sat on the mat. " + b" " * 64 + b"| | | | | " + b" " * 64
        m = mf.junk_sample_metrics([s, s], b"The cat sat on the mat. xxxxxxxx")
        self.assertGreater(m["space_run8_frac"], 0.5)
        self.assertEqual(m["table_junk_samples"], 0)


if __name__ == "__main__":
    unittest.main()
