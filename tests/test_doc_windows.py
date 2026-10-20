import json, os, tempfile, unittest
import numpy as np, torch
import data, train


class DocWindows(unittest.TestCase):
    def test_windows_inside_docs(self):
        with tempfile.TemporaryDirectory() as d:
            old = data.DATA_ROOT; data.DATA_ROOT = d
            try:
                os.makedirs(os.path.join(d, "x"))
                docs = [b"a" * 10 + b"\n\n", b"b" * 3 + b"\n\n", b"c" * 9 + b"\n\n", b"d" * 4 + b"\n\n"]
                off = 0
                with open(os.path.join(d, "x", "documents.jsonl"), "w") as f:
                    for i, t in enumerate(docs):
                        f.write(json.dumps(dict(split="val" if i == 3 else "train", offset=off, length=len(t))) + "\n"); off += len(t)
                st, stats = data.doc_window_starts("x", "train", 4)
                self.assertEqual(st.tolist(), [0, 4, 17, 21])  # doc1 too short: none; tails dropped
            except AssertionError:
                raise
            finally:
                data.DATA_ROOT = old

    def test_windows_get(self):
        toks = torch.arange(100, dtype=torch.uint8)
        w = train.Windows(toks, 4, np.array([0, 50]))
        self.assertEqual(w.get(torch.tensor([1, 0])).tolist(), [[50, 51, 52, 53], [0, 1, 2, 3]])
        self.assertEqual(train.Windows(toks, 4).get(torch.tensor([1])).tolist(), [[4, 5, 6, 7]])


if __name__ == "__main__":
    unittest.main()
