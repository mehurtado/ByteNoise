import json, statistics, subprocess, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python")


def report(run):
    p = os.path.join(ROOT, "runs", run, "eval_val_ema.json")
    if not os.path.exists(p):
        subprocess.run([PY, "eval.py", "--run", f"runs/{run}", "--split", "val", "--which", "ema", "--max-windows", "300"],
                       cwd=ROOT, check=True, capture_output=True)
    return json.load(open(p))


def main(prefixes, seeds):
    table = {}
    for pre in prefixes:
        reps = [report(f"{pre}_s{s}") for s in seeds]
        keys = [(t, r) for t in reps[0]["tasks"] for r in reps[0]["tasks"][t]]
        table[pre] = {f"{t}/{r}": [x["tasks"][t][r]["token_acc"] for x in reps] for t, r in keys}
    names = list(table)
    print("| task/regime | " + " | ".join(f"{n} mean±sd (token acc)" for n in names) + " |")
    print("|---|" + "---|" * len(names))
    for k in table[names[0]]:
        cells = [f"{statistics.mean(table[n][k]):.3f}±{statistics.pstdev(table[n][k]):.3f}" for n in names]
        print(f"| {k} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1].split(","), [int(s) for s in sys.argv[2].split(",")])
