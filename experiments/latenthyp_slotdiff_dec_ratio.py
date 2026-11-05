"""Aux-gradient contribution at fixed batches: START (source net weights) vs END (new net weights). Eval only."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import numpy as np, torch
import latenthyp_slotdiff_dec as D
S = D.S
ms, mu, sd, st = D.setup(64, 100)
coef = json.loads((D.RUN / 'protocol.json').read_text())['aux']['coefficient']
cs = {'START (source net weights, step 0)': torch.load(D.SRC / 'model_final.pt', map_location=D.DEV, weights_only=False)['net'], 'END (new net weights, step 9500)': torch.load(D.RUN / 'model_final.pt', map_location=D.DEV, weights_only=False)['net']}
rg = np.random.default_rng(31); batches = []; keep = st['rng']; st['rng'] = rg
for _ in range(6): batches.append(D.get_batch(st))
res = {}
for nm, sdict in cs.items():
    st['net'].load_state_dict(sdict); rows = []
    for k, b in enumerate(batches):
        torch.manual_seed(1000 + k); o = D.forward_losses(st, b)
        if o['n_aux'] == 0: continue
        g1 = torch.autograd.grad(o['L_diff'], o['pred'], retain_graph=True)[0]; g2 = torch.autograd.grad(o['L_aux'], o['pred'], retain_graph=True)[0]
        ps = list(st['net'].parameters()); p1 = torch.autograd.grad(o['L_diff'], ps, retain_graph=True); p2 = torch.autograd.grad(o['L_aux'], ps)
        pn = lambda g: float(torch.sqrt(sum((v.double() ** 2).sum() for v in g)))
        rows.append({'batch': k, 'n_aux': o['n_aux'], 'output_ratio(coef*aux / diff)': round(coef * float(g2.norm() / g1.norm()), 3), 'param_ratio(coef*aux / diff)': round(coef * pn(p2) / pn(p1), 3), 'aux_CE_per_example_mean': round(o['ce_aux'], 4), 'L_diff': round(float(o['L_diff']), 4)})
    res[nm] = {'rows': rows, 'mean_output_ratio': round(float(np.mean([r['output_ratio(coef*aux / diff)'] for r in rows])), 3), 'mean_param_ratio': round(float(np.mean([r['param_ratio(coef*aux / diff)'] for r in rows])), 3), 'mean_aux_CE': round(float(np.mean([r['aux_CE_per_example_mean'] for r in rows])), 4)}
    print(nm, json.dumps(res[nm]))
(D.RUN / 'aux_ratio_fixed_batches.json').write_text(json.dumps({'coef': coef, **res}, indent=1))
