"""DECODER-AWARE continuation of the slot-diffusion denoiser. Warm start from runs/latenthyp_slotdiff_s17/model_final.pt; ONLY the denoiser trains.
Frozen: prefix encoder, posterior/extractor, byte decoder (the whole SlotCodec) and slot-normalization stats. Loss = unchanged diffusion loss + c * byte-CE of frozen-decoder(unnormalized x0-prediction) on a subset of t<=0.5 examples.
Modes: check (CPU) | bench | train --updates N | eval. Run dir runs/latenthyp_slotdiff_dec_s17 (new; refuses non-empty)."""
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import numpy as np, torch
import torch.nn.functional as F
import latenthyp_slotdiff as S
import latenthyp_slotdiff_lendiag as LD
import latentloop_train as lt
import latentloop_eval as le
from latenthyp.slotdiff import ab, q_sample, sample

SRC = S.RUN; RUN = ROOT / 'runs' / 'latenthyp_slotdiff_dec_s17'; DEV = S.DEV; T = S.T
HPC = dict(lr=2e-4, lr_min=5e-5, warmup=200, betas=(0.9, 0.98), wd=0.01, clip=1.0)
EMA_D = 0.999; P_SC = 0.5; T_MAX = 0.5; COVER = 0.25; TARGET_RATIO = 0.17


def chash2(ms):
    h = hashlib.sha256()
    for n, p in ms.named_parameters(): h.update(p.detach().cpu().numpy().tobytes())
    for n, b in ms.named_buffers(): h.update(b.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def get_batch(st):
    x, L, y, tl = T(S.make_batch(st['corpus'], st['rng'], st['bs'])); ms = st['ms']
    with torch.no_grad():
        enc = ms.encode(x, L); x0 = (ms.slots(enc, y, tl) - st['mu']) / st['sd']
    return x0, enc['ctx'], enc['m'], x, L, y, tl


def aux_ce(ms, mu, sd, pred, x, L, y, tl):
    """per-example byte CE of frozen decoder(unnormalized predicted slots) vs recorded bytes; padding excluded; normalized by valid length. Gradient flows into `pred`."""
    with torch.no_grad(): enc = ms.encode(x, L)
    n = int(tl.max()); lg = ms.decode(pred * sd + mu, tl, enc, n=n)
    ce = F.cross_entropy(lg.float().reshape(-1, 256), y[:, :n].reshape(-1), reduction='none').view(-1, n)
    valid = (torch.arange(n, device=y.device)[None] < tl[:, None]).float()
    return (ce * valid).sum(1) / tl.float()


def forward_losses(st, batch, gen=None):
    net = st['net']; x0, ctx, cm, x, L, y, tl = batch; B = x0.size(0)
    t = torch.rand(B, device=DEV).clamp(1e-3, 1.0); a = ab(t).view(-1, 1, 1); xt = q_sample(x0, t, torch.randn_like(x0))
    with torch.no_grad(): sc = net(xt, t, tl, ctx, cm, None) * (torch.rand(B, 1, 1, device=DEV) < P_SC).float()
    pred = net(xt, t, tl, ctx, cm, sc); mse = ((pred - x0) ** 2).mean((1, 2)); snr = (a / (1 - a)).view(-1)
    L_diff = (mse * snr.clamp(0.2, 5.0)).mean()
    sel = (t <= T_MAX) & (torch.rand(B, device=DEV) < COVER); L_aux = pred.new_zeros(()); cem = float('nan')
    if sel.any():
        idx = sel.nonzero().squeeze(1); ce = aux_ce(st['ms'], st['mu'], st['sd'], pred[idx], x[idx], L[idx], y[idx], tl[idx]); L_aux = ce.sum() / B; cem = float(ce.mean())
    return dict(pred=pred, L_diff=L_diff, L_aux=L_aux, n_aux=int(sel.sum()), ce_aux=cem, mse=float(mse.mean()), t=t)


def grad_ratio(st, batch):
    """||dL_aux/dpred|| / ||dL_diff/dpred|| (coef=1) at the denoiser OUTPUT."""
    o = forward_losses(st, batch)
    if o['n_aux'] == 0: return None
    g1 = torch.autograd.grad(o['L_diff'], o['pred'], retain_graph=True)[0]; g2 = torch.autograd.grad(o['L_aux'], o['pred'])[0]
    return float(g2.norm() / g1.norm())


def calibrate(st, nb=8):
    rg = np.random.default_rng(SEED_CAL); keep = st['rng']; st['rng'] = rg; rs = []
    for _ in range(nb):
        r = grad_ratio(st, get_batch(st))
        if r is not None: rs.append(r)
    st['rng'] = keep; med = float(np.median(rs)); return TARGET_RATIO / med, rs


SEED_CAL = 18


def make_state(ms, mu, sd, total, bs):
    st = S.new_state(ms, mu, sd, total, bs); st['corpus'] = S.Corpus('train'); c = torch.load(SRC / 'model_final.pt', map_location=DEV, weights_only=False)
    st['net'].load_state_dict(c['net']); st['ema'].load_state_dict(c['ema']); st['opt'] = torch.optim.AdamW(
        [{'params': [p for n, p in st['net'].named_parameters() if p.dim() >= 2 and n != 'pos'], 'weight_decay': HPC['wd']}, {'params': [p for n, p in st['net'].named_parameters() if not (p.dim() >= 2 and n != 'pos')], 'weight_decay': 0.0}],
        lr=HPC['lr'], betas=HPC['betas']); st['coef'] = 0.0; return st


def train_step(st):
    net, opt = st['net'], st['opt']; step = st['step']
    for g in opt.param_groups: g['lr'] = lt.lr_at(step, st['total'], HPC)
    o = forward_losses(st, get_batch(st)); loss = o['L_diff'] + st['coef'] * o['L_aux']
    opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(st['params'], HPC['clip']); opt.step()
    with torch.no_grad():
        for pe, p in zip(st['ema'].parameters(), net.parameters()): pe.mul_(EMA_D).add_(p.detach(), alpha=1 - EMA_D)
    st['step'] += 1
    return dict(loss=float(loss), L_diff=float(o['L_diff']), L_aux=float(o['L_aux']), ce_aux=o['ce_aux'], n_aux=o['n_aux'], mse=o['mse'], gn=float(gn))


def codec_frozen_checks(ms):
    assert not ms.training and not any(p.requires_grad for p in ms.parameters())
    assert all((not isinstance(m, torch.nn.Dropout)) or m.p == 0 or not m.training for m in ms.modules())


def run_check(a):
    ms = S.load_codec(); codec_frozen_checks(ms); rg = np.random.default_rng(9001); trc = S.Corpus('train'); tr = [trc.sample(rg, S.TAG_CONT) for _ in range(128)]
    St = S.codes(ms, tr); mu, sd = St.mean(0), St.std(0).clamp(min=1e-6); res = {}
    t = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0]); aa = ab(t); print('SCHEDULE: t | alpha_bar | SNR'); [print(f'  t={float(ti):.2f}  ab={float(ai):.4f}  SNR={float(ai/(1-ai+1e-12)):.4f}') for ti, ai in zip(t, aa)]
    res['schedule'] = 't=0 is CLEAN (alpha_bar~1), t=1 is pure noise (alpha_bar~0); t=0.5: alpha_bar=%.3f, SNR=%.3f -> t<=0.5 is the LOW-to-medium-noise half' % (float(ab(torch.tensor(0.5))), float(ab(torch.tensor(0.5)) / (1 - ab(torch.tensor(0.5)))))
    print(res['schedule']); assert float(ab(torch.tensor(0.0))) > 0.99 and float(ab(torch.tensor(1.0))) < 1e-6
    st = make_state(ms, mu, sd, 100, 16); h0 = chash2(ms)
    # finite aux gradients reach the denoiser
    batch = get_batch(st); torch.manual_seed(0); o = forward_losses(st, batch)
    while o['n_aux'] == 0: o = forward_losses(st, batch)
    st['net'].zero_grad(); o['L_aux'].backward(); gs = [p.grad for p in st['net'].parameters()]; nrm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs if g is not None)))
    res['aux_grad_into_denoiser'] = {'all_finite': all(g is not None and bool(torch.isfinite(g).all()) for g in gs), 'norm': nrm}; assert res['aux_grad_into_denoiser']['all_finite'] and nrm > 0
    # padding contributes zero loss
    x0, ctx, cm, x, L, y, tl = batch; pr = torch.randn(x0.shape, device=DEV); c1 = aux_ce(ms, mu, sd, pr, x, L, y, tl); y2 = y.clone(); pad = torch.arange(y.size(1), device=DEV)[None] >= tl[:, None]; y2[pad] = (y2[pad] + 7) % 256
    c2 = aux_ce(ms, mu, sd, pr, x, L, y2, tl); res['padding_zero_loss(max |diff| when padded targets changed)'] = float((c1 - c2).abs().max()); assert res['padding_zero_loss(max |diff| when padded targets changed)'] == 0.0
    # per-example normalization by valid length
    res['aux_ce_example_values'] = [round(float(v), 3) for v in c1[:4]]
    st['coef'], rs = calibrate(st, 4); res['calibration(cpu small)'] = {'coef': st['coef'], 'ratios': rs}
    for _ in range(3): r = train_step(st); assert np.isfinite(r['loss'])
    assert chash2(ms) == h0; codec_frozen_checks(ms); res['codec_params_and_buffers_unchanged_after_3_updates'] = True
    # checkpoint round trip (bit exact on CPU)
    def mk(): s_ = make_state(ms, mu, sd, 100, 8); s_['coef'] = st['coef']; return s_
    A = mk(); train_step(A); train_step(A); B = mk(); train_step(B); S.save_state(B, ROOT / 'runs' / 'latenthyp_slotdiff_dec_check_tmp.pt'); C = mk(); S.load_state(C, ROOT / 'runs' / 'latenthyp_slotdiff_dec_check_tmp.pt'); train_step(C)
    res['checkpoint_roundtrip_bitexact'] = bool(all(torch.equal(p, q) for p, q in zip(A['net'].parameters(), C['net'].parameters())) and all(torch.equal(p, q) for p, q in zip(A['ema'].parameters(), C['ema'].parameters()))); assert res['checkpoint_roundtrip_bitexact']
    print(json.dumps(res, indent=1, default=float)); (ROOT / 'runs' / 'latenthyp_slotdiff_dec_check.json').write_text(json.dumps(res, indent=1, default=float))


def setup(bs, total):
    ms = S.load_codec(); codec_frozen_checks(ms); nd = torch.load(SRC / 'norm.pt', map_location=DEV); return ms, nd['mu'], nd['sd'], make_state(ms, nd['mu'], nd['sd'], total, bs)


def run_bench(a):
    ms, mu, sd, st = setup(64, 100000); torch.cuda.reset_peak_memory_stats(); st['coef'], rs = calibrate(st); print('calibration coef', st['coef'], 'ratios', [round(r, 3) for r in rs])
    for k in range(15 + a.steps):
        if k == 15: torch.cuda.synchronize(); t0 = time.time()
        train_step(st)
    torch.cuda.synchronize(); print('s/update', round((time.time() - t0) / a.steps, 4), 'peakMB', round(torch.cuda.max_memory_allocated() / 2 ** 20))


def run_train(a):
    if RUN.exists() and any(RUN.iterdir()): raise SystemExit(f'Refusing non-empty {RUN}')
    RUN.mkdir(parents=True, exist_ok=True); ms, mu, sd, st = setup(64, a.updates); h = chash2(ms); torch.save({'mu': mu, 'sd': sd}, RUN / 'norm.pt'); nh = hashlib_t(mu, sd)
    st['coef'], rs = calibrate(st); torch.manual_seed(S.SEED + 1); torch.cuda.manual_seed_all(S.SEED + 1)
    proto = {'experiment': 'decoder-aware continuation of slot-diffusion denoiser (warm start from runs/latenthyp_slotdiff_s17/model_final.pt; net weights and EMA both loaded; fresh AdamW)', 'seed': S.SEED, 'batch': 64, 'total_updates': a.updates, 'hp': HPC, 'ema_decay': EMA_D,
             'frozen': 'whole SlotCodec (prefix encoder, posterior/extractor, byte decoder, buffers) + normalization stats; hash asserted at end', 'codec_hash_params_and_buffers': h, 'norm_hash': nh,
             'aux': {'eligible': 't<=0.5 (verified: t=0 clean, t=1 pure noise; alpha_bar(0.5)=%.3f SNR=%.3f)' % (float(ab(torch.tensor(0.5))), float(ab(torch.tensor(0.5)) / (1 - ab(torch.tensor(0.5))))), 'coverage': '25% of eligible, random per example', 'loss': 'byte CE of frozen decoder(unnormalized predicted x0) vs recorded bytes, padding excluded, per-example valid-length normalized, summed over selected /B (batch-mean with zeros for unselected)',
                     'gradient path': 'logits CE only; no detach/no_grad/argmax between denoiser output and decoder logits',
                     'coefficient': st['coef'], 'calibration': f'8 batches at warm-start weights; target ratio ||g_aux||/||g_diff|| at denoiser output = {TARGET_RATIO} (user range .10-.25); measured per-batch ratios at coef=1: {[round(r,4) for r in rs]}; fixed, no sweep'},
             'diffusion_loss': 'unchanged (x0-pred, truncated-SNR weight clip(SNR,.2,5), self-cond p=.5)', 'lr': 'cosine 2e-4 -> 5e-5, warmup 200 (continuation; source ended at 1e-4; this choice is mine)'}
    (RUN / 'protocol.json').write_text(json.dumps(proto, indent=2, default=float)); print('coef', st['coef'], 'cal ratios', [round(r, 4) for r in rs], flush=True)
    log = open(RUN / 'metrics.jsonl', 'a'); win = []; t0 = time.time(); torch.cuda.reset_peak_memory_stats(); val = S.eval_cont(S.Corpus('val'), 64, seed=555, n_each=64); vb = T(S.to_arrays(val)); vx, vL, vy, vtl = vb
    while st['step'] < st['total']:
        r = train_step(st)
        if not np.isfinite(r['loss']): raise SystemExit(f'non-finite at {st["step"]}')
        win.append(r)
        if st['step'] % 100 == 0:
            g = np.array([w['gn'] for w in win]); ces = [w['ce_aux'] for w in win if np.isfinite(w['ce_aux'])]
            rec = {'step': st['step'], 'L_diff': round(float(np.mean([w['L_diff'] for w in win])), 4), 'L_aux(CE mean over selected, per-example)': round(float(np.mean(ces)), 4), 'aux_examples_per_batch': round(float(np.mean([w['n_aux'] for w in win])), 2), 'x0_mse': round(float(np.mean([w['mse'] for w in win])), 4),
                   'gnorm_median': round(float(np.median(g)), 3), 'clip_frac': round(float((g > 1).mean()), 3), 'lr': lt.lr_at(st['step'], st['total'], HPC), 'elapsed_s': round(time.time() - t0, 1)}
            if st['step'] % 1000 == 0:
                rr = [grad_ratio(st, get_batch(st)) for _ in range(3)]; rr = [r_ * st['coef'] for r_ in rr if r_ is not None]; rec['aux/diff grad ratio at output (with coef; monitored)'] = round(float(np.mean(rr)), 3)
                with torch.no_grad():
                    gg = torch.Generator(device=DEV); gg.manual_seed(3); enc = ms.encode(vx, vL); x0 = (ms.slots(enc, vy, vtl) - mu) / sd; vv = {}
                    for tv in (0.2, 0.5, 0.8):
                        tt = torch.full((x0.size(0),), tv, device=DEV); xt = q_sample(x0, tt, torch.randn(x0.shape, device=DEV, generator=gg)); pr = st['ema'](xt, tt, vtl, enc['ctx'], enc['m']); lg = ms.decode(pr * sd + mu, vtl, enc, n=vy.size(1))
                        ce = F.cross_entropy(lg.reshape(-1, 256), vy.reshape(-1), reduction='none').view_as(vy); v_ = torch.arange(vy.size(1), device=DEV)[None] < vtl[:, None]
                        vv[str(tv)] = {'mse': round(float(((pr - x0) ** 2).mean()), 4), 'decoded_CE': round(float((ce * v_).sum() / v_.sum()), 3)}
                rec['heldout64_ema'] = vv
            win = []; log.write(json.dumps(rec) + '\n'); log.flush()
            if st['step'] % 500 == 0: print(json.dumps(rec), flush=True)
    assert chash2(ms) == h and hashlib_t(mu, sd) == nh; src = torch.load(S.CODEC, map_location=DEV, weights_only=False)['model']; assert all(torch.equal(src[k], v) for k, v in ms.state_dict().items())
    S.save_state(st, RUN / 'checkpoint.pt'); torch.save({'ema': st['ema'].state_dict(), 'net': st['net'].state_dict(), 'step': st['step']}, RUN / 'model_final.pt')
    (RUN / 'status.json').write_text(json.dumps({'updates': st['step'], 'seconds': time.time() - t0, 'codec_and_norm_frozen_unchanged': True, 'peak_mem_MB': torch.cuda.max_memory_allocated() / 2 ** 20})); print('done; frozen verified', flush=True)


def hashlib_t(*ts):
    h = hashlib.sha256()
    for t in ts: h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ eval
def boot(Hn, Tn, Rn, Hs, Ts, Rs, n=2000, seed=5):
    rg = np.random.default_rng(seed); idx = rg.integers(0, len(Hn), (n, len(Hn)))
    dw = Hn[idx].sum(1) / Tn[idx].sum(1) - Hs[idx].sum(1) / Ts[idx].sum(1); dr = Rn[idx].mean(1) - Rs[idx].mean(1)
    f = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {'word_hit diff (new-source) point': round(float(Hn.sum() / Tn.sum() - Hs.sum() / Ts.sum()), 4), 'word_hit 95% CI': f(dw), 'valid-word-run diff point': round(float(Rn.mean() - Rs.mean()), 3), 'valid-word-run 95% CI': [round(float(np.percentile(dr, 2.5)), 3), round(float(np.percentile(dr, 97.5)), 3)]}


def run_eval(a):
    words = le.wordlist(); LD.words = words; ms = S.load_codec(); h = chash2(ms); nd = torch.load(SRC / 'norm.pt', map_location=DEV); mu, sd = nd['mu'], nd['sd']
    nets = {'SOURCE': S.load_ema(SRC / 'model_final.pt')[0], 'NEW(dec-aware)': S.load_ema(RUN / 'model_final.pt')[0]}
    val = S.Corpus('val'); ho = S.eval_cont(val, 256); T4 = T(S.to_arrays(ho)); x, L, y, tl = T4; tl_np = tl.cpu().numpy(); tg = [e[1] for e in ho]; pref = [e[0][1:] for e in ho]; other = [pref[(i + 37) % 256] for i in range(256)]
    with torch.no_grad(): rec_codes = torch.cat([ms.slots(ms.encode(x[s:s + 128], L[s:s + 128]), y[s:s + 128], tl[s:s + 128]) for s in range(0, 256, 128)])
    rec = S.decode_raw(ms, T4, rec_codes); G = {}; Z = {}
    for nm, net in nets.items():
        for seed in (101, 202): Z[(nm, seed)], G[(nm, seed)] = S.generate(ms, net, T4, mu, sd, seed, 50)
    R = {}; BK = {'ALL': range(256)}
    print('256 HELD-OUT AGGREGATES (per seed; true length)')
    for (nm, seed), o in G.items():
        R[f'{nm} seed{seed}'] = {**S.gm(o, ho, range(256), words), **LD.met(o, range(256))}; print(' ', nm, seed, json.dumps(R[f'{nm} seed{seed}']))
    R['RECON'] = {**S.gm(rec, ho, range(256), words), **LD.met(rec, range(256))}; print('  RECON', json.dumps(R['RECON']))
    # paired bootstrap
    def arrs(o): w = [LD.wh1(s) for s in o]; return np.array([a_[0] for a_ in w], float), np.array([a_[1] for a_ in w], float), np.array([LD.run1(s) for s in o], float)
    P = {}
    for nm in nets: A = [arrs(G[(nm, sd_)]) for sd_ in (101, 202)]; P[nm] = tuple(A[0][k] + A[1][k] for k in range(3)); P[nm] = (P[nm][0], P[nm][1], (A[0][2] + A[1][2]) / 2)
    R['paired_bootstrap_new_minus_source(256 prefixes x 2 seeds, resample prefixes)'] = boot(*P['NEW(dec-aware)'], *P['SOURCE']); print('BOOTSTRAP', json.dumps(R['paired_bootstrap_new_minus_source(256 prefixes x 2 seeds, resample prefixes)']))
    sub = np.arange(16); R['paired_bootstrap_16_prefixes'] = boot(*(v[sub] for v in P['NEW(dec-aware)']), *(v[sub] for v in P['SOURCE'])); print('BOOTSTRAP16', json.dumps(R['paired_bootstrap_16_prefixes']))
    # overlap, diversity, geometry
    mo = lambda po, pf: round(float(np.mean([S.ovl(po[i], pf[i]) for i in range(256)])), 3)
    ag = lambda A, B, i, j: float(np.mean([A[i][k] == B[j][k] for k in range(min(len(A[i]), len(B[j])))]))
    trs, Str, _, _ = S.train_stats(ms); Ntr = (Str - mu) / sd; Ztr = Ntr.flatten(1)
    def slotnn(Zq): return torch.stack([(torch.cdist(Zq[:, k], Ntr[:, k]) / 4).min(1).values for k in range(16)], 1)
    for nm in nets:
        o1, o2 = G[(nm, 101)], G[(nm, 202)]; z1, z2 = Z[(nm, 101)], Z[(nm, 202)]
        R[f'{nm}: prefix word overlap'] = {'right prefix': mo(o1, pref), 'unrelated prefix': mo(o1, other)}
        R[f'{nm}: diversity'] = {'byte agreement seed1 vs seed2 same prefix': round(float(np.mean([ag(o1, o2, i, i) for i in range(256)])), 3), 'seed1 vs seed1 other prefix': round(float(np.mean([ag(o1, o1, i, (i + 37) % 256) for i in range(256)])), 3), 'unique outputs/256': len(set(o1)),
                                'slot dist seed1 vs seed2 same prefix': round(float((z1 - z2).pow(2).mean((1, 2)).sqrt().mean()), 3)}
        sg = slotnn(z1); R[f'{nm}: geometry'] = {'per-slot nearest same-index train slot (mean, p5)': [round(float(sg.mean()), 3), round(float(sg.flatten().quantile(.05)), 3)], 'whole-vector nearest train code (median)': round(float((torch.cdist(z1.flatten(1), Ztr) / 16).min(1).values.median()), 3),
                                                  'per-dim std mean': round(float(z1.std(0).mean()), 3), 'per-dim |mean| mean': round(float(z1.mean(0).abs().mean()), 3)}
        print(nm, json.dumps({k: v for k, v in R.items() if k.startswith(nm + ':')}, default=float))
    sgt = slotnn((rec_codes - mu) / sd); R['TRUE held-out codes: per-slot nearest train (mean,p5)'] = [round(float(sgt.mean()), 3), round(float(sgt.flatten().quantile(.05)), 3)]; print('TRUE per-slot', R['TRUE held-out codes: per-slot nearest train (mean,p5)'])
    # length offsets
    offs = {}
    for nm_, raw in [('REAL posterior-mean', rec_codes)] + [(f'{n} seed{s_}', Z[(n, s_)] * sd + mu) for n in nets for s_ in (101, 202)]:
        offs[nm_] = {f'{d:+.1f}': {k: v for k, v in LD.met(LD.dec_at(ms, T4, raw, LD.lens_for(tl, 1 + d)), range(256)).items() if k in ('word_hit', 'longest_valid_word_run')} for d in (-0.1, 0.0, 0.1)}
    R['length_offset_curve'] = offs; print('LENGTH OFFSETS'); [print(' ', k, v) for k, v in offs.items()]
    # denoising MSE and decoded-CE at noise levels (context only)
    ctxs = {}
    with torch.no_grad():
        enc_all = [ms.encode(x[s:s + 128], L[s:s + 128]) for s in range(0, 256, 128)]
        x0 = (rec_codes - mu) / sd; gg = torch.Generator(device=DEV); gg.manual_seed(11)
        for tv in (0.2, 0.5, 0.8):
            noise = torch.randn(x0.shape, device=DEV, generator=gg); tt = torch.full((256,), tv, device=DEV); xt = q_sample(x0, tt, noise); row = {}
            for nm, net in nets.items():
                ces = []; se = 0.0
                for k, s in enumerate(range(0, 256, 128)):
                    q = slice(s, s + 128); pr = net(xt[q], tt[q], tl[q], enc_all[k]['ctx'], enc_all[k]['m']); se += float(((pr - x0[q]) ** 2).sum()); lg = ms.decode(pr * sd + mu, tl[q], enc_all[k], n=y.size(1))
                    ce = F.cross_entropy(lg.float().reshape(-1, 256), y[q].reshape(-1), reduction='none').view_as(y[q]); v_ = torch.arange(y.size(1), device=DEV)[None] < tl[q][:, None]; ces.append(((ce * v_).sum(), v_.sum()))
                row[nm] = {'x0_mse': round(se / x0.numel(), 4), 'decoded_CE_per_byte': round(float(sum(c[0] for c in ces) / sum(c[1] for c in ces)), 3)}
            ctxs[str(tv)] = row
    R['denoise_context(not pass criteria)'] = ctxs; print('DENOISE CONTEXT', json.dumps(ctxs))
    L_ = []
    for i in range(16):
        L_.append(f'--- heldout256[{i}] ({tl_np[i]}B)\n  TARGET : {tg[i]!r}\n  RECON  : {rec[i]!r}' + ''.join(f'\n  seed{sd_} SOURCE: {G[("SOURCE", sd_)][i]!r}\n  seed{sd_} NEW   : {G[("NEW(dec-aware)", sd_)][i]!r}' for sd_ in (101, 202)))
    (RUN / 'samples.txt').write_text('\n'.join(L_), encoding='utf-8'); (RUN / 'eval.json').write_text(json.dumps(R, indent=1, default=float)); print('\n'.join(L_)); assert chash2(ms) == h


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['check', 'bench', 'train', 'eval']); p.add_argument('--updates', type=int, default=0); p.add_argument('--steps', type=int, default=60)
    a = p.parse_args(); {'check': run_check, 'bench': run_bench, 'train': run_train, 'eval': run_eval}[a.mode](a)
