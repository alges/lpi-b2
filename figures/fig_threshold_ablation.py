"""
postproc_03_threshold_ablation.py
Demonstrates that the A/B/C taxonomy is robust to threshold choices.
Sweeps tau_delta ∈ {0.5,0.6,0.7,0.8,0.9} and tau_H ∈ {0.6,0.7,0.8,0.9}.

NOTE: entropy in sample_posteriors.csv is in BITS [0,1] (Shannon entropy,
      base-2 log). Max = 1.0 at p=0.5. tau_H values are interpreted in bits.
      discordance = |y_supplied − p_truth| ∈ [0,1].
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / 'data'
FIGURES = ROOT / 'outputs'

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 12,
                     'legend.fontsize': 10, 'figure.dpi': 110})
TAG = '[postproc_03]'
ARCH_COLORS = {'LR': '#9467bd', 'RF': '#2ca02c', 'SVM': '#1f77b4', 'GB': '#ff7f0e', 'MLP': '#d62728'}

NEEDED = [TABLES / 'sample_posteriors_yc.csv']
for p in NEEDED:
    if not p.exists():
        print(f'{TAG} ERROR: missing {p}'); sys.exit(1)

print(f'{TAG} Loading sample_posteriors_yc.csv …')
post = pd.read_csv(TABLES / 'sample_posteriors_yc.csv')
print(f'{TAG}   shape={post.shape}  entropy range=[{post.entropy.min():.4f}, {post.entropy.max():.4f}]')
print(f'{TAG}   NOTE: entropy treated as bits [0,1]; tau_H thresholds in bits.')

TAU_DELTA = [0.5, 0.6, 0.7, 0.8, 0.9]
TAU_H     = [0.6, 0.7, 0.8, 0.9]

# ── Per-seed, per-threshold taxonomy counts ────────────────────────────────
print(f'{TAG} Computing taxonomy across {len(TAU_DELTA)}×{len(TAU_H)} threshold grid …')
records = []
for (arch, seed, flip_y), grp in post.groupby(['arch', 'seed', 'flip_y']):
    disc = grp['discordance'].values
    entr = grp['entropy'].values
    n    = len(disc)
    for td in TAU_DELTA:
        for th in TAU_H:
            mask_A = disc > td
            mask_B = (~mask_A) & (entr > th)
            mask_C = (~mask_A) & (~mask_B)
            records.append({'arch': arch, 'seed': seed, 'flip_y': flip_y,
                            'tau_delta': td, 'tau_H': th,
                            'f_A': mask_A.mean(), 'f_B': mask_B.mean(), 'f_C': mask_C.mean()})

df = pd.DataFrame(records)
print(f'{TAG} Raw records: {len(df)}')

# ── Aggregate over seeds ───────────────────────────────────────────────────
agg = df.groupby(['arch', 'flip_y', 'tau_delta', 'tau_H']).agg(
    n_seeds=('seed', 'count'),
    f_A_mean=('f_A', 'mean'), f_A_std=('f_A', 'std'),
    f_B_mean=('f_B', 'mean'), f_B_std=('f_B', 'std'),
    f_C_mean=('f_C', 'mean'), f_C_std=('f_C', 'std'),
).reset_index()

# Add class_sep column (single file = sep=1.0)
agg['class_sep'] = 1.0

out_csv = TABLES / 'postproc_03_threshold_ablation.csv'
agg.to_csv(out_csv, index=False)
print(f'{TAG} Saved CSV: {out_csv} (shape={agg.shape})')

# ── Figure 1: heatmap 5×4 for a representative condition ──────────────────
# LR, flip_y=0.3: shows real variation (~0.21 to ~0.38) vs RF which is ~0.001 uniform
rep_arch = 'LR'; rep_phi = 0.3
subset = agg[(agg['arch'] == rep_arch) & (agg['flip_y'] == rep_phi)]
print(f'{TAG} heatmap subset: arch={rep_arch}, phi={rep_phi}, shape={subset.shape}')
print(f'{TAG}   f_A range: {subset.f_A_mean.min():.4f} → {subset.f_A_mean.max():.4f}')

heat_data = subset.pivot(index='tau_delta', columns='tau_H', values='f_A_mean')
heat_data = heat_data.reindex(index=TAU_DELTA, columns=TAU_H)

vmin = heat_data.values.min()
vmax = heat_data.values.max()

fig1, ax1 = plt.subplots(figsize=(7, 5.5))
im = ax1.imshow(heat_data.values, aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax)
ax1.set_xticks(range(len(TAU_H)));     ax1.set_xticklabels([f'{t}' for t in TAU_H])
ax1.set_yticks(range(len(TAU_DELTA))); ax1.set_yticklabels([f'{t}' for t in TAU_DELTA])
ax1.set_xlabel('τ_H (entropy threshold, bits)')
ax1.set_ylabel('τ_δ (discordance threshold)')
ax1.set_title(f'f_A mean — {rep_arch}, φ={rep_phi}, sep=1.0')
DEFAULT_TD = 0.8; DEFAULT_TH = 0.8
for i, td in enumerate(TAU_DELTA):
    for j, th in enumerate(TAU_H):
        val = heat_data.values[i, j]
        is_default = (td == DEFAULT_TD and th == DEFAULT_TH)
        label = f'★\n{val:.2f}' if is_default else f'{val:.2f}'
        mid = (vmin + vmax) / 2
        ax1.text(j, i, label, ha='center', va='center',
                 fontsize=9, color='white' if val > mid else 'black',
                 fontweight='bold' if is_default else 'normal')
        if is_default:
            rect = plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                  fill=False, edgecolor='black', lw=2.5, zorder=4)
            ax1.add_patch(rect)
plt.colorbar(im, ax=ax1, label='f_A (fraction Type A)')
plt.tight_layout()
out_fig1 = FIGURES / 'postproc_03_threshold_heatmap.pdf'
fig1.savefig(out_fig1, bbox_inches='tight')
fig1.savefig(str(out_fig1).replace('.pdf', '.png'), bbox_inches='tight', dpi=110)
plt.close()
print(f'{TAG} heatmap regenerated for LR @ φ=0.3, sep=1.0')
print(f'{TAG} f_A range: {vmin:.2f} → {vmax:.2f} (vs RF: ~0.001 uniform)')
print(f'{TAG} regenerated: {out_fig1}')

# ── Figure 2: robustness curves — f_A vs tau_delta, tau_H=0.8 fixed ───────
TH_FIXED = 0.8
PHIS = [0.1, 0.3, 0.5]
ARCHS = ['LR', 'RF', 'SVM', 'GB', 'MLP']

fig2, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
for ax, phi in zip(axes, PHIS):
    sub = agg[(agg['tau_H'] == TH_FIXED) & (agg['flip_y'] == phi)]
    for arch in ARCHS:
        s = sub[sub['arch'] == arch].sort_values('tau_delta')
        if s.empty:
            continue
        ax.plot(s['tau_delta'], s['f_A_mean'], color=ARCH_COLORS[arch],
                lw=1.8, marker='o', ms=6, label=arch)
        ax.fill_between(s['tau_delta'],
                        s['f_A_mean'] - s['f_A_std'],
                        s['f_A_mean'] + s['f_A_std'],
                        color=ARCH_COLORS[arch], alpha=0.12)
    ax.axvspan(0.7, 0.9, alpha=0.06, color='grey', label='τ_δ ∈ [0.7,0.9]')
    ax.axvline(0.8, color='k', ls='--', lw=1.0, alpha=0.5)
    ax.set_xlabel('τ_δ (discordance threshold)')
    ax.set_ylabel('f_A (fraction Type A)' if phi == PHIS[0] else '')
    ax.set_title(f'φ = {phi}')
    ax.legend(fontsize=8)

fig2.suptitle(f'f_A robustness to τ_δ (τ_H={TH_FIXED} fixed, sep=1.0)', fontsize=12)
plt.tight_layout()
out_fig2 = FIGURES / 'postproc_03_threshold_robustness.pdf'
fig2.savefig(out_fig2, bbox_inches='tight')
plt.close()
print(f'{TAG} Saved figure: {out_fig2}')

# ── Robustness report ──────────────────────────────────────────────────────
print(f'\n{TAG} ── ROBUSTNESS REPORT: variation of f_A when τ_δ ∈ [0.7, 0.9] ──')
print(f'{"arch":>5}  {"flip_y":>7}  {"f_A@0.7":>9}  {"f_A@0.9":>9}  {"rel.range%":>10}  verdict')
for phi in PHIS:
    sub = agg[(agg['tau_H'] == TH_FIXED) & (agg['flip_y'] == phi)]
    for arch in ARCHS:
        s = sub[(sub['arch'] == arch) & (sub['tau_delta'].isin([0.7, 0.9]))]
        if len(s) < 2:
            continue
        f07 = s[s['tau_delta'] == 0.7]['f_A_mean'].values[0]
        f09 = s[s['tau_delta'] == 0.9]['f_A_mean'].values[0]
        base = max(f07, f09, 1e-6)
        rel  = abs(f07 - f09) / base * 100
        verdict = 'default justified' if rel < 10 else f'SENSITIVE ({rel:.1f}%)'
        print(f'{arch:>5}  {phi:>7.1f}  {f07:>9.4f}  {f09:>9.4f}  {rel:>9.1f}%  {verdict}')
