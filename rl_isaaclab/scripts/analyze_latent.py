"""
Latent vector & tactile force correlation analysis.

Usage:
    python analyze_latent.py --latent logs/latent_analysis/latent_XXX.npy \
                              --force  logs/latent_analysis/force_XXX.npy  \
                              [--action logs/latent_analysis/action_XXX.npy] \
                              [--out_dir figs/]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

parser = argparse.ArgumentParser()
parser.add_argument("--latent", type=str, required=True)
parser.add_argument("--force",  type=str, required=True)
parser.add_argument("--action", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="figs")
parser.add_argument("--control_freq", type=float, default=20.0, help="Hz")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
latent = np.load(args.latent)   # (T, 8)
force  = np.load(args.force)    # (T, 5)
T = latent.shape[0]
t = np.arange(T) / args.control_freq   # time axis in seconds

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
LATENT_NAMES = [f"z[{i}]" for i in range(latent.shape[1])]
print(f"Loaded {T} steps  ({T/args.control_freq:.1f} s)  "
      f"latent={latent.shape}  force={force.shape}")


# ── 1. Time-series ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax = axes[0]
for i in range(latent.shape[1]):
    ax.plot(t, latent[:, i], lw=0.8, label=LATENT_NAMES[i])
ax.set_ylabel("Latent value (tanh)")
ax.legend(ncol=4, fontsize=7, loc="upper right")
ax.set_title("Latent vector over time")
ax.axhline(0, color='k', lw=0.4, ls='--')

ax = axes[1]
for i, name in enumerate(FINGER_NAMES):
    ax.plot(t, force[:, i], lw=0.8, label=name)
ax.set_ylabel("Contact force (scaled)")
ax.set_xlabel("Time (s)")
ax.legend(ncol=5, fontsize=7, loc="upper right")
ax.set_title("Tactile force over time")

plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "1_timeseries.png"), dpi=150)
plt.close()
print("Saved: 1_timeseries.png")


# ── 2. Pearson correlation heatmap  (latent × force) ──────────────────────────
# Correlation matrix: rows=latent dims (8), cols=fingers (5)
corr = np.corrcoef(latent.T, force.T)[:latent.shape[1], latent.shape[1]:]
# corr shape: (8, 5)

fig, ax = plt.subplots(figsize=(6, 7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(5));  ax.set_xticklabels(FINGER_NAMES, fontsize=9)
ax.set_yticks(range(latent.shape[1])); ax.set_yticklabels(LATENT_NAMES, fontsize=9)
for i in range(latent.shape[1]):
    for j in range(5):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha='center', va='center', fontsize=8,
                color='white' if abs(corr[i,j]) > 0.5 else 'black')
plt.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Latent–Force Correlation")
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "2_correlation_heatmap.png"), dpi=150)
plt.close()
print("Saved: 2_correlation_heatmap.png")

# Print top correlations
print("\n[Top |r| > 0.3]")
for i in range(latent.shape[1]):
    for j in range(5):
        if abs(corr[i, j]) > 0.3:
            print(f"  {LATENT_NAMES[i]} ↔ {FINGER_NAMES[j]}:  r={corr[i,j]:+.3f}")


# ── 3. Cross-correlation (lag analysis) ────────────────────────────────────────
# For each (latent_dim, finger) pair with |r| > 0.2, find peak lag
MAX_LAG = int(args.control_freq * 2)  # ±2 seconds

def xcorr(a, b, max_lag):
    """Normalized cross-correlation, returns (lags_in_steps, values)."""
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    full = np.correlate(a, b, mode='full') / len(a)
    mid = len(full) // 2
    lags = np.arange(-max_lag, max_lag + 1)
    return lags, full[mid - max_lag: mid + max_lag + 1]

# Find pairs worth plotting
strong_pairs = [(i, j) for i in range(latent.shape[1])
                for j in range(5) if abs(corr[i, j]) > 0.2]

if strong_pairs:
    n_pairs = len(strong_pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 2.5 * n_pairs), squeeze=False)
    lags_sec = np.arange(-MAX_LAG, MAX_LAG + 1) / args.control_freq

    for idx, (i, j) in enumerate(strong_pairs):
        lags, xc = xcorr(latent[:, i], force[:, j], MAX_LAG)
        peak_lag = lags[np.argmax(np.abs(xc))]
        peak_val = xc[np.argmax(np.abs(xc))]
        ax = axes[idx][0]
        ax.plot(lags_sec, xc, lw=1.0)
        ax.axvline(0, color='k', lw=0.5, ls='--')
        ax.axvline(peak_lag / args.control_freq, color='r', lw=0.8, ls=':')
        ax.set_title(f"{LATENT_NAMES[i]} × {FINGER_NAMES[j]}  "
                     f"(peak lag={peak_lag/args.control_freq:+.2f}s, r={peak_val:+.3f})",
                     fontsize=9)
        ax.set_ylabel("xcorr")
        ax.set_xlabel("Lag (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "3_xcorr.png"), dpi=150)
    plt.close()
    print("Saved: 3_xcorr.png")
else:
    print("No pairs with |r| > 0.2, skipping cross-correlation plot.")


# ── 4. Contact event analysis ──────────────────────────────────────────────────
# Detect force onset/offset events and plot latent response window
WINDOW_BEFORE = int(args.control_freq * 1.0)   # 1 s before
WINDOW_AFTER  = int(args.control_freq * 2.0)   # 2 s after
FORCE_ONSET_THRESH = 0.1

total_force = force.sum(axis=1)   # scalar contact signal
# onset = rising edge (0→nonzero)
diff = np.diff((total_force > FORCE_ONSET_THRESH).astype(int))
onset_steps  = np.where(diff ==  1)[0] + 1
offset_steps = np.where(diff == -1)[0] + 1

print(f"\nDetected {len(onset_steps)} contact onset events.")

if len(onset_steps) > 0:
    n_events = min(len(onset_steps), 6)
    fig, axes = plt.subplots(n_events, 2, figsize=(12, 3 * n_events))
    if n_events == 1:
        axes = axes[np.newaxis, :]

    for k, onset in enumerate(onset_steps[:n_events]):
        s = max(0, onset - WINDOW_BEFORE)
        e = min(T, onset + WINDOW_AFTER)
        t_win = (np.arange(e - s) - (onset - s)) / args.control_freq

        ax_f = axes[k][0]
        for j, name in enumerate(FINGER_NAMES):
            ax_f.plot(t_win, force[s:e, j], lw=0.8, label=name)
        ax_f.axvline(0, color='r', lw=1, ls='--', label='onset')
        ax_f.set_title(f"Event {k+1} (step {onset}) — Force")
        ax_f.set_ylabel("Force")
        ax_f.legend(fontsize=7, ncol=3)

        ax_l = axes[k][1]
        for i in range(latent.shape[1]):
            ax_l.plot(t_win, latent[s:e, i], lw=0.8, label=LATENT_NAMES[i])
        ax_l.axvline(0, color='r', lw=1, ls='--', label='onset')
        ax_l.set_title(f"Event {k+1} (step {onset}) — Latent")
        ax_l.set_ylabel("Latent")
        ax_l.legend(fontsize=7, ncol=4)

    for ax in axes.flat:
        ax.set_xlabel("Time from onset (s)")
    plt.suptitle("Latent response around contact onset", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "4_contact_events.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 4_contact_events.png")


# ── 5. PCA of latent space, colored by total force ────────────────────────────
try:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    z2d = pca.fit_transform(latent)   # (T, 2)
    var_ratio = pca.explained_variance_ratio_

    total_force_norm = (total_force - total_force.min()) / (total_force.max() - total_force.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(z2d[:, 0], z2d[:, 1],
                    c=total_force_norm, cmap='plasma', s=4, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Total force (normalized)")
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title("Latent PCA — colored by total contact force")

    # draw time arrow (first→last)
    ax.annotate("", xy=z2d[-1], xytext=z2d[0],
                 arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "5_pca.png"), dpi=150)
    plt.close()
    print("Saved: 5_pca.png")
except ImportError:
    print("sklearn not found, skipping PCA plot.")


# ── 6. Latent variance per finger contact state ────────────────────────────────
# Compare latent distribution when each finger is in contact vs not
fig, axes = plt.subplots(latent.shape[1], 5, figsize=(14, 1.8 * latent.shape[1]))

for j, fname in enumerate(FINGER_NAMES):
    in_contact    = force[:, j] > FORCE_ONSET_THRESH
    not_in_contact = ~in_contact

    for i in range(latent.shape[1]):
        ax = axes[i][j]
        ax.hist(latent[not_in_contact, i], bins=40, alpha=0.6,
                color='steelblue', label='no contact', density=True)
        ax.hist(latent[in_contact, i], bins=40, alpha=0.6,
                color='tomato', label='contact', density=True)
        if i == 0:
            ax.set_title(fname, fontsize=9)
        if j == 0:
            ax.set_ylabel(LATENT_NAMES[i], fontsize=8)
        ax.tick_params(labelsize=6)

axes[0][-1].legend(fontsize=6, loc='upper right')
plt.suptitle("Latent distribution: contact vs no-contact per finger", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "6_latent_dist_per_finger.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 6_latent_dist_per_finger.png")


print(f"\nAll plots saved to: {os.path.abspath(args.out_dir)}")
