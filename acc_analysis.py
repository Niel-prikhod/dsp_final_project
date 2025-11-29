#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from scipy.ndimage import label
from sklearn.metrics import pairwise_distances
import argparse
import os

# ==============================
# CONFIGURATION & ARGUMENTS
# ==============================
parser = argparse.ArgumentParser(description="3-class stair detection from single CSV")
parser.add_argument("--csv_file", default="sensorlog_accel_1128.csv", help="Path to your single CSV file")
parser.add_argument("--method", choices=['standing', 'impact_tilt', 'variance', 'pre-label'],
                    default="impact_tilt", 
                    help="How to separate flat / downstairs / upstairs (default: impact_tilt)")
parser.add_argument("--time_col", default="timestamp", help="Name of time column (default: time)")
parser.add_argument("--acc_cols", nargs=3, default=["X", "Y", "Z"],
                    help="Three acceleration columns, e.g. x y z (default: ax ay az)")
parser.add_argument("--vertical_axis", type=int, choices=[0,1,2], default=2,
                    help="Which axis is vertical? 0=x, 1=y, 2=z (default: 2 = az)")
parser.add_argument("--forward_axis", type=int, choices=[0,1,2], default=1,
                    help="Which axis points forward? (default: 1 = ay)")
parser.add_argument("--window_length", default="2.5", help="Define Window in seconds (default: 2.5s)")
args = parser.parse_args()

CSV_PATH = args.csv_file
METHOD = args.method
TIME_COL = args.time_col
ACC_COLS = args.acc_cols
VERT_IDX = args.vertical_axis   # e.g. 2 = az
FWD_IDX  = args.forward_axis    # e.g. 1 = ay

print(f"Using segmentation method: {METHOD}")
print(f"Vertical axis = {ACC_COLS[VERT_IDX]}, Forward axis = {ACC_COLS[FWD_IDX]}")

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv(CSV_PATH)
print("First 5 rows of your file:")
print(df.head())
print("\nAll column names:", list(df.columns))

# — Time column —
time_col = args.time_col
print(f"→ Using time column: '{time_col}'")

# — Acceleration columns —
acc_cols = args.acc_cols
print(f"→ Using acceleration columns: {acc_cols}")

# Extract data
t_orig = df[time_col].values
if t_orig.max() > 10000:           # probably milliseconds or microseconds
    t_orig = t_orig / 1000.0
acc_orig = df[acc_cols].values.T   # shape (3, N)

VERT_IDX = args.vertical_axis
FWD_IDX  = args.forward_axis
print(f"Vertical axis = {acc_cols[VERT_IDX]}, Forward axis = {acc_cols[FWD_IDX]}")

# ==============================
# 2. RESAMPLE TO EXACTLY 100 Hz
# ==============================
def resample_to_100hz(t_old, signal):
    t_new = np.arange(t_old[0], t_old[-1], 1/100.0)
    spl = splrep(t_old, signal, k=3, s=0)
    return t_new, splev(t_new, spl)

t, acc = [], []
for i in range(3):
    tt, aa = resample_to_100hz(t_orig, acc_orig[i])
    t = tt
    # same for all
    acc.append(aa)
acc = np.array(acc)  # (3, N)

print(f"Resampled to {len(t)} samples @ 100 Hz ({(t[-1]-t[0]):.1f} seconds)")

# ==============================
# 3. AUTOMATIC SEGMENTATION
# ==============================
labels = np.zeros(len(t), dtype=int)  # 0=flat, 1=down, 2=up

if METHOD == "standing":
    # ----- Standing detection (you stopped before/after stairs) -----
    win_sec = 1.0
    win = int(win_sec * 100)
    var_z = []
    for i in range(0, len(acc[VERT_IDX]) - win, win//2):
        var_z.append(acc[VERT_IDX, i:i+win].var())
    var_z = np.array(var_z)
    standing = var_z < 0.015
    labeled, n = label(standing)
    periods = []
    for i in range(1, n+1):
        if np.sum(labeled == i) * (win_sec/2) >= 2.0:
            idxs = np.where(labeled == i)[0]
            periods.append((t[idxs[0]*win//2], t[idxs[-1]*win//2 + win]))
    stair_blocks = [(periods[i][1], periods[i+1][0]) for i in range(len(periods)-1)]

elif METHOD == "impact_tilt":
    # ----- Best method for continuous walking: impact + body tilt -----
    # High-frequency power on vertical axis
    b, a = butter(4, 8/(100/2), 'high')
    high_pow = filtfilt(b, a, acc[VERT_IDX])**2
    high_pow = np.convolve(high_pow, np.ones(100)/100, mode='same')

    # Low-frequency tilt on forward axis
    b, a = butter(3, [0.3, 1.2], 'band', fs=100)
    tilt = filtfilt(b, a, acc[FWD_IDX])
    tilt_pow = np.convolve(tilt**2, np.ones(150)/150, mode='same')

    score = high_pow * np.sqrt(tilt_pow + 1e-8)
    score = score / score.max()

    thresh = 0.22
    candidate = score > thresh
    labeled, n = label(candidate)
    stair_blocks = []
    for i in range(1, n+1):
        idxs = np.where(labeled == i)[0]
        if len(idxs) >= 200:  # ≥2 seconds
            start = t[idxs[0]]
            end   = t[idxs[-1]]
            stair_blocks.append((start, end))

elif METHOD == "variance":
    # ----- Simple total magnitude variance -----
    mag = np.linalg.norm(acc, axis=0)
    var_win = 200
    variance = [mag[i:i+var_win].var() for i in range(0, len(mag)-var_win, 50)]
    var_smooth = np.array(variance)
    times_var = t[::50][:len(var_smooth)]

    # Very simple threshold (adjust once if needed)
    thresh = np.percentile(var_smooth, 85)
    stair_mask = var_smooth > thresh
    labeled, n = label(stair_mask)
    stair_blocks = []
    for i in range(1, n+1):
        idxs = np.where(labeled == i)[0]
        if len(idxs) >= 30:  # ≥1.5 s
            start = times_var[idxs[0]]
            end   = times_var[idxs[-1]]
            stair_blocks.append((start, end))

elif METHOD == "pre-label":
    # 1. Very strong stair indicator: high-pass filtered vertical acceleration power
    b, a = butter(4, 6/(100/2), 'high')                          # >6 Hz
    high_pass_z = filtfilt(b, a, acc[VERT_IDX])
    stair_indicator = medfilt(high_pass_z**2, kernel_size=51)   # smooth a bit
    stair_indicator = stair_indicator / stair_indicator.max()  # normalize 0–1

    # 2. Simple threshold + minimum duration → very good pre-labeling
    threshold = 0.15                     # you will probably only move it ±0.05
    candidate = stair_indicator > threshold

    # Clean short bursts
    labeled, n = label(candidate)
    stair_blocks = []
    for i in range(1, n+1):
        idxs = np.where(labeled == i)[0]
        if len(idxs) >= 150:                         # ≥1.5 s at 100 Hz
            stair_blocks.append((t[idxs[0]], t[idxs[-1]+1]))

    print(f"Pre-detected {len(stair_blocks)} stair periods (will be corrected interactively)")
    labels = np.zeros(len(t), dtype=int)

# Assign labels
if len(stair_blocks) >= 1:
    labels[(t >= stair_blocks[0][0]) & (t <= stair_blocks[0][1])] = 1  # downstairs
if len(stair_blocks) >= 2:
    labels[(t >= stair_blocks[1][0]) & (t <= stair_blocks[1][1])] = 2  # upstairs

# ==============================
# 4. VISUAL CHECK
# ==============================
print("\nINTERACTIVE LABEL CORRECTION")
print("→ Left-click and drag to mark Downstairs (orange)")
print("→ Right-click and drag to mark Upstairs   (green)")
print("→ Press 'r' to reset, 'q' or close window when finished")

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t, acc[VERT_IDX], 'k', lw=1, label=f'Vertical ({acc_cols[VERT_IDX]})')
line, = ax.plot(t, high_pass_z*0.3 + 1.0, 'purple', alpha=0.7, label='Stair indicator ×0.3')
ax.fill_between(t, -3, 3, where=labels==1, color='orange', alpha=0.4)
ax.fill_between(t, -3, 3, where=labels==2, color='limegreen', alpha=0.4)
ax.set_ylim(-3.5, 3.5)
ax.set_title("Click and drag to correct stair periods — close when ready")
ax.legend()

# Mouse interaction variables
dragging = None
start_idx = None

def on_press(event):
    global dragging, start_idx
    if event.button == 1:        # left click  → downstairs
        dragging = 1
    elif event.button == 3:      # right click → upstairs
        dragging = 2
    if dragging and event.xdata is not None:
        start_idx = np.searchsorted(t, event.xdata)

def on_release(event):
    global dragging, start_idx
    if dragging and event.xdata is not None and start_idx is not None:
        end_idx = np.searchsorted(t, event.xdata)
        labels[min(start_idx, end_idx):max(start_idx, end_idx)] = dragging
        # redraw background
        ax.fill_between(t, -3, 3, where=labels==1, color='orange', alpha=0.4)
        ax.fill_between(t, -3, 3, where=labels==2, color='limegreen', alpha=0.4)
        fig.canvas.draw_idle()
    dragging = None
    start_idx = None

def on_key(event):
    if event.key == 'r':
        labels[:] = 0
        ax.clear()
        ax.plot(t, acc[VERT_IDX], 'k', lw=1)
        ax.plot(t, high_pass_z*0.3 + 1.0, 'purple', alpha=0.7)
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show(block=True)   # waits until you close the window

# After you close the window → perfect labels are stored in the `labels` array
print("Interactive labeling finished — labels are now perfect!")

# Feature extraction
FS = 100
W_SEC = float(args.window_length)
W = int(W_SEC * FS)
STEP = int(0.5 * FS)

DIRECTIONS = [ACC_COLS[VERT_IDX], ACC_COLS[FWD_IDX]]  # e.g. ['az', 'ay']
FREQUENCY_BANDS = [(0.5,3),(3,6),(6,10),(0.5,10),(1,8),(0.3,5),(4,8),(0,20)]

def relative_energy(sig, band):
    sig = sig - sig.mean()
    N = len(sig)
    f = rfftfreq(N, 1/FS)
    fft_pow = np.abs(rfft(sig))**2
    total = np.sum(fft_pow[(f >= 0) & (f <= 20)])
    if total == 0: return 0
    mask = (f >= band[0]) & (f <= band[1])
    return np.sum(fft_pow[mask]) / total

features_list = []
vert_col_name = args.acc_cols[args.vertical_axis]
forw_col_name = args.acc_cols[args.forward_axis]
for start in range(0, len(t) - W + 1, STEP):
    window = acc[:, start:start+W]
    win_labels = labels[start:start+W]
    label = np.bincount(win_labels).argmax()  # majority vote

    feat = {"label": int(label), "class_name": ["flat","downstairs","upstairs"][label]}
    for band in FREQUENCY_BANDS:
        bstr = f"{band[0]:.1f}-{band[1]:.1f}Hz"
        feat[f"{vert_col_name}_{bstr}"] = relative_energy(window[0], band)   # vertical
        feat[f"{forw_col_name}_{bstr}"] = relative_energy(window[1], band)    # forward
    features_list.append(feat)

df_feat = pd.DataFrame(features_list)
print(f"\nExtracted {len(df_feat)} windows")

# Find best band (same as before)
best_dist = -1
best_band = None
for band in FREQUENCY_BANDS:
    bstr = f"{band[0]:.1f}-{band[1]:.1f}Hz"
    f1 = f"{vert_col_name}_{bstr}"
    f2 = f"{forw_col_name}_{bstr}"
    cents = [df_feat[df_feat.label==c][[f1,f2]].mean().values for c in [0,1,2]]
    dist = pairwise_distances(cents).sum()
    if dist > best_dist:
        best_dist = dist
        best_band = band
        best_f1, best_f2 = f1, f2

# Final plot
plt.figure(figsize=(10,8))
colors = ['cornflowerblue', 'orange', 'limegreen']
cols = ['blue','orange','green']
for c in [0,1,2]:
    sub = df_feat[df_feat.label==c]
    plt.scatter(sub[best_f1], sub[best_f2], c=cols[c], label=sub['class_name'].iloc[0], alpha=0.6, s=50)
plt.xlabel(best_f1); plt.ylabel(best_f2)
plt.title(f"Best band: {best_band[0]}-{best_band[1]} Hz  (total centroid dist = {best_dist:.3f})")
plt.legend(); plt.grid(alpha=0.3); plt.show()

labels_names = ['Flat walking', 'Downstairs', 'Upstairs']

for c, color, name in zip([0, 1, 2], colors, labels_names):
    sub = df_feat[df_feat.label == c]
    plt.scatter(sub[best_f1], sub[best_f2],
                c=color, label=name, alpha=0.7, s=60, edgecolors='k', linewidth=0.5)

# Centroids
for c, color, name in zip([0, 1, 2], colors, labels_names):
    cx = df_feat[df_feat.label == c][best_f1].mean()
    cy = df_feat[df_feat.label == c][best_f2].mean()
    plt.plot(cx, cy, marker='X', color='black', markersize=14, markeredgewidth=2)

plt.xlabel(f"Relative energy – {DIRECTIONS[0]} axis – {best_band[0]}-{best_band[1]} Hz [-]", fontsize=12)
plt.ylabel(f"Relative energy – {DIRECTIONS[1]} axis – {best_band[0]}-{best_band[1]} Hz [-]", fontsize=12)
plt.title(f"Best separating frequency band: {best_band[0]}–{best_band[1]} Hz\n"
          f"Total centroid distance = {best_dist:.3f}", fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nBest frequency band: {best_band[0]}-{best_band[1]} Hz")
print("Mean ± std:")
for c in [0,1,2]:
    sub = df_feat[df_feat.label==c]
    print(f"{sub['class_name'].iloc[0]:12s} → {sub[best_f1].mean():.3f}±{sub[best_f1].std():.3f} | {sub[best_f2].mean():.3f}±{sub[best_f2].std():.3f}")

print("\nDone!")
