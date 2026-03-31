import matplotlib
matplotlib.use('TkAgg')  # force correct backend on Windows

import pyaudio
import numpy as np
from scipy.signal import butter,lfilter, find_peaks
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections
import time
import sys

# --- config ---
RATE      = 44100
CHUNK     = 2048
CUTOFF    = 0.45
ORDER     = 3
WINDOW    = 500
MIN_DIST  = 25       # min samples between breath peaks
THRESH_K  = 0.5     # how many std above mean to set threshold

# --- mic setup ---
pa = pyaudio.PyAudio()
mic_index = None
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        mic_index = i
        print(f"mic: {info['name']}")
        break

if mic_index is None:
    print("no mic found"); sys.exit(1)

stream = pa.open(
    format=pyaudio.paFloat32, channels=1,
    rate=RATE, input=True,
    input_device_index=mic_index,
    frames_per_buffer=CHUNK
)

# --- butterworth low-pass ---
# keeps only slow amplitude swells (breathing), kills everything faster
nyq = (RATE / CHUNK) / 2
b, a = butter(ORDER, min(CUTOFF / nyq, 0.95), btype='low')
zi   = np.zeros(max(len(a), len(b)) - 1)

# --- data buffers ---
env_buf  = collections.deque([0.0] * WINDOW, maxlen=WINDOW)
raw_buf  = collections.deque([0.0] * WINDOW, maxlen=WINDOW)

# breath timestamps for BPM rolling average
breath_ts = collections.deque(maxlen=12)
last_counted = 0.0
bpm = 0.0

# --- plot setup ---
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 6), facecolor='#0d0d0d')
fig.subplots_adjust(hspace=0.5, top=0.82)

for ax in (ax_top, ax_bot):
    ax.set_facecolor('#111')
    ax.tick_params(colors='#555')
    for sp in ax.spines.values():
        sp.set_color('#222')

ax_top.set_title("raw rms", color='#555', fontsize=9, loc='left', pad=4)
ax_top.set_xlim(0, WINDOW)
ax_top.set_ylim(0, 0.05)
ax_top.set_xticks([]); ax_top.set_yticks([])

ax_bot.set_title("breath envelope  (green peaks = detected breaths)", color='#555', fontsize=9, loc='left', pad=4)
ax_bot.set_xlim(0, WINDOW)
ax_bot.set_ylim(0, 0.02)
ax_bot.set_xticks([]); ax_bot.set_yticks([])

xs = list(range(WINDOW))

line_raw, = ax_top.plot(xs, list(raw_buf), color='#1a6e8c', lw=0.9)
line_env, = ax_bot.plot(xs, list(env_buf), color='#27ae60', lw=1.4)
thr_line  = ax_bot.axhline(0.004, color='#e74c3c', lw=0.8, ls='--', alpha=0.7)

# scatter placeholder — redrawn each frame
peak_scatter = ax_bot.scatter([], [], color='#2ecc71', s=22, zorder=5)

# big BPM display above the plots
bpm_text   = fig.text(0.5, 0.93, "– –", ha='center', va='top',
                      fontsize=46, fontweight='bold', color='#2ecc71',
                      transform=fig.transFigure)
unit_text  = fig.text(0.5, 0.86, "breaths / min", ha='center',
                      color='#555', fontsize=9, transform=fig.transFigure)
state_text = fig.text(0.5, 0.835, "", ha='center',
                      color='#27ae60', fontsize=10, transform=fig.transFigure)

def update(_):
    global zi, last_counted, bpm

    # read mic
    data    = stream.read(CHUNK, exception_on_overflow=False)
    samples = np.frombuffer(data, dtype=np.float32)
    rms     = float(np.sqrt(np.mean(samples ** 2)))

    # filter — one sample keeps zi state continuous between frames
    filtered, zi = lfilter(b, a, [rms], zi=zi)
    env = abs(filtered[0])

    raw_buf.append(rms)
    env_buf.append(env)

    arr = np.array(env_buf)

    # adaptive threshold: mean + k*std, with a hard floor
    thresh = max(arr.mean() + THRESH_K * arr.std(), 0.003)
    thr_line.set_ydata([thresh])

    # find breath peaks in the full envelope window
    peaks, _ = find_peaks(arr, height=thresh, distance=MIN_DIST)

    # update peak scatter properly (no collections.clear() — just set offsets)
    if len(peaks):
        peak_scatter.set_offsets(np.column_stack([peaks, arr[peaks]]))
    else:
        peak_scatter.set_offsets(np.empty((0, 2)))

    # count a new breath only when a peak is near the right edge (fresh)
    now = time.time()
    if len(peaks) and peaks[-1] >= WINDOW - 5 and now - last_counted > 2.5:
        last_counted = now
        breath_ts.append(now)

    # BPM from inter-breath gaps, last 40 seconds only
    recent = [t for t in breath_ts if now - t < 40]
    if len(recent) >= 2:
        gaps = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        bpm  = 60.0 / (sum(gaps) / len(gaps))

    # update lines
    line_raw.set_ydata(list(raw_buf))
    line_env.set_ydata(list(env_buf))

    # auto-scale y axes
    r_peak = max(list(raw_buf)[-120:]) or 0.005
    e_peak = max(list(env_buf)[-120:]) or thresh
    ax_top.set_ylim(0, r_peak * 1.7)
    ax_bot.set_ylim(0, max(e_peak * 1.7, thresh * 2.5))

    # update text
    if bpm > 0:
        color = '#2ecc71' if bpm < 20 else '#e67e22' if bpm < 26 else '#e74c3c'
        bpm_text.set_text(f"{bpm:.1f}")
        bpm_text.set_color(color)
    else:
        bpm_text.set_text("– –")

    breathing = env > thresh * 0.75
    state_text.set_text("breathing…" if breathing else "")

ani = animation.FuncAnimation(
    fig, update,
    interval=int(1000 * CHUNK / RATE),
    blit=False,
    cache_frame_data=False
)

plt.show()

stream.stop_stream()
stream.close()
pa.terminate()
print(f"done — last BPM: {bpm:.1f}")
