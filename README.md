# Day 06 — BreathClock

Microphone as an analog sensor. Detects breathing rhythm from raw audio amplitude, visualizes it live, and computes breaths-per-minute.

## Run
```
D:\BuildOrcas\run.bat Day6_BreathClock
```

## How it works
Each audio chunk gets its RMS computed — that's the amplitude. A Butterworth low-pass filter (0.45 Hz cutoff) removes everything faster than breathing. What's left is a smooth swell that rises and falls with each breath. `find_peaks` detects each cycle, timestamps it, and divides 60 by the average gap.

## Shipped
- [x] Mic captures audio
- [x] Live waveform and envelope plot
- [x] BPM updates with each detected breath
- [x] Adaptive threshold — no manual mic calibration needed

## Stack
`pyaudio` `scipy` `matplotlib` `numpy`
