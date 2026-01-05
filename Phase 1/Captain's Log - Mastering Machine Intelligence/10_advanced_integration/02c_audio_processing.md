# Specialization: Audio & Speech Processing (The Bard)

## 📜 Story Mode: The Bard

> **Mission Date**: 2043.12.01
> **Location**: Deep Space Outpost "Vector Prime" - Comms Array
> **Officer**: Specialist Lyra
>
> **The Problem**: The alien signal isn't text. It's sound.
> A complex waveform modulated at 40kHz.
> My FFT shows structure, but I need to *understand* it.
>
> **The Solution**: **Spectrograms**.
> I will convert time to frequency images.
> I will train a CNN to "see" the sound.
>
> *"Computer! Apply Mel-Filterbank. Run Whisper Encoder. Transcribe."*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**:
    *   **ASR**: Automatic Speech Recognition (Audio to Text).
    *   **TTS**: Text to Speech (Text to Audio).
    *   **Waveform**: 1D array of amplitude over time.
2.  **WHY**: Voice Assistants (Siri), Music Generation, Sonar.
3.  **WHEN**: Input is sound (Microphone).
4.  **WHERE**: `Librosa`, `Torchaudio`, `HuggingFace Audio`.
5.  **WHO**: Griffin-Lim (Signal reconstruction).
6.  **HOW**: Fourier Transform ($Time \to Freq$).

---

## 2. Mathematical Problem Formulation

### The Fourier Transform (DFT)
Decomposing a signal $x[n]$ into sine waves.
$$ X[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2\pi k n / N} $$

### The Spectrogram (STFT)
DFT over short windows (25ms).
Result: Image where X=Time, Y=Frequency, Color=Intensity.
**Mel Scale**: Humans hear log-scale. We squash frequencies to match human ear.

---

## 3. The Trifecta: Implementation Levels

### The Ship's Code (Polyglot: Pure Python + Libraries)

```python
import numpy as np
import math
import torch
import torchaudio

# LEVEL 0: Pure Python (DFT Logic - O(N^2))
def dft_pure(x):
    N = len(x)
    X = []
    for k in range(N):
        re = 0
        im = 0
        for n in range(N):
            phi = (2 * math.pi * k * n) / N
            re += x[n] * math.cos(phi)
            im -= x[n] * math.sin(phi)
        X.append(complex(re, im))
    return X

# LEVEL 1: NumPy (FFT - O(N log N))
def spectrogram_numpy(x, sr=16000):
    # Short-Time Fourier Transform
    D = np.abs(np.fft.rfft(x))
    return D

# LEVEL 2: PyTorch (Mel Spectrogram)
def audio_preprocess(waveform):
    # waveform: [1, Time]
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=64
    )
    return transform(waveform)
```

---

## 4. System-Level Integration (Whisper)

```mermaid
graph LR
    Mic[Microphone] --> Wave[Waveform]
    Wave --> Mel[Mel Spectrogram]
    Mel --> Conv[CNN Encoder]
    Conv --> Transformer[Transformer Decoder]
    Transformer --> Text[Tokens]
```

---

## 13. Assessment & Mastery Checks

**Q1: Nyquist Theorem**
If I want to capture 20kHz audio, what sample rate do I need?
*   *Answer*: > 40kHz (Double the max frequency). Standard is 44.1kHz.

**Q2: CTC Loss**
Why is ASR hard?
*   *Answer*: "Hello" (audio) is longer than "Hello" (text). Alignment is unknown. Connectionist Temporal Classification (CTC) solves this alignment.

### 14. Common Misconceptions (Debug Your Thinking)

> [!WARNING]
> **"MP3 is raw audio."**
> *   **Correction**: MP3 is compressed (lossy). For ML, use WAV or FLAC (lossless) to avoid artifacts confusing the model.
