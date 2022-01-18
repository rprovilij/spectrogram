import matplotlib.pyplot as plt
from scipy.io import wavfile
from skimage import util
import numpy as np
import pyaudio
import wave
import gc
import os

# AUDIO-IN
FORMAT 		= pyaudio.paInt16
CHANNELS 	= 2
RATE 		= 44100
CHUNK 		= 1024
RECORD_SECONDS 	= 2
WAV_AUDIO 	= "audio.wav"

audio = pyaudio.PyAudio()

# Start recording
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("> Recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	data = stream.read(CHUNK)
	frames.append(data)
print("finished recording")

# Stop recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAV_AUDIO, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

# Set path to audio file
rate, audio = wavfile.read('audio.wav')

# Convert audio to mono by averaging both channels
audio = np.mean(audio, axis=1)

# Calculating audio file length
N = audio.shape[0]
L = N / rate

print(f'audio length: {L:.2f} seconds')

# Discrete Fourier Transform (DFT) pre-process
M = 1024
slices = util.view_as_windows(audio, window_shape=(M,), step=100)
win = np.hanning(M + 1)[:-1]
slices = slices * win
slices = slices.T

# Calculate DFT
spectrum = np.fft.fft(slices, axis=0)[:M // 2 + 1:-1]
spectrum = np.abs(spectrum)

# Log plot
f, ax = plt.subplots(figsize=(12, 6))
S = np.abs(spectrum)
S = 20 * np.log10(S / np.max(S))

ax.imshow(S, origin='lower', cmap='ocean', extent=(0, L, 0, rate / 2 / 1000))
plt.axis('off')
ax.axis('tight')

plt.ylim(0, 9)  # Limit set due to max freq. detectability of microphone.
print('plotting spectrogram...')

# Save figure | adjust pixel density (Warning: this will influence size)
plt.savefig('spect.png', dpi=400, pad_inches=0.0)
plt.show()

plt.cla()
plt.close()
collected = gc.collect()
print("Garbage collector: collected %d objects" % collected)

# Clean-up
os.remove('audio.wav')
print('audio deleted...')

# os.remove('spect.png')
