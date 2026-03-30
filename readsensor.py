import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.fftpack import dct

def compute_mfcc(audio, sr, n_mfcc=13, n_fft=256, hop_length=256):
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize
    audio = audio.astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # Number of frames
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop_length]

    # Apply Hamming window
    frames = frames * np.hamming(n_fft)

    # FFT and power spectrum
    mag = np.abs(np.fft.rfft(frames, axis=1)) ** 2

    # Log energy
    log_energy = np.log(mag + 1e-10)

    # DCT → MFCC
    mfcc = dct(log_energy, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    return mfcc


def create_sensor_audio_matrix(folder_path, output_csv="final_matrix.csv"):
    sensor_data_list = []
    min_rows = float('inf')

    # ---- Read CSV files ----
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))

            # Drop first 2 columns
            df = df.iloc[:, 2:]

            min_rows = min(min_rows, len(df))
            sensor_data_list.append(df)

    # Truncate and stack
    sensor_matrix = np.hstack([df.iloc[:min_rows].values for df in sensor_data_list])

    # ---- Read WAV file ----
    wav_file = [f for f in os.listdir(folder_path) if f.endswith(".wav")][0]
    sr, audio = wavfile.read(os.path.join(folder_path, wav_file))

    # ---- Compute MFCC safely ----
    mfcc = compute_mfcc(audio, sr)

    # ---- Match row count ----
    N = mfcc.shape[0]

    if N > min_rows:
        mfcc = mfcc[:min_rows]
    elif N < min_rows:
        pad = np.zeros((min_rows - N, mfcc.shape[1]))
        mfcc = np.vstack([mfcc, pad])

    # ---- Combine ----
    final_matrix = np.hstack([sensor_matrix, mfcc])

    # ---- Save ----
    pd.DataFrame(final_matrix).to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print("Shape:", final_matrix.shape)

    return final_matrix

datapath = "D:/sensordata/Walk1_raod_002"
create_sensor_audio_matrix(datapath, output_csv= datapath + ".csv")
