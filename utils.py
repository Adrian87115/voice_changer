import librosa
import pyworld
import numpy as np
import soundfile as sf
import os
from scipy.ndimage import zoom
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt

class MCEPDataset(Dataset):
    def __init__(self, mcep_data, label_id):
        self.mcep_data = [torch.tensor(mcep, dtype=torch.float32) for mcep in mcep_data]
        self.label_id = label_id

    def __len__(self):
        return len(self.mcep_data)

    def __getitem__(self, idx):
        return self.mcep_data[idx], self.label_id

def scaleDown(mcep, target_size = 128):
    original_size = mcep.shape[1]
    scale_factor = target_size / original_size
    mcep_scaled_down = zoom(mcep, (1, scale_factor), order=1)
    return mcep_scaled_down

def scaleUp(mcep, original_size = 512):
    target_size = original_size
    current_size = mcep.shape[1]
    scale_factor = target_size / current_size
    mcep_scaled_up = zoom(mcep, (1, scale_factor), order=1)
    return mcep_scaled_up

def pitchShiftWavFileTest(pitch_dataset, wav_file_path, output_wav_path):
    y, sr = librosa.load(wav_file_path, sr=None)
    f0, sp, ap = pyworld.wav2world(y.astype(np.float64), sr)
    log_f0 = np.log(f0 + 1e-5)
    exp_f0 = np.exp(log_f0)
    f0_converted = pitch_dataset.pitchConversion(log_f0)
    plt.subplot(2, 2, 1)
    plt.plot(f0, label='F0')
    plt.title('F0')
    plt.xlabel('Time Frames')
    plt.ylabel('F0')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(log_f0, label='log_f0')
    plt.title('log_f0')
    plt.xlabel('Time Frames')
    plt.ylabel('log_f0')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(exp_f0, label='exp_f0')
    plt.title('exp_f0')
    plt.xlabel('Time Frames')
    plt.ylabel('exp_f0')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(f0_converted, label='Converted F0 (log)')
    plt.title('Converted Log F0')
    plt.xlabel('Time Frames')
    plt.ylabel('Converted Log F0')
    plt.legend()
    plt.tight_layout()
    plt.show()
    file = pyworld.synthesize(f0_converted, sp, ap, 22050, 5.0)
    sf.write(output_wav_path, file, sr)

def loadWav(wav_file, sr):
    wav, _ = librosa.load(wav_file, sr = sr, mono = True)
    return wav

def decomposeWav(wav_file, fs = 22050, frame_period = 5.0, mcep_dim = 35):
    wav = loadWav(wav_file, sr = fs)
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    mcep = pyworld.code_spectral_envelope(sp, fs, mcep_dim)
    return f0, mcep, ap, fs, frame_period

def decodeMcep(mcep, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(mcep, fs, fftlen)
    return decoded_sp

def reassembleWav(f0, mcep, ap, fs, frame_period):
    decoded_sp = decodeMcep(mcep, fs)
    synthesized_wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    return synthesized_wav

def saveWav(wav, output_file, fs):
    sf.write(output_file, wav, fs)

def processWav(wav_file, output_file, fs, frame_period = 5.0, mcep_dim = 35):
    f0, mcep, ap, fs, frame_period = decomposeWav(wav_file, fs, frame_period, mcep_dim)
    reassembled_wav = reassembleWav(f0, mcep, ap, fs, frame_period)
    saveWav(reassembled_wav, output_file, fs)

def batchProcessAudio(input_dir):
    parent_dir = os.path.dirname(input_dir)
    output_dir = os.path.join(parent_dir, 'transformed_audio')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    speaker_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    speaker_folders = [folder for folder in speaker_folders if folder not in ['transformed_audio', '.', '..']]

    for speaker_id in speaker_folders:
        input_speaker_folder = os.path.join(input_dir, speaker_id)
        output_speaker_folder = os.path.join(output_dir, speaker_id)
        if not os.path.exists(output_speaker_folder):
            os.makedirs(output_speaker_folder)
        audio_files = [f for f in os.listdir(input_speaker_folder) if f.endswith('.wav')]
        for audio_file in audio_files:
            input_filename = os.path.join(input_speaker_folder, audio_file)
            output_filename = os.path.join(output_speaker_folder, audio_file[:-4] + '.npz')
            processAudio(input_filename, output_filename)

def processAudio(input_filename, output_filename):
    fs = 22050
    f0, mcep, ap, fs, frame_period = decomposeWav(input_filename, fs)
    log_f0 = np.log(f0 + 1e-5)
    log_f0[np.isneginf(log_f0)] = np.nan
    mean_log_f0 = np.nanmean(log_f0)
    std_log_f0 = np.nanstd(log_f0)
    norm_log_f0 = (log_f0 - mean_log_f0) / std_log_f0
    tf = mcep.shape[0]
    np.savez(output_filename, log_f0 = log_f0, norm_log_f0 = norm_log_f0, mean_log_f0 = mean_log_f0, std_log_f0 = std_log_f0, mcep = mcep, source_parameter = ap, time_frames = tf)

# processWav("C:/Users/adria/Desktop/test/audio/VCC2SF1/10001.wav", "C:/Users/adria/Desktop/test/audio/VCC2SF1/output.wav", fs = 22050, frame_period = 5.0, mcep_dim = 35)

# batchProcessAudio("C:/Users/adria/Desktop/Adrian/projects/PyCharm/voice_changer/training_data/audio")
# batchProcessAudio("C:/Users/adria/Desktop/Adrian/projects/PyCharm/voice_changer/evaluation_data/audio")