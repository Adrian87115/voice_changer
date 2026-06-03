import librosa
import pyworld
import numpy as np
import soundfile as sf
import os
from torch.utils.data import Dataset
import torch
import math

class MCEPDataset(Dataset):
    def __init__(self, mcep_data, label_id):
        self.mcep_data = [torch.tensor(mcep, dtype = torch.float32) for mcep in mcep_data]
        self.label_id = label_id

    def __len__(self):
        return len(self.mcep_data)

    def __getitem__(self, idx):
        return self.mcep_data[idx], self.label_id

def calculateMcd(source, target):
    diff = source[..., 1:] - target[..., 1:] 
    squared_diff = diff ** 2
    sum_squared_diff = torch.sum(squared_diff, dim = -1)
    constant = (10.0 * math.sqrt(2.0)) / math.log(10.0)
    mcd = torch.mean(torch.sqrt(sum_squared_diff)) * constant
    return mcd.item()

def calculateMsd(source, target):
    diff = source - target
    squared_diff = diff ** 2
    msd = torch.mean(torch.sqrt(torch.sum(squared_diff, dim = -1)))
    return msd.item()

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
    tf = mcep.shape[0]
    np.savez(output_filename, f0 = f0, mcep = mcep, source_parameter = ap, time_frames = tf)

if __name__ == "__main__":
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)

    training_data_dir = os.path.join(project_root, 'data', 'training_data', 'audio')
    evaluation_data_dir = os.path.join(project_root, 'data', 'evaluation_data', 'audio')

    print(f"Processing training data at: {training_data_dir}")
    batchProcessAudio(training_data_dir)
    
    print(f"Processing evaluation data at: {evaluation_data_dir}")
    batchProcessAudio(evaluation_data_dir)