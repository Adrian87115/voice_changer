import librosa
import pyworld
import numpy as np
import soundfile as sf
import os
from scipy.ndimage import zoom

def loadWav(wav_file, sr):
    wav, _ = librosa.load(wav_file, sr = sr, mono = True)
    return wav

def decomposeWav(wav_file, fs = 22050, frame_period = 5.0, mcc_dim = 35):
    wav = loadWav(wav_file, sr = fs)
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    mcc = pyworld.code_spectral_envelope(sp, fs, mcc_dim)
    return f0, mcc, ap, fs, frame_period

def decodeMcc(mcc, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(mcc, fs, fftlen)
    return decoded_sp

def reassembleWav(f0, mcc, ap, fs, frame_period):
    decoded_sp = decodeMcc(mcc, fs)
    synthesized_wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    return synthesized_wav

def saveWav(wav, output_file, fs):
    sf.write(output_file, wav, fs)

def processWav(wav_file, output_file, fs, frame_period = 5.0, mcc_dim = 35):
    f0, mcc, ap, fs, frame_period = decomposeWav(wav_file, fs, frame_period, mcc_dim)
    reassembled_wav = reassembleWav(f0, mcc, ap, fs, frame_period)
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
    f0, mcc, ap, fs, frame_period = decomposeWav(input_filename, fs)
    log_f0 = np.log(f0 + 1e-5)
    log_f0[np.isneginf(log_f0)] = np.nan
    mean_log_f0 = np.nanmean(log_f0)
    std_log_f0 = np.nanstd(log_f0)
    norm_log_f0 = (log_f0 - mean_log_f0) / std_log_f0
    tf = mcc.shape[0]
    np.savez(output_filename, norm_log_f0 = norm_log_f0, mean_log_f0 = mean_log_f0, std_log_f0 = std_log_f0, mcc = mcc, source_parameter = ap, time_frames = tf)

def resizeBatchAudio(input_dir):
    parent_dir = os.path.dirname(input_dir)
    output_dir = os.path.join(parent_dir, 'resized_audio')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    speaker_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    speaker_folders = [folder for folder in speaker_folders if folder not in ['transformed_audio', 'resized_audio', '.', '..']]

    for speaker_id in speaker_folders:
        input_speaker_folder = os.path.join(input_dir, speaker_id)
        output_speaker_folder = os.path.join(output_dir, speaker_id)
        if not os.path.exists(output_speaker_folder):
            os.makedirs(output_speaker_folder)
        npz_files = [f for f in os.listdir(input_speaker_folder) if f.endswith('.npz')]
        for npz_file in npz_files:
            input_filename = os.path.join(input_speaker_folder, npz_file)
            output_filename = os.path.join(output_speaker_folder, npz_file[:-4] + '.npz')
            data = np.load(input_filename)
            tf = data['time_frames']
            mcc = data['mcc']
            norm_log_f0 = data['norm_log_f0']
            mean_log_f0 = data['mean_log_f0']
            std_log_f0 = data['std_log_f0']
            source_parameter = data['source_parameter']
            mcc = zoom(mcc, (512 / mcc.shape[0], 1), order = 1)
            norm_log_f0 = zoom(norm_log_f0, (512 / norm_log_f0.size,), order = 1)
            np.savez(output_filename, norm_log_f0 = norm_log_f0, mean_log_f0 = mean_log_f0, std_log_f0 = std_log_f0, mcc = mcc, source_parameter = source_parameter, time_frames = tf)

# processWav("C:/Users/adria/Desktop/test/audio/VCC2SF1/10001.wav", "C:/Users/adria/Desktop/test/audio/VCC2SF1/output.wav", fs = 22050, frame_period = 5.0, mcc_dim = 35)
#
# batchProcessAudio("C:/Users/adria/Desktop/Adrian/projects/PyCharm/voice_changer/training_data/audio")
# batchProcessAudio("C:/Users/adria/Desktop/Adrian/projects/PyCharm/voice_changer/reference_data/audio")
# batchProcessAudio("C:/Users/adria/Desktop/Adrian/projects/PyCharm/voice_changer/evaluation_data/audio")
#
# resizeBatchAudio("C:/Users/adria/Desktop/Adrian/projects/PyCharm/voice_changer/training_data/transformed_audio")
# resizeBatchAudio("C:/Users/adria/Desktop/Adrian/projects/PyCharm/voice_changer/reference_data/transformed_audio")
# resizeBatchAudio("C:/Users/adria/Desktop/Adrian/projects/PyCharm/voice_changer/evaluation_data/transformed_audio")

