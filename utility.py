import librosa
import numpy as np
import pyworld as pw
import scipy.io.wavfile as wav
from scipy.interpolate import interp1d

def mcc_to_spectrogram(mcc, sr=22050, n_mels=24):
    # Remove batch dimension and move to CPU if required (assuming PyTorch tensor)
    if hasattr(mcc, 'cpu'):
        mcc = mcc.squeeze().cpu().numpy()
    else:
        mcc = mcc.squeeze()

    # Apply inverse DCT to get Mel-spectrogram from MCC
    mel_spectrogram = librosa.feature.inverse.mfcc_to_mel(mcc)

    # Convert Mel-spectrogram back to a linear spectrogram (STFT)
    spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=sr)

    return spectrogram

def generate_waveform(f0, spectrogram, aperiodicity, fs=22050):
    """
    Generate a waveform using the WORLD vocoder.
    Arguments:
        f0: Fundamental frequency contour (1D array)
        spectrogram: Spectrogram obtained from MCC (2D array)
        aperiodicity: Aperiodicity for the WORLD vocoder (2D array)
        fs: Sampling frequency
    """
    # Ensure that f0, spectrogram, and aperiodicity are of type np.float64
    f0 = f0.astype(np.float64)
    spectrogram = spectrogram.astype(np.float64)
    aperiodicity = aperiodicity.astype(np.float64)

    # Check the dimensions and reshape if necessary
    if f0.ndim != 1:
        raise ValueError(f"f0 must be a 1D array, but got {f0.ndim}D")

    if spectrogram.ndim != 2:
        raise ValueError(f"spectrogram must be a 2D array, but got {spectrogram.ndim}D")

    if aperiodicity.ndim != 2:
        raise ValueError(f"aperiodicity must be a 2D array, but got {aperiodicity.ndim}D")

    # Ensure spectrogram and aperiodicity have the same number of frames as f0
    if spectrogram.shape[0] != f0.shape[0]:
        # Interpolate spectrogram to match the number of frames in f0
        x = np.linspace(0, spectrogram.shape[0] - 1, spectrogram.shape[0])
        x_new = np.linspace(0, spectrogram.shape[0] - 1, f0.shape[0])
        interp_func = interp1d(x, spectrogram, kind='linear', axis=0, fill_value="extrapolate")
        spectrogram = interp_func(x_new)

    if aperiodicity.shape[0] != f0.shape[0]:
        # Interpolate aperiodicity to match the number of frames in f0
        aperiodicity = interp1d(x, aperiodicity, kind='linear', axis=0, fill_value="extrapolate")(x_new)

    # Reconstruct the waveform from the spectral envelope, F0, and aperiodicity
    waveform = pw.synthesize(f0, spectrogram, aperiodicity, fs)

    return waveform

def save_waveform_to_wav(waveform, filename, fs=22050):
    """
    Save the generated waveform to a .wav file.
    Arguments:
        waveform: The generated waveform
        filename: Path to save the .wav file
        fs: Sampling frequency
    """
    wav.write(filename, fs, (waveform * 32767).astype(np.int16))

def convert_mcc_to_wav(mcc, f0, aperiodicity, output_wav_file, sr=22050):
    """
    Convert MCC to a WAV file using the WORLD vocoder.
    Arguments:
        mcc: Mel-Cepstral Coefficients
        f0: Fundamental frequency (pitch) contour
        aperiodicity: Aperiodicity (noise component in the waveform)
        output_wav_file: Path to save the output .wav file
    """
    # Convert MCC to spectrogram
    spectrogram = mcc_to_spectrogram(mcc, sr=sr)

    # Generate the waveform using the WORLD vocoder
    waveform = generate_waveform(f0, spectrogram, aperiodicity, fs=sr)

    # Save the waveform to a .wav file
    save_waveform_to_wav(waveform, output_wav_file)
    print(f"Waveform saved to: {output_wav_file}")
