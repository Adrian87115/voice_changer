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





import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
import soundfile as sf

# Example placeholders, replace with your actual data
mcc = np.random.randn(36, 512)  # Mel-cepstral coefficients (MCC)
f0 = np.random.rand(512) * 100  # F0 (fundamental frequency) values
aperiodicity = np.random.rand(513, 1340)  # Aperiodicity (513 frequency bins, 1340 frames)

# Resample/Interpolate aperiodicity to match 512 frames
n_frames_target = 512
n_frequency_bins = aperiodicity.shape[0]  # 513 frequency bins

# Create interpolation function for each frequency bin
interpolated_aperiodicity = np.zeros((n_frequency_bins, n_frames_target))
x_old = np.linspace(0, 1, aperiodicity.shape[1])  # Original 1340 frames
x_new = np.linspace(0, 1, n_frames_target)        # New 512 frames

for i in range(n_frequency_bins):
    interp_func = interp1d(x_old, aperiodicity[i], kind='linear')
    interpolated_aperiodicity[i] = interp_func(x_new)

# Now the aperiodicity has the shape (513, 512), which matches the expected shape
aperiodicity_resized = interpolated_aperiodicity.T  # Shape should be (512, 513)

# Ensure MCC and aperiodicity_resized are C-contiguous before passing to PyWorld
mcc_contiguous = np.ascontiguousarray(mcc.T)  # Transpose and make C-contiguous
aperiodicity_resized_contiguous = np.ascontiguousarray(aperiodicity_resized)  # Make C-contiguous

# Convert MCC to spectrogram
spectrogram = pw.decode_spectral_envelope(mcc_contiguous, fs=16000, fft_size=1024)

# Ensure the spectrogram is C-contiguous
spectrogram_contiguous = np.ascontiguousarray(spectrogram)

# Synthesize the waveform
wav = pw.synthesize(f0, spectrogram_contiguous, aperiodicity_resized_contiguous, fs=16000)
print(f0.shape, spectrogram.shape, aperiodicity_resized_contiguous.shape)
# Save the wav file
sf.write('output.wav', wav, 16000)
