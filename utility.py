import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.io
from scipy.fftpack import dct
from scipy.fftpack import idct

# # Example placeholders, replace with your actual data
# mcc = np.random.randn(36, 512)  # Mel-cepstral coefficients (MCC)
# f0 = np.random.rand(512) * 100  # F0 (fundamental frequency) values
# aperiodicity = np.random.rand(513, 1340)  # Aperiodicity (513 frequency bins, 1340 frames)
#
# # Resample/Interpolate aperiodicity to match 512 frames
# n_frames_target = 512
# n_frequency_bins = aperiodicity.shape[0]  # 513 frequency bins
#
# # Create interpolation function for each frequency bin
# interpolated_aperiodicity = np.zeros((n_frequency_bins, n_frames_target))
# x_old = np.linspace(0, 1, aperiodicity.shape[1])  # Original 1340 frames
# x_new = np.linspace(0, 1, n_frames_target)        # New 512 frames
#
# for i in range(n_frequency_bins):
#     interp_func = interp1d(x_old, aperiodicity[i], kind='linear')
#     interpolated_aperiodicity[i] = interp_func(x_new)
#
# # Now the aperiodicity has the shape (513, 512), which matches the expected shape
# aperiodicity_resized = interpolated_aperiodicity.T  # Shape should be (512, 513)
#
# # Ensure MCC and aperiodicity_resized are C-contiguous before passing to PyWorld
# mcc_contiguous = np.ascontiguousarray(mcc.T)  # Transpose and make C-contiguous
# aperiodicity_resized_contiguous = np.ascontiguousarray(aperiodicity_resized)  # Make C-contiguous
#
# # Convert MCC to spectrogram
# spectrogram = pw.decode_spectral_envelope(mcc_contiguous, fs=16000, fft_size=1024)
#
# # Ensure the spectrogram is C-contiguous
# spectrogram_contiguous = np.ascontiguousarray(spectrogram)
#
# # Synthesize the waveform
# wav = pw.synthesize(f0, spectrogram_contiguous, aperiodicity_resized_contiguous, fs=16000)
# print(f0.shape, spectrogram.shape, aperiodicity_resized_contiguous.shape)
# # Save the wav file
# sf.write('output.wav', wav, 16000)

def plot_mcc_comparison(original_mcc, fake_mcc_interp):
    # Ensure both MCCs have the same shape for comparison
    if original_mcc.shape != fake_mcc_interp.shape:
        print(f"Original MCC shape: {original_mcc.shape}")
        print(f"Fake MCC shape: {fake_mcc_interp.shape}")
        raise ValueError("Original and fake MCCs must have the same shape for comparison!")

    # Plot the original and fake MCCs
    plt.figure(figsize=(12, 6))

    # Plot original MCC
    plt.subplot(1, 2, 1)
    plt.imshow(original_mcc, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('Original MCC')
    plt.xlabel('Time Frames')
    plt.ylabel('MCC Coefficients')

    # Plot fake MCC
    plt.subplot(1, 2, 2)
    plt.imshow(fake_mcc_interp, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('Fake MCC')
    plt.xlabel('Time Frames')
    plt.ylabel('MCC Coefficients')

    # Show plots
    plt.tight_layout()
    plt.show()
    plt.show()

def normLogf0Tof0(logf0):
    f0 = np.exp(logf0)
    return f0

def process_wav_file(wav_path):
    # Read the wav file
    fs, x = wavfile.read(wav_path)

    # If the file is stereo, convert to mono by averaging the two channels
    if x.ndim == 2:
        x = np.mean(x, axis=1)

    # Normalize the waveform if it's in integer format (16-bit PCM)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / np.max(np.abs(x))

    # Step 1: Raw pitch extractor (F0)
    _f0, t = pw.dio(x, fs)  # DIO: estimated F0 and temporal positions

    # Step 2: Pitch refinement
    f0 = pw.stonemask(x, _f0, t, fs)  # Refine F0 by StoneMask

    # Step 3: Spectrogram extraction
    sp = pw.cheaptrick(x, f0, t, fs)  # Smooth spectrogram
    sp_t = sp.T  # Transpose to get shape (513, 1228)

    # Initialize MCC matrix
    mcc = np.zeros((sp_t.shape[1], 36))  # Initialize with time frames x 36

    # Calculate MCC
    for i in range(sp_t.shape[1]):
        log_spectrum = np.log(sp_t[:, i] + np.finfo(float).eps)
        c = dct(log_spectrum, type=2, norm='ortho')
        mcc[i, :] = c[:36]

    mcc = mcc.T

    plt.imshow(mcc, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('Original Spectrogram')
    plt.xlabel('Time Frames')
    plt.ylabel('MCC Coefficients')

    # Show plots
    plt.tight_layout()
    plt.show()
    plt.show()

    # Step 4: Aperiodicity extraction
    ap = pw.d4c(x, f0, t, fs)  # Aperiodicity extraction
    print(f0.shape, sp.shape, ap.shape)
    return f0, sp, ap, fs

# Process the wav file
f0, sp, ap, fs = process_wav_file("C:/Users/adria/Desktop/test/audio/VCC2SF1/10001.wav")

# Synthesize the audio
# y = pw.synthesize(f0, sp, ap, fs)

# Save the output waveform
# sf.write("reassembled.wav", y, fs)

mat_data = scipy.io.loadmat("C:/Users/adria/Desktop/test/transformed_audio/VCC2SF1/10001.wav.mat")
mcc = mat_data['mcc']
spectral_envelope = idct(mcc, type=2, axis=0, norm='ortho')
spectral_envelope = np.exp(spectral_envelope)
spectral_envelope = spectral_envelope.astype(np.float64)
plt.subplot(1, 2, 2)
plt.imshow(mcc, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar()
plt.title('from mat MCC')
plt.xlabel('Time Frames')
plt.ylabel('MCC Coefficients')
plt.show()