import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import scipy.io
from scipy.fftpack import dct
from scipy.fftpack import idct

def normLogTof0(norm_log_f0, mean_log_f0, std_log_f0):
    log_f0 = (norm_log_f0 * std_log_f0) + mean_log_f0
    f0 = np.exp(log_f0)
    f0 = np.reshape(f0, (f0.shape[1],))
    return f0

def plot_mcc_comparison(original_mcc, fake_mcc_interp):
    # Ensure both MCCs have the same shape for comparison
    if original_mcc.shape != fake_mcc_interp.shape:
        print(f"Original MCC shape: {original_mcc.shape}")
        print(f"Fake MCC shape: {fake_mcc_interp.shape}")
        raise ValueError("Original and fake MCCs must have the same shape for comparison!")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_mcc, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('Original MCC')
    plt.xlabel('Time Frames')
    plt.ylabel('MCC Coefficients')

    plt.subplot(1, 2, 2)
    plt.imshow(fake_mcc_interp, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('Fake MCC')
    plt.xlabel('Time Frames')
    plt.ylabel('MCC Coefficients')

    plt.tight_layout()
    plt.show()
    plt.show()

def process_wav_file(wav_path):
    fs, x = wavfile.read(wav_path)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / np.max(np.abs(x))
    _f0, t = pw.dio(x, fs)
    f0 = pw.stonemask(x, _f0, t, fs)
    sp = pw.cheaptrick(x, f0, t, fs)
    sp_t = sp.T
    mcc = np.zeros((sp_t.shape[1], sp_t.shape[0]))#mcc = np.zeros((sp_t.shape[1], 36))
    for i in range(sp_t.shape[1]):
        log_spectrum = np.log(sp_t[:, i] + np.finfo(float).eps)
        c = dct(log_spectrum, type=2, norm='ortho')
        mcc[i, :] = c # mcc[i, :] = c[:36]
    mcc = mcc.T
    sp2 = reverse_mcc_to_spectral(mcc).T # i test here to see why so much lose happens, and if it affects the result - it doesnt
    sp2 = np.ascontiguousarray(sp2)
    # plt.imshow(sp, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
    # plt.colorbar()
    # plt.title('Original Spectrogram')
    # plt.xlabel('Time Frames')
    # plt.ylabel('MCC Coefficients')
    # plt.show()

    # there is a risk that when transposed it is now backwards
    ap = pw.d4c(x, f0, t, fs)
    print(f0.shape, sp.shape, ap.shape)
    return f0, sp, ap, fs


def reverse_mcc_to_spectral(mcc):
    # mcc should be transposed back to time-major (frames x coefficients)
    mcc = mcc.T

    # Initialize an array to hold the recovered log-spectra
    recovered_spectrogram = np.zeros((mcc.shape[1], mcc.shape[0]))

    # Loop over each frame and apply inverse DCT to reconstruct log-spectra
    for i in range(mcc.shape[0]):
        c = mcc[i, :]  # MCC coefficients for frame i
        log_spectrum = idct(c, type=2, norm='ortho', n=recovered_spectrogram.shape[0])
        recovered_spectrogram[:, i] = log_spectrum

    # Exponentiate to get back the original spectral parameters
    spectral_parameters = np.exp(recovered_spectrogram)

    return spectral_parameters

f0, sp, ap, fs = process_wav_file("C:/Users/adria/Desktop/test/audio/VCC2SF1/10001.wav")


y = pw.synthesize(f0, sp, ap, fs)

plt.plot(f0)
plt.title('Pitch Contour (f0)')
plt.xlabel('Time Frames')
plt.ylabel('Pitch (Hz)')
plt.grid(True)
plt.show()

plt.imshow(sp, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
plt.colorbar()
plt.title('Original Spectrogram')
plt.xlabel('Time Frames')
plt.ylabel('MCC Coefficients')
plt.show()

plt.imshow(ap, aspect='auto', origin='lower', cmap='viridis', interpolation='none')
plt.colorbar()
plt.title('Aperiodicity')
plt.xlabel('Time Frames')
plt.ylabel('val')
plt.show()
sf.write("reassembled.wav", y, fs)

# mat_data = scipy.io.loadmat("C:/Users/adria/Desktop/test/resized_audio/VCC2SF1/10001.wav.mat")
# mcc = mat_data['mcc']
# print(mcc.shape)
# print(mat_data['original_mcc_size'])
# print(mat_data['source_parameter']['aperiodicity'][0][0].shape)
# print(mat_data['norm_log_f0'].shape)
# spectral_envelope = idct(mcc, type=2, axis=0, norm='ortho')
# spectral_envelope = np.exp(spectral_envelope)
# spectral_envelope = spectral_envelope.astype(np.float64)
# plt.subplot(1, 2, 2)
# plt.imshow(spectral_envelope, aspect='auto', origin='lower', cmap='viridis')
# plt.colorbar()
# plt.title('from mat MCC')
# plt.xlabel('Time Frames')
# plt.ylabel('MCC Coefficients')
# plt.show()
