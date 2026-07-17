from scipy.signal import butter, filtfilt


# Bidirectional Butterworth low-pass filter:
def butterworth_lowpass(sig, order, normal_cutoff, pad_len):
    b, a = butter(order, normal_cutoff)
    sig_filtered = filtfilt(b, a, sig, axis=1, padlen=pad_len)
    return sig_filtered
