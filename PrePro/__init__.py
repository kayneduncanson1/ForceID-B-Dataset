from scipy.signal import butter, filtfilt


# Bidirectional Butterworth low-pass filter:
def butterworth_lowpass(sig, order, normal_cutoff, pad_len):
    b, a = butter(order, normal_cutoff)
    sig_filtered = filtfilt(b, a, sig, axis=1, padlen=pad_len)
    return sig_filtered


def normalise_grf_by_body_weight(signals, masses, gravity):

    """Return signals with Fx/Fy/Fz in % body weight and COP left in metres.

    Parameters
    ----------
    signals : array, shape (n_signals, 5, 100)
        The second axis represents channels Fx, Fy, Fz, Cx and Cy (respectively).
    masses : 1-D array, length n_signals
        Participant body mass without carried objects (unit: kg).
    """

    values = signals.copy()
    values[:, :3, :] = values[:, :3, :] / (masses[:, None, None] * gravity) * 100.0

    return values
