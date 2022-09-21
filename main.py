import numpy as np
from numpy import abs as np_abs
from numpy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import matplotlib as mpl

SAMPLE_RATE = 44100  # Гц
DURATION = 1 # Секунды


def get_y_array(freq, t):
    sin_sig = []
    for i in t:
        sin_sig.append(np.sin((2 * np.pi) * freq * i))  # график сигнала
    return sin_sig


def spectrum_sig(sin_sig1, sin_sig2, sin_sig3, sin_sig4):
    xf = fftfreq(SAMPLE_RATE * DURATION, 1 / SAMPLE_RATE)

    yf1 = fft(sin_sig1)
    yf2 = fft(sin_sig2)
    yf3 = fft(sin_sig3)
    yf4 = fft(sin_sig4)

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].plot(xf, yf1)
    axis[0, 0].set_title("Signal with frequency 1")

    axis[1, 0].plot(xf, yf2)
    axis[1, 0].set_title("Signal with frequency 2")

    axis[0, 1].plot(xf, yf3)
    axis[0, 1].set_title("Signal with frequency 4")

    axis[1, 1].plot(xf, yf4)
    axis[1, 1].set_title("Signal with frequency 8")

    plt.grid(True)
    plt.show()


def graphic_sig():
    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, endpoint=False)

    sig_with_w1 = get_y_array(1, t)
    sig_with_w2 = get_y_array(2, t)
    sig_with_w3 = get_y_array(4, t)
    sig_with_w4 = get_y_array(8, t)

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].plot(t, sig_with_w1)
    axis[0, 0].set_title("Signal with frequency 1")

    axis[1, 0].plot(t, sig_with_w2)
    axis[1, 0].set_title("Signal with frequency 2")

    axis[0, 1].plot(t, sig_with_w3)
    axis[0, 1].set_title("Signal with frequency 4")

    axis[1, 1].plot(t, sig_with_w4)
    axis[1, 1].set_title("Signal with frequency 8")

    plt.grid(True)
    plt.show()

    spectrum_sig(sig_with_w1, sig_with_w2, sig_with_w3, sig_with_w4)


def main():
    mpl.rcParams['font.family'] = 'fantasy'
    mpl.rcParams['font.fantasy'] = 'Comic Sans MS, Arial'

    graphic_sig()


if __name__ == '__main__':
    main()
