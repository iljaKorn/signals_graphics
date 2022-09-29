import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

SAMPLE_RATE = 200  # Гц
DURATION = 5  # Секунды


def get_y_array(freq, t):
    sin_sig = []
    for i in t:
        sin_sig.append(np.sin((2 * np.pi) * freq * i))  # график сигнала
    return sin_sig


def spectrum_sig(sin_sig1, sin_sig2, sin_sig3, sin_sig4):
    yf1 = rfft(sin_sig1)  # можно использовать fft, чтобы было с симметрией
    yf2 = rfft(sin_sig2)
    yf3 = rfft(sin_sig3)
    yf4 = rfft(sin_sig4)

    xf = rfftfreq(SAMPLE_RATE * DURATION, 1 / SAMPLE_RATE)

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].plot(xf, np.abs(yf1))
    axis[0, 0].set_title("Signal with frequency 1")
    axis[0, 0].grid()

    axis[1, 0].plot(xf, np.abs(yf2))
    axis[1, 0].set_title("Signal with frequency 2")
    axis[1, 0].grid()

    axis[0, 1].plot(xf, np.abs(yf3))
    axis[0, 1].set_title("Signal with frequency 4")
    axis[0, 1].grid()

    axis[1, 1].plot(xf, np.abs(yf4))
    axis[1, 1].set_title("Signal with frequency 8")
    axis[1, 1].grid()

    plt.show()


def graphic_harmonic_sig(t, sig_with_w1, sig_with_w2, sig_with_w3, sig_with_w4):
    figure, axis = plt.subplots(2, 2)

    axis[0, 0].plot(t, sig_with_w1)
    axis[0, 0].set_title("Signal with frequency 1")
    axis[0, 0].grid()

    axis[1, 0].plot(t, sig_with_w2)
    axis[1, 0].set_title("Signal with frequency 2")
    axis[1, 0].grid()

    axis[0, 1].plot(t, sig_with_w3)
    axis[0, 1].set_title("Signal with frequency 4")
    axis[0, 1].grid()

    axis[1, 1].plot(t, sig_with_w4)
    axis[1, 1].set_title("Signal with frequency 8")
    axis[1, 1].grid()

    plt.show()

    spectrum_sig(sig_with_w1, sig_with_w2, sig_with_w3, sig_with_w4)


def get_digital_sig(sig):
    new_sig = np.sign(sig)
    for i in range(len(new_sig)):
        if new_sig[i] < 0:
            new_sig[i] = 0
    return new_sig


def graphic_digital_sig(t, sig_with_w1, sig_with_w2, sig_with_w3, sig_with_w4):
    sig_w1 = get_digital_sig(sig_with_w1)
    sig_w2 = get_digital_sig(sig_with_w2)
    sig_w3 = get_digital_sig(sig_with_w3)
    sig_w4 = get_digital_sig(sig_with_w4)

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].plot(t, sig_w1)
    axis[0, 0].set_title("Signal with frequency 1")
    axis[0, 0].grid()

    axis[1, 0].plot(t, sig_w2)
    axis[1, 0].set_title("Signal with frequency 2")
    axis[1, 0].grid()

    axis[0, 1].plot(t, sig_w3)
    axis[0, 1].set_title("Signal with frequency 4")
    axis[0, 1].grid()

    axis[1, 1].plot(t, sig_w4)
    axis[1, 1].set_title("Signal with frequency 8")
    axis[1, 1].grid()

    plt.show()

    spectrum_sig(sig_w1, sig_w2, sig_w3, sig_w4)


def main():
    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, endpoint=False)

    sig_with_w1 = get_y_array(1, t)
    sig_with_w2 = get_y_array(2, t)
    sig_with_w3 = get_y_array(4, t)
    sig_with_w4 = get_y_array(8, t)

    graphic_harmonic_sig(t, sig_with_w1, sig_with_w2, sig_with_w3, sig_with_w4)

    graphic_digital_sig(t, sig_with_w1, sig_with_w2, sig_with_w3, sig_with_w4)


if __name__ == '__main__':
    main()
