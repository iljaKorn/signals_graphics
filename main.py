from numpy import array, arange, abs as np_abs
from numpy.fft import rfft, rfftfreq
from math import sin, pi
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_y_array(A, w, N):
    sin_sig = array([A * sin(w * t) for t in arange(0, N)])  # график сигнала
    return sin_sig


def spectr_sig(N, sin_sig):
    spectr_sin = rfft(sin_sig)
    # вычисляем дискретное действительное rfft  преобразование Фурье
    plt.plot(rfftfreq(N), np_abs(spectr_sin) / N)  # график спектра
    plt.xlabel('Частота, Гц')
    plt.ylabel('Амплитуда  сигнала')
    plt.title('Спектр синусоидального сигнала')
    plt.grid(True)
    plt.show()


def grapfic_sig(A, N):
    sig_with_w1 = get_y_array(A, 2. * pi * 1, N)
    sig_with_w2 = get_y_array(A, 2. * pi * 2, N)
    sig_with_w3 = get_y_array(A, 2. * pi * 4, N)
    sig_with_w4 = get_y_array(A, 2. * pi * 8, N)

    figure, axis = plt.subplots(2, 2)

    t = arange(0, N, 0.01)

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


def main():
    mpl.rcParams['font.family'] = 'fantasy'
    mpl.rcParams['font.fantasy'] = 'Comic Sans MS, Arial'

    N = 10  # длителльность сигнала
    A = 1.0  # амплитуда сигнала

    # сгенерируем чистый синусоидальный сигнал с частотой F длиной N

    sig_with_w1 = get_y_array(A, 2. * pi * 1, N)
    spectr_sig(N, sig_with_w1)
    # grapfic_sig(A, N)


if __name__ == '__main__':
    main()
