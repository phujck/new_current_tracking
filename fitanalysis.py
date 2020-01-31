import numpy as np
import matplotlib.pyplot as plt
import definition as hams
import numpy.ma as ma
from matplotlib import cm as cm
from scipy.signal import blackman
from scipy.signal import stft
import harmonic as har_spec
from scipy.signal import savgol_filter
from scipy import linalg

from scipy.interpolate import interp1d


def iFT(A):
    """
    Inverse Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    A = np.array(A)
    minus_one = (-1) ** np.arange(A.size)
    result = np.fft.ifft(minus_one * A)
    result *= minus_one
    result *= np.exp(1j * np.pi * A.size / 2)
    return result


def FT(A):
    """
    Fourier transform
    :param A:  1D numpy.array
    :return:
    """
    # test
    # print(A.size)
    A = np.array(A)
    k = A.size
    # A = np.pad(A, (0, 4 * k), 'constant')
    minus_one = (-1) ** np.arange(A.size)
    # result = np.fft.fft(minus_one * A)
    result = np.fft.fft(minus_one * A)
    # minus_one = (-1) ** np.arange(result.size)
    result *= minus_one
    result *= np.exp(-1j * np.pi * A.size / 2)
    # print(result.size)
    return result


def smoothing(A, b=1, c=5, d=0):
    if b == 1:
        b = int(A.size / 50)
    if b % 2 == 0:
        b = b + 1
    j = savgol_filter(A, b, c, deriv=d)
    return j


def current(sys, phi, neighbour):
    conjugator = np.exp(-1j * phi) * neighbour
    c = sys.a * sys.t * 2 * np.imag(conjugator)
    return c


def plot_spectra(U, w, spec, min_spec, max_harm):
    # spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        plt.semilogy(w, spec[:, i], label='$\\frac{U}{t_0}=$ %.1f' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([10 ** (-15), spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()


def plot_spectra_switch(U, w, spec, min_spec, max_harm):
    spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        plt.plot(w, spec[:, i], label='%s' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([-min_spec, spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()


def plot_spectra_track(U, w, spec, min_spec, max_harm):
    # spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        print(i)
        print(i % 2)
        if i < 2:
            plt.semilogy(w, spec[:, i], label='%s' % (j))
        else:
            plt.semilogy(w, spec[:, i], linestyle='dashed', label='%s' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([10 ** (-min_spec), spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()


def FT_count(N):
    if N % 2 == 0:
        return int(1 + N / 2)
    else:
        return int((N + 1) / 2)


# These are Parameters I'm using
# number=2
# nelec = (number, number)
# nx = 4®
# ny = 0
# t = 0.191
# U = 0.1 * t
# delta = 2
# cycles = 10
params = {
    'axes.labelsize': 30,
    # 'legend.fontsize': 28,
    'legend.fontsize': 23,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [2 * 3.375, 2 * 3.375],
    'text.usetex': True
}

plt.rcParams.update(params)
print(plt.rcParams.keys())
# Load parameters and data. 2 suffix is for loading in a different simulation for comparison
number = 3
number2 = 3
nelec = (number, number)
nx = 6
nx2 = 6
ny = 0
t = 0.52
t1 = t
t2 = 0.52
U = 0 * t
U2 = 0 * t
delta = 0.05
delta1 = delta
delta2 = 0.05
cycles = 10
cycles2 = 10
# field= 32.9
field = 32.9
field2 = 32.9
F0 = 10
a = 4
scalefactor = 1
scalefactor2 = 1
ascale = 1
ascale2 = 1

"""Turn this to True in order to load tracking files"""
fitting = False

prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
prop2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t2, F0=F0, a=a, bc='pbc')
print(prop.field)
print(prop2.field)
if fitting:
    """Notice the U's have been swapped on the presumption of tracking the _other_ system."""
    prop_track = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t, F0=F0, a=ascale * a,
                          bc='pbc')
    prop_track2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=ascale2 * a,
                           bc='pbc')
    delta_track = prop_track.freq * delta / prop.freq
    delta_track2 = prop_track2.freq * delta2 / prop.freq
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)

    J_field_track = np.load('./data/tracking/Jfield' + newparameternames) / scalefactor
    phi_track = np.load('./data/tracking/phi' + newparameternames)
    neighbour_track = np.load('./data/tracking/neighbour' + newparameternames)
    two_body_track = np.load('./data/tracking/twobody' + newparameternames)
    t_track = np.linspace(0.0, cycles, len(J_field_track))
    D_track = np.load('./data/tracking/double' + newparameternames)

# load files
parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)
newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale)
J_field = np.load('./data/original/Jfield' + parameternames)
phi_original = np.load('./data/original/phi' + parameternames)
phi_reconstruct = np.load('./data/original/phirecon' + parameternames)
neighbour = np.load('./data/original/neighbour' + parameternames)

two_body = np.load('./data/original/twobody' + parameternames)
D = np.load('./data/original/double' + parameternames)

parameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0)
newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0, ascale2)
J_field2 = np.load('./data/original/Jfield' + parameternames2)
two_body2 = np.load('./data/original/twobody' + parameternames2)
neighbour2 = np.load('./data/original/neighbour' + parameternames2)
phi_original2 = np.load('./data/original/phi' + parameternames2)
D2 = np.load('./data/original/double' + parameternames2)

times = np.linspace(0.0, cycles, len(J_field))
times2 = np.linspace(0.0, cycles, len(J_field2))

rank_test = np.load('./data/original/ranktestcurrent.npy')
plt.plot(rank_test)
plt.title('rank test')
plt.show()
"""Plot currents"""

plt.plot([6, 8, 10], [82, 245, 1988])
plt.show()

plt.subplot(211)
plt.plot(times, J_field, label='$\\frac{U}{t_0}=0$')
if Tracking:
    plt.plot(t_track * prop_track.freq / prop.freq, J_field_track, linestyle='dashed',
             label='Tracked Current')
# plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
# plt.legend(loc='upper right')
plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)

plt.subplot(212)
plt.plot(times2, J_field2, label='\\frac{U}{t_0}=7')
if Tracking:
    plt.plot(t_track2, J_field_track2, linestyle='dashed',
             label='Tracked current')
plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
plt.annotate('b)', xy=(0.3, np.max(J_field2) - 0.05), fontsize=25)
plt.show()

diff_track = phi_track - theta
J_grad_track = -2. * prop_track.a * prop_track.t * np.gradient(phi_track, delta_track) * np.abs(
    neighbour_track) * np.cos(diff_track)
exact_track = np.gradient(J_field_track, delta_track)
exact_track2 = np.gradient(J_field_track2, delta_track2)

print(max(exact) / max(exact_track))
# eq 32 is the Ehrenfest theorem from direct differentiation of J expectation
eq32_track = (J_grad_track - prop_track.a * prop_track.t * prop_track.U * extra_track) / scalefactor
print(max(eq32) / max(eq32_track))
# eq33 works in terms of the nearest neighbour expectation. It should give the same result as eq32,
# but it's vital that the angles have their discontinuities removed first.

eq33_track = (J_grad_track + 2. * prop_track.a * prop_track.t * (
        np.gradient(theta, delta_track) * np.abs(neighbour_track) * np.cos(
    diff_track) - np.gradient(
    np.abs(neighbour_track), delta_track) * np.sin(diff_track))) / scalefactor
plt.subplot(311)
# plt.plot(t, eq32, label='original')
plt.plot(t_track, exact_track,
         label='original')
# plt.plot(t_track[:-5], eq33_track[:-5], linestyle='-.',
#          label='analytical')
plt.plot(t_track[:-5], eq32_track[:-5], linestyle='dashed',
         label='tracked')
plt.ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
plt.legend()  #
prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
prop2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t2, F0=F0, a=a, bc='pbc')
print(prop.field)
print(prop2.field)
if Tracking:
    """Notice the U's have been swapped on the presumption of tracking the _other_ system."""
    prop_track = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t, F0=F0, a=ascale * a,
                          bc='pbc')
    prop_track2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=ascale2 * a,
                           bc='pbc')
    delta_track = prop_track.freq * delta / prop.freq
    delta_track2 = prop_track2.freq * delta2 / prop.freq
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)

    J_field_track = np.load('./data/tracking/Jfield' + newparameternames) / scalefactor
    phi_track = np.load('./data/tracking/phi' + newparameternames)
    neighbour_track = np.load('./data/tracking/neighbour' + newparameternames)
    two_body_track = np.load('./data/tracking/twobody' + newparameternames)
    t_track = np.linspace(0.0, cycles, len(J_field_track))
    D_track = np.load('./data/tracking/double' + newparameternames)

    newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles2, U2, t2, number2, delta2, field, F0, ascale2, scalefactor2)

    J_field_track2 = np.load('./data/tracking/Jfield' + newparameternames2) / scalefactor2
    phi_track2 = np.load('./data/tracking/phi' + newparameternames2)
    neighbour_track2 = np.load('./data/tracking/neighbour' + newparameternames2)
    two_body_track2 = np.load('./data/tracking/twobody' + newparameternames2)
    t_track2 = np.linspace(0.0, cycles, len(J_field_track2))
    D_track2 = np.load('./data/tracking/double' + newparameternames2)

    times_track = np.linspace(0.0, cycles, len(J_field_track))
    times_track2 = np.linspace(0.0, cycles, len(J_field_track2))

# load files
parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)
newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale)
J_field = np.load('./data/original/Jfield' + parameternames)
phi_original = np.load('./data/original/phi' + parameternames)
phi_reconstruct = np.load('./data/original/phirecon' + parameternames)
neighbour = np.load('./data/original/neighbour' + parameternames)

two_body = np.load('./data/original/twobody' + parameternames)
D = np.load('./data/original/double' + parameternames)

parameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0)
newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0, ascale2)
J_field2 = np.load('./data/original/Jfield' + parameternames2)
two_body2 = np.load('./data/original/twobody' + parameternames2)
neighbour2 = np.load('./data/original/neighbour' + parameternames2)
phi_original2 = np.load('./data/original/phi' + parameternames2)
D2 = np.load('./data/original/double' + parameternames2)

times = np.linspace(0.0, cycles, len(J_field))
times2 = np.linspace(0.0, cycles, len(J_field2))

plt.subplot(312)
plt.plot(times, np.abs(neighbour), label='original')
plt.plot(t_track, np.abs(neighbour_track), linestyle='dashed', label='tracked')
plt.ylabel('$R(\psi)$')

plt.subplot(313)
plt.plot(times, np.abs(two_body), label='original')
plt.plot(t_track, np.abs(two_body_track), linestyle='dashed')
plt.ylabel('$C(\psi)$')
plt.xlabel('Time [cycles]')

plt.show()

"""Plotting spectrum."""
method = 'welch'
min_spec = 15
max_harm = 60
gabor = 'fL'

spec = np.zeros((FT_count(len(J_field)), 2))
if method == 'welch':
    w, spec[:, 0] = har_spec.spectrum_welch(exact, delta1)
    # w2, spec[:,1] = har_spec.spectrum_welch(exact_track, delta_track)
    w2, spec[:, 1] = har_spec.spectrum_welch(exact2, delta2)
elif method == 'hann':
    w, spec[:, 0] = har_spec.spectrum_hanning(exact, delta1)
    w2, spec[:, 0] = har_spec.spectrum_hanning(exact2, delta2)
elif method == 'none':
    w, spec[:, 0] = har_spec.spectrum(exact, delta1)
    w2, spec[:, 0] = har_spec.spectrum(exact2, delta2)
else:
    print('Invalid spectrum method')
w *= 2. * np.pi / prop.field
w2 *= 2. * np.pi / prop.field
plot_spectra([prop.U, prop2.U], w, spec, min_spec, max_harm)
