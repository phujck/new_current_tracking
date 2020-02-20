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
    'legend.fontsize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.figsize': [2 * 3.375, 2 * 3.375],
    'text.usetex': True
}

plt.rcParams.update(params)
print(plt.rcParams.keys())
# Load parameters and data. 2 suffix is for loading in a different simulation for comparison
number = 5
number2 = 5
nelec = (number, number)
nx = 10
nx2 = 10
ny = 0
t = 0.52
t1 = t
t2 = 0.52
U = 1 * t
U2 = 0 * t
U_track = U
U_track2 = U2
delta = 0.02
delta1 = delta
delta2 = 0.02
cycles = 10
cycles2 = 10
# field= 32.9
field = 32.9
field2 = 32.9
F0 = 10
a = 4
scalefactor = 1
scalefactor2 = 1
ascale = 10
ascale2 = 1
degree = 3
"""Turn this to True in order to load tracking files"""
Tracking = True

prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
prop2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t2, F0=F0, a=a, bc='pbc')
print(prop.field)
print(prop2.field)
if Tracking:
    """Notice the U's have been swapped on the presumption of tracking the _other_ system."""
    prop_track = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U_track, t=t, F0=F0, a=ascale * a,
                          bc='pbc')
    prop_track2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U_track2, t=t, F0=F0, a=ascale2 * a,
                           bc='pbc')
    delta_track = prop_track.freq * delta / prop.freq
    delta_track2 = prop_track2.freq * delta2 / prop.freq
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor-%s-degree.npy' % (
        nx, cycles, U_track, t, number, delta, field, F0, ascale, scalefactor, degree)

    J_field_track = np.load('./data/fitted/Jfield' + newparameternames) / scalefactor
    t_track = np.linspace(0.0, cycles, len(J_field_track))
    # length= len(J_field_track)/10
    # J_field_track=J_field_track[int(1.4*length):int(9*length)]
    # t_track = np.linspace(1.4, 9, len(J_field_track))
    phi_track = np.load('./data/fitted/phi' + newparameternames)

    newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor-%s-degree.npy' % (
        nx, cycles, U_track2, t, number, delta, field, F0, ascale2, scalefactor, degree)

    J_field_track2 = np.load('./data/fitted/Jfield' + newparameternames2) / scalefactor
    t_track2 = np.linspace(0.0, cycles, len(J_field_track2))

    # length2 = len(J_field_track2) / 10
    # J_field_track2 = J_field_track2[int(1.4 * length2):int(8.5 * length2)]
    # t_track2 = np.linspace(1.4, 8.5, len(J_field_track2))
    phi_track2 = np.load('./data/fitted/phi' + newparameternames2)

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

plt.plot(times, J_field)
plt.plot(times, (nx / nx2) * J_field2)
plt.show()
method = 'welch'
min_spec = 15
max_harm = 60
gabor = 'fL'

zeroparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, 0.0, t, number, delta, field, F0)
# zeroparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
#     nx, cycles, 1 * t, t, number, delta, field, F0)
ref_J = np.load('./data/original/Jfield' + zeroparameternames) / scalefactor
times = np.linspace(0.0, cycles, len(ref_J))

plt.figure(figsize=(16 * 1, 9 * 1.4))
plt.subplot(211)
plt.plot(times, J_field, label='$J(t)$')
if Tracking:
    plt.plot(t_track * prop_track.freq / prop.freq, J_field_track,
             label='$\\bar{J}_T(t)$', color='red')
# plt.xlabel('Time [cycles]')
plt.plot(times, ref_J, linestyle='dashed', label='Target', color='black')
plt.ylabel('$J(t)$')
plt.xlabel('Time [cycles]')
plt.legend(loc='upper right')

plt.subplot(212)

exact = np.gradient(J_field_track, delta_track)
w, spec = har_spec.spectrum_welch(exact, delta1)
w *= 2. * np.pi / prop.field
# spec[int(len(spec) / 100):] = spec[int(len(spec) / 100):] / 5
plt.semilogy(w, spec, label='$\\bar{J}_T(t)$', color='red')

exact = np.gradient(ref_J, delta)
w, spec = har_spec.spectrum_welch(exact, delta1)
w *= 2. * np.pi / prop.field
plt.semilogy(w, spec, linestyle='dashed', label='Target', color='black')

axes = plt.gca()
axes.set_xlim([0, max_harm])
axes.set_ylim([10 ** (-min_spec), spec.max()])

exact = np.gradient(J_field, delta)
w, spec = har_spec.spectrum_welch(exact, delta1)
w *= 2. * np.pi / prop.field
plt.semilogy(w, spec, label='$J(t)$')

xlines = [2 * i - 1 for i in range(1, 6)]

for xc in xlines:
    plt.axvline(x=xc, color='black', linestyle='dashed')
plt.xlabel('Harmonic Order')
plt.ylabel('HHG spectra')
# plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(16, 9))
plt.subplot(211)
plt.plot(times, J_field, label='$J(t)$')
if Tracking:
    plt.plot(t_track * prop_track.freq / prop.freq, J_field_track, linestyle='dashed',
             label='$\\bar{J}_T(t)$', color='red')
# plt.xlabel('Time [cycles]')
plt.plot(times, ref_J, linestyle='-.', label='Target', color='black')
plt.ylabel('$J(t)$')
plt.legend(loc='upper right')
plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)

plt.subplot(212)
plt.plot(times2, J_field2, label='$J(t)$')
if Tracking:
    plt.plot(t_track2, J_field_track2, linestyle='dashed',
             label='Tracked current', color='red')
plt.plot(times, ref_J, linestyle='-.', label='Target', color='black')
plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
plt.annotate('b)', xy=(0.3, np.max(J_field2) - 0.05), fontsize=25)
plt.show()
norm1 = np.linalg.norm(J_field2 - ref_J, ord=2)
norm2 = np.linalg.norm(ref_J - J_field_track2, ord=2)
print(norm1)
print(norm2)

exact_track = np.gradient(J_field_track, delta_track)
exact_track2 = np.gradient(J_field_track2, delta_track2)

exact = np.gradient(J_field, delta)
# exact2 = np.gradient(J_field2, delta2)
times = np.linspace(0.0, cycles, len(J_field))
times2 = np.linspace(0.0, cycles, len(J_field2))

exact2 = np.gradient(J_field_track, delta_track)
# exact2= np.gradient(J_field_track2, delta_track2)
"""Plotting spectrum."""

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
#

ascales = [1.5, 5, 10]
Us = [0.1, 0.5, 1]
normsU = []
degree = 6
for j in range(len(ascales)):
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor-%s-degree.npy' % (
        nx, cycles, Us[j] * t, t, number, delta, field, F0, ascales[j], scalefactor, degree)
    J_field_track = np.load('./data/fitted/Jfield' + newparameternames) / scalefactor
    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
        nx, cycles, Us[j] * t, t, number, delta, field, F0)
    J_field = np.load('./data/original/Jfield' + parameternames)
    # norms.append(np.linalg.norm(ref_J-J_field_track,ord=2)/np.linalg.norm(ref_J-J_field,ord=2))
    normsU.append(np.linalg.norm(ref_J - J_field_track, ord=2))
    exact = np.gradient(J_field_track, delta_track)
    w, spec = har_spec.spectrum_welch(exact, delta1)
    w *= 2. * np.pi / prop.field
    plt.semilogy(w, spec, label='$\\frac{U}{t_0}=$%s' % (Us[j]))

exact = np.gradient(ref_J, delta)
w, spec = har_spec.spectrum_welch(exact, delta1)
w *= 2. * np.pi / prop.field
plt.semilogy(w, spec, linestyle='dashed', label='Target Spectra', color='black')
axes = plt.gca()
axes.set_xlim([0, max_harm])
axes.set_ylim([10 ** (-min_spec), spec.max()])
xlines = [2 * i - 1 for i in range(1, 6)]

for xc in xlines:
    plt.axvline(x=xc, color='black', linestyle='dashed')
plt.xlabel('Harmonic Order')
plt.ylabel('HHG spectra')
plt.legend(loc='upper right')
plt.show()

plt.plot(Us, normsU)
plt.show()

# deltas=[0.02,0.005]
#
#
# for j in range(len(deltas)):
#     ascale=1.5
#     U=0.1*t
#     newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor-%s-degree.npy' % (
#         nx, cycles, U, t, number, deltas[j], field, F0, ascale, scalefactor,degree)
#     J_field_track = np.load('./data/fitted/Jfield' + newparameternames) / scalefactor
#     delta_track=deltas[j]
#     exact=np.gradient(J_field_track,delta_track)
#     w, spec = har_spec.spectrum_welch(exact, delta_track)
#     w *= 2. * np.pi / prop.field
#     plt.semilogy(w, spec, label='$\\Delta=$%s' % (deltas[j]))
#
# exact = np.gradient(ref_J, delta)
# w, spec = har_spec.spectrum_welch(exact, delta1)
# w *= 2. * np.pi / prop.field
# plt.semilogy(w, spec, linestyle='dashed', label='Target Spectra',color='black' )
# axes = plt.gca()
# axes.set_xlim([0, max_harm])
# axes.set_ylim([10 ** (-min_spec), spec.max()])
# xlines = [2 * i - 1 for i in range(1, 6)]
#
# for xc in xlines:
#     plt.axvline(x=xc, color='black', linestyle='dashed')
# plt.xlabel('Harmonic Order')
# plt.ylabel('HHG spectra')
# plt.legend(loc='upper right')
# plt.show()

# ascales=[1.1,2.5,3.3]
# Us=[0.1,0.5,1]
#
# for j in range(len(ascales)):
#     newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
#         nx, cycles, Us[j]*t, t, number, delta, field, F0, ascales[j], scalefactor)
#     J_field_track = np.load('./data/fitted/Jfield' + newparameternames) / scalefactor
#
#     exact=np.gradient(J_field_track,delta_track)
#     w, spec = har_spec.spectrum_welch(exact, delta1)
#     w *= 2. * np.pi / prop.field
#     plt.semilogy(w, spec, label='$\\frac{U}{t_0}=$%s' % (Us[j]))
#
# exact = np.gradient(J_field, delta)
# w, spec = har_spec.spectrum_welch(exact, delta1)
# w *= 2. * np.pi / prop.field
# plt.semilogy(w, spec, linestyle='dashed', label='Target Spectra',color='black' )
# axes = plt.gca()
# axes.set_xlim([0, max_harm])
# axes.set_ylim([10 ** (-min_spec), spec.max()])
# xlines = [2 * i - 1 for i in range(1, 6)]
#
# for xc in xlines:
#     plt.axvline(x=xc, color='black', linestyle='dashed')
# plt.xlabel('Harmonic Order')
# plt.ylabel('HHG spectra')
# plt.legend(loc='upper right')
# plt.show()
#

degrees = [1, 2, 3, 4, 6, 8, 10]
norms = []

for j in range(len(degrees)):
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor-%s-degree.npy' % (
        nx, cycles, 1 * t, t, number, delta, field, F0, ascale, scalefactor, degrees[j])
    J_field_track = np.load('./data/fitted/Jfield' + newparameternames) / scalefactor
    norms.append(np.linalg.norm(ref_J - J_field_track, ord=1))
    if degrees[j] == 2:
        lowdegree = J_field_track
    if degrees[j] == 10:
        highdegree = J_field_track
    exact = np.gradient(J_field_track, delta_track)
    w, spec = har_spec.spectrum_welch(exact, delta1)
    w *= 2. * np.pi / prop.field
    plt.semilogy(w, spec, label='Degree =%s' % (degrees[j]))

exact = np.gradient(ref_J, delta)
w, spec = har_spec.spectrum_welch(exact, delta1)
w *= 2. * np.pi / prop.field
plt.semilogy(w, spec, linestyle='dashed', label='Target Spectra', color='black')
axes = plt.gca()
axes.set_xlim([0, max_harm])
axes.set_ylim([10 ** (-min_spec), spec.max()])
xlines = [2 * i - 1 for i in range(1, 6)]

for xc in xlines:
    plt.axvline(x=xc, color='black', linestyle='dashed')
plt.xlabel('Harmonic Order')
plt.ylabel('HHG spectra')
plt.legend(loc='upper right')
plt.title('$\\frac{U}{t_0}=0.1$')
plt.show()
plt.plot(degrees, norms)
plt.show()
# for j in range(len(degrees)):
#     ascale=3.3
#     newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor-%s-degree.npy' % (
#         nx, cycles, 1*t, t, number, delta, field, F0, ascale, scalefactor,degrees[j])
#     J_field_track = np.load('./data/fitted/Jfield' + newparameternames) / scalefactor
#     if degrees[j]==2:
#         lowdegree=J_field_track
#     if degrees[j]==20:
#         highdegree=J_field_track
#     exact=np.gradient(J_field_track,delta_track)
#     w, spec = har_spec.spectrum_welch(exact, delta1)
#     w *= 2. * np.pi / prop.field
#     plt.semilogy(w, spec, label='Degree =%s' % (degrees[j]))
#
# exact = np.gradient(J_field, delta)
# w, spec = har_spec.spectrum_welch(exact, delta1)
# w *= 2. * np.pi / prop.field
# plt.semilogy(w, spec, linestyle='dashed', label='Target Spectra',color='black' )
# axes = plt.gca()
# axes.set_xlim([0, max_harm])
# axes.set_ylim([10 ** (-min_spec), spec.max()])
# xlines = [2 * i - 1 for i in range(1, 6)]
#
# for xc in xlines:
#     plt.axvline(x=xc, color='black', linestyle='dashed')
# plt.xlabel('Harmonic Order')
# plt.ylabel('HHG spectra')
# plt.legend(loc='upper right')
# plt.title('$\\frac{U}{t_0}=1$')
# plt.show()
#

J_field = ref_J
J_field2 = ref_J
plt.subplot(211)
plt.plot(times, J_field, label='$\\frac{U}{t_0}=0$')
if Tracking:
    plt.plot(t_track * prop_track.freq / prop.freq, lowdegree, linestyle='dashed',
             label='Tracked Current')
# plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
# plt.legend(loc='upper right')
plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)

plt.subplot(212)
plt.plot(times2, J_field2, label='\\frac{U}{t_0}=')
if Tracking:
    plt.plot(t_track2, highdegree, linestyle='dashed',
             label='Tracked current')
plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
plt.annotate('b)', xy=(0.3, np.max(J_field2) - 0.05), fontsize=25)
plt.show()
