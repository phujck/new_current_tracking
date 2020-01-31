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
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.stats import norm

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


def plot_phi_spectra(U, w, spec, min_spec, max_harm):
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
        plt.ylabel('$\\mathcal{F}\\left(\\frac{a_T}{a}\\Phi_T(t)\\right)$')
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
U = 0.2 * t
U2 = U
degree = 2
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
k = 2.2
# k=1
scalefactor = 1
scalefactor2 = 1
ascale = k
ascale2 = k


def phi_unwravel(phi_track_shift):
    # THIS REMOVES DISCONTINUITIES FROM PHI-THETA. IMPORTANT FOR GETTING EHRENFEST RIGHT!
    for j in range(1, int(len(phi_track_shift))):
        k = phi_track_shift[j] - phi_track_shift[j - 1]

        if k > 1.8 * np.pi:
            phi_track_shift[j:] = phi_track_shift[j:] - 2 * np.pi
        if k < -1.8 * np.pi:
            phi_track_shift[j:] = phi_track_shift[j:] + 2 * np.pi
    return phi_track_shift


"""Turn this to True in order to load tracking files"""
Tracking = True

prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=0.1 * t, t=t, F0=F0, a=a, bc='pbc')
print(prop.field)
parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, 0.1 * t, t, number, delta, field, F0)
newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale)
phi_original = np.load('./data/original/phi' + parameternames)
if Tracking:
    prop_track = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=ascale * a,
                          bc='pbc')
    prop_track2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t, F0=F0, a=ascale2 * a,
                           bc='pbc')
    delta_track = prop_track.freq * delta / prop.freq
    delta_track2 = prop_track2.freq * delta2 / prop.freq
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)

    J_field_track = np.load('./data/tracking/Jfield' + newparameternames) / scalefactor
    phi_track = phi_unwravel(np.load('./data/tracking/phi' + newparameternames))
    t_track = np.linspace(0.0, cycles, len(J_field_track))

    newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
        nx, cycles2, U2, t2, number2, delta2, field, F0, ascale2, scalefactor2)

    J_field_track2 = np.load('./data/tracking/Jfield' + newparameternames2) / scalefactor2
    t_track2 = np.linspace(0.0, cycles, len(J_field_track2))
    phi_track2 = phi_unwravel(np.load('./data/tracking/phi' + newparameternames2))
    times_track = np.linspace(0.0, cycles, len(J_field_track))
    times_track2 = np.linspace(0.0, cycles, len(J_field_track2))

plt.subplot(211)
if Tracking:
    plt.plot(t_track * prop_track.freq / prop.freq, J_field_track, linestyle='dashed',
             label='Tracked Current')
# plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
plt.legend(loc='upper right')

plt.subplot(212)
if Tracking:
    plt.plot(t_track2, J_field_track2, linestyle='dashed',
             label='Tracked current')
plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
plt.show()
#
# plt.plot(times_track, phi_track.real,
#          label='$\\frac{U}{t_0}=$ %.1f scaling $J$' % (prop_track.U))
# plt.plot(times_track, phi_track2.real, label='$\\frac{U}{t_0}=$ %.1f scaling $a$' % (prop_track2.U),
#          linestyle='--')
# lat = prop
# current_time = times_track
# plt.plot(times_track, phi_original, label='reference $\\Phi(t)$')
# plt.legend()
# plt.xlabel('Time [cycles]')
# plt.ylabel('$\\frac{a_T}{ka}\\Phi_T(t)$')
# # plt.yticks(np.arange(-3 * np.pi, 3 * np.pi, 0.5 * np.pi),
# #            [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-3 * np.pi, 3 * np.pi, .5 * np.pi)])
# plt.show()
# method = 'welch'
# min_spec = 15
# max_harm = 60
# gabor = 'fL'
#
# spec = np.zeros((FT_count(len(phi_track)), 3))
# if method == 'welch':
#     w, spec[:, 0] = har_spec.spectrum_welch(phi_track * ascale, delta1)
#     # w2, spec[:,1] = har_spec.spectrum_welch(exact_track, delta_track)
#     w2, spec[:, 1] = har_spec.spectrum_welch(phi_track2 * ascale, delta2)
# else:
#     print('Invalid spectrum method')
# w *= 2. * np.pi / prop.field
# w2 *= 2. * np.pi / prop.field
# plot_phi_spectra([prop_track.U, prop_track2.U], w, spec, min_spec, max_harm)


lat = prop
prefactor = (lat.a * lat.F0 / lat.field)

# print(mean, std)
# gaussian=norm.pdf(t_track/prop_track.freq,5,std)


envelope = np.abs(hilbert(phi_track - phi_original))
mean, std = norm.fit(envelope)


def gauss_fit(time, prefactor, mean, std, prefactor_sin, pre2, omega1, omega2):
    f = prefactor * np.exp((-(time - mean) ** 2 / std))
    # f = prefactor * np.sin(mean*time/cycles)**2
    return f


popt, pcov = curve_fit(gauss_fit, t_track, envelope, maxfev=10000)
best_env = gauss_fit(t_track, *popt)
plt.plot(t_track, best_env, label='best fit')
plt.plot(t_track, envelope, label='$\\Phi_T(t)-\\Phi(t)$ envelope')
# plt.plot(t_track, envelope-best_env, label='$\\Phi_T(t)-\\Phi(t)$ envelope')
# plt.plot(t_track, stretchsin, label='$\\Phi_T(t)-\\Phi(t)$ envelope')

plt.show()
env_func = interp1d(t_track, best_env, fill_value=0, bounds_error=False, kind='cubic')


# p0=np.zeros(3)
# p0[0]=np.amax(phi_track)
# p0[1]=prop_track.field
# p0=np.random.randint(np.pi,size=6)
# chirp = (phi_track-phi_original)/(best_env)
# chirp /= chirp.max()
# chirp = np.arcsin(chirp)
# for j in range(1, len(chirp)):
#     if chirp[j] - chirp[j - 1] > np.pi:
#         chirp[j:] = chirp[j:] - 2 * np.pi
#     if chirp[j] - chirp[j - 1] < -np.pi:
#         chirp[j:] = chirp[j:] + 2 * np.pi

# plt.plot(t_track,chirp)
# plt.show()

def phi_fit(current_time, *params):
    f = len(params)
    prefactors = []
    omegas = []
    phases = []
    chirps = []
    for j in range(f):
        if j < int(f / 4):
            # print(params[j])
            prefactors.append(params[j])
        elif j < int(2 * f / 4):
            omegas.append(params[j])
        elif j < int(3 * f / 4):
            phases.append(params[j])
        else:
            chirps.append(params[j])
    k = 0
    current_time = current_time / prop_track.freq
    for j in range(int(f / 4)):
        # print('success')
        k += prefactors[j] * np.sin((omegas[j] + chirps[j] * current_time) * current_time - phases[j])
        # k += prefactors[j] * np.sin((omegas[j]) * current_time - phases[j])* (np.sin((prop_track.field+chirps[j]*current_time) * current_time / (2 * cycles)))
    # phi = k* (np.sin(prop_track.field * current_time / (2 *cycles)))
    phi = k * env_func(current_time * prop_track.freq)
    # phi = k
    return phi


p0 = np.zeros(degree * 4)
k = 1
for j in range(len(p0)):
    if j < int(len(p0) / 4):
        p0[j] = (np.pi / 2) / int(len(p0) / 4)
        p0[-j] = k * prop_track.field
    elif j < int(2 * len(p0) / 4):
        p0[j] = k * prop_track.field
        k += 2
    else:
        p0[j] = 0
popt, pcov = curve_fit(phi_fit, t_track, (phi_track - phi_original), p0=p0, maxfev=10000)

plt.plot(t_track, (phi_original + phi_fit(t_track, *popt)), label='best fit')
plt.plot(t_track, phi_track, label='$\\Phi_T(t)$')
plt.xlabel('Time [cycles]')
plt.ylabel('$\\Phi_T(t)$')
plt.yticks(np.arange(-0.75 * np.pi, 1 * np.pi, 0.25 * np.pi),
           [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1.25 * np.pi, .25 * np.pi)])

plt.legend()
plt.show()

plt.plot(t_track, phi_original + phi_fit(t_track, *popt), label='best fit')
# plt.plot(t_track,phi_fit(t_track,*p0), label='initial parameters')
plt.plot(t_track, phi_track, label='original')
# plt.plot(t_track,phi_original)
plt.legend()
plt.show()

phi_fitted = phi_original + phi_fit(t_track, *popt)
cutparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)
np.save('./data/fitted/phi' + cutparameternames, phi_fitted)

k = 1
for j in range(len(p0)):
    if j < int(len(p0) / 4):
        print("value of prefactor %s is %.6f of Phi prefactor" % (j, popt[j] / prefactor))
    elif j < int(2 * len(p0) / 4):
        print("value of omega %s is %.2f omega_0" % (j, popt[j] / prop.field))
        p0[j] = k * prop_track.field
    elif j < int(3 * len(p0) / 4):
        print("value of phase %s is %.2f *2pi" % (j, popt[j] / (2 * np.pi)))
    else:
        print("value of chirp %s is %.5e omega_0" % (j, popt[j] / (prop.field)))

#
# for k in [50, 55, 60]:
#     ascale = 1
#     scalefactor = 1 / k
#     U = 7 * t
#     newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor.npy' % (
#         nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor)
#     phi_track = phi_unwravel(np.load('./data/tracking/phi' + newparameternames))
#     plt.subplot(211)
#     plt.plot(t_track, phi_track, linestyle='dashed', label='$J_s=\\frac{J}{%s}$' % (k))
#
#     plt.legend()
#     plt.ylabel('$\\Phi_T(t)$')
#
#     # plt.xlabel('Time [cycles]')
#     # plt.legend(loc='upper right')
#
#     ascale = k
#     scalefactor = 1
#     U = 7 * t
#     newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-scalefactor-%s-degree.npy' % (
#         nx, cycles, U, t, number, delta, field, F0, ascale, scalefactor,degree)
#     phi_track = phi_unwravel(np.load('./data/tracking/phi' + newparameternames))
#
#     plt.subplot(212)
#     plt.plot(t_track, phi_track, linestyle='dashed', label='$a_T=%s a$' % (k))
#
#     plt.legend()
# plt.subplot(211)
# plt.plot(times_track, phi_original, label='reference $\\Phi(t)$')
# plt.subplot(212)
# plt.plot(times_track, phi_original, label='reference $\\Phi(t)$')
# plt.xlabel('Time [cycles]')
# plt.ylabel('$\\Phi_T(t)$')
# plt.show()
