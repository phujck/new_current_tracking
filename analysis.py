import numpy as np
import matplotlib.pyplot as plt
import definition as hams
import numpy.ma as ma
from matplotlib import cm as cm
from scipy.signal import blackman
from scipy.signal import stft
import harmonic as har_spec
from scipy.signal import savgol_filter
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


Tracking = False
Track_Branch = False


def plot_spectra(U, w, spec, min_spec, max_harm):
    # spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        plt.semilogy(w, spec[:, i], label='$\\frac{U}{t_0}=$ %.1f' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([10**(-15), spec.max()])
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
        axes.set_ylim([10**(-min_spec), spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()


def plot_spectrogram(t, w, spec, min_spec=11, max_harm=60):
    w = w[w <= max_harm]
    t, w = np.meshgrid(t, w)
    spec = np.log10(spec[:len(w)])
    specn = ma.masked_where(spec < -min_spec, spec)
    cm.RdYlBu_r.set_bad(color='white', alpha=None)
    plt.pcolormesh(t, w, specn, cmap='RdYlBu_r')
    plt.colorbar()
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Harmonic Order')
    plt.title('Time-Resolved Emission')
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
    'figure.figsize': [2*3.375, 2*3.375],
    'text.usetex': True
}

plt.rcParams.update(params)
print(plt.rcParams.keys())
# Load parameters and data. 2 suffix is for loading in a different simulation for comparison
number = 5
number2 = number
nelec = (number, number)
nx = 10
nx2 = nx
ny = 0
t = 0.52
t1 = t
t2 = 0.52
U = 0 * t
U2 = 7 * t
delta = 0.05
delta2 = 0.05
cycles = 10
cycles2 = 10
# field= 32.9
field = 32.9
field2 = 32.9
F0 = 10
a = 4
scalefactor = 1/100
scalefactor2 = 1
ascale = 1
ascale2 = 1
Jscale = 1
cutoff = 60
cutoff2 = 10

Tracking = True
CutSpec = True
Switch = False

prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
prop2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t2, F0=F0, a=a, bc='pbc')
print(prop.field)
print(prop2.field)
if Tracking:
    prop_track = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t, F0=F0, a=ascale * a,
                          bc='pbc')
    prop_track2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=ascale2 * a,
                           bc='pbc')
    delta_track = prop_track.freq * delta / prop.freq
    delta_track2 = prop_track2.freq * delta2 / prop.freq
if Switch:
    prop_switch = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')

# factor=prop.factor
delta1 = delta
delta_switch = 0.05

# load files
parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)
newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale)
J_field = np.load('./data/original/Jfield' + parameternames)
phi_original = np.load('./data/original/phi' + parameternames)
phi_reconstruct = np.load('./data/original/phirecon' + parameternames)
neighbour = np.load('./data/original/neighbour' + parameternames)
# neighbour_check = np.load('./data/original/neighbour_check' + parameternames)
# energy = np.load('./data/original/energy' + parameternames)
# doublon_energy = np.load('./data/original/doublonenergy' + parameternames)
# doublon_energy_L = np.load('./data/original/doublonenergy2' + parameternames)
# singlon_energy = np.load('./data/original/singlonenergy' + parameternames)

two_body = np.load('./data/original/twobody' + parameternames)
# two_body_old=np.load('./data/original/twobodyold'+parameternames)
D = np.load('./data/original/double' + parameternames)

error = np.load('./data/original/error' + parameternames)

parameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0)
newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
    nx2, cycles2, U2, t2, number2, delta2, field2, F0, ascale2)
J_field2 = np.load('./data/original/Jfield' + parameternames2)
two_body2 = np.load('./data/original/twobody' + parameternames2)
neighbour2 = np.load('./data/original/neighbour' + parameternames2)
phi_original2 = np.load('./data/original/phi' + parameternames2)
# energy2 = np.load('./data/original/energy' + parameternames2)
# doublon_energy2 = np.load('./data/original/doublonenergy' + parameternames2)
# doublon_energy_L2 = np.load('./data/original/doublonenergy2' + parameternames2)
# singlon_energy2 = np.load('./data/original/singlonenergy' + parameternames2)
# error2 = np.load('./data/original/error' + parameternames2)
D2 = np.load('./data/original/double' + parameternames2)

if Tracking:
    # parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    # nx, cycles, U, t, number, delta, field, F0)
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
        nx, cycles, U, t, number, delta, field, F0, ascale)

    J_field_track = np.load('./data/tracking/Jfield' + newparameternames) * Jscale / scalefactor
    phi_track = np.load('./data/tracking/phi' + newparameternames)
    # phi_reconstruct = np.load('./data/tracking/phirecon' + parameternames)
    neighbour_track = np.load('./data/tracking/neighbour' + newparameternames)
    two_body_track = np.load('./data/tracking/twobody' + newparameternames)
    t_track = np.linspace(0.0, cycles, len(J_field_track))
    D_track = np.load('./data/tracking/double' + newparameternames)

    newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
        nx, cycles2, U2, t2, number2, delta2, field, F0, ascale2)

    J_field_track2 = np.load('./data/tracking/Jfield' + newparameternames2) / scalefactor2
    phi_track2 = np.load('./data/tracking/phi' + newparameternames2)
    # phi_reconstruct = np.load('./data/tracking/phirecon' + parameternames)
    neighbour_track2 = np.load('./data/tracking/neighbour' + newparameternames2)
    two_body_track2 = np.load('./data/tracking/twobody' + newparameternames2)
    t_track2 = np.linspace(0.0, cycles, len(J_field_track2))
    D_track2 = np.load('./data/tracking/double' + newparameternames2)

if Switch:
    # parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    # nx, cycles, U, t, number, delta, field, F0)
    newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
        nx, cycles, U, t, number, delta_switch, field, F0, ascale)
    switch_function = np.load('./data/switch/switchfunc' + newparameternames) / scalefactor
    J_field_switch = np.load('./data/switch/Jfield' + newparameternames) / scalefactor
    phi_switch = np.load('./data/switch/phi' + newparameternames)
    # phi_reconstruct = np.load('./data/tracking/phirecon' + parameternames)
    neighbour_switch = np.load('./data/switch/neighbour' + newparameternames)
    two_body_switch = np.load('./data/switch/twobody' + newparameternames)
    cut = 5
    switch_function = switch_function[:-cut]
    J_field_switch = J_field_switch[:-cut]
    phi_switch = phi_switch[:-cut]
    neighbour_switch = neighbour_switch[:-cut]
    two_body_switch = two_body_switch[:-cut]
    t_switch = np.linspace(0.0, cycles, len(J_field_switch))

if CutSpec:
    prop_cut = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t, F0=F0, a=ascale * a,
                          bc='pbc')
    prop_cut2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=ascale * a,
                           bc='pbc')
    delta_cut = prop_cut.freq * delta / prop.freq
    delta_cut2 = prop_cut2.freq * delta2 / prop.freq
    cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
    nx, cycles, U, t, number, delta, field, F0, ascale, cutoff)
    J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
    J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
    phi_cut_1 = np.load('./data/cutfreqs/phi_recon' + cutparameternames1)
    cutparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
        nx2, cycles2, U2, t2, number2, delta2, field2, F0, ascale2, cutoff2)
    J_cut_2 = np.load('./data/cutfreqs/Jfield' + cutparameternames2)

omegas = (np.arange(len(J_field)) - len(J_field) / 2) / cycles
omegas2 = (np.arange(len(J_field2)) - len(J_field2) / 2) / cycles
if Switch:
    omeaga_switch = (np.arange(len(phi_switch)) - len(phi_switch) / 2) / cycles
# delta2=delta2*factor

t = np.linspace(0.0, cycles, len(J_field))
t2 = np.linspace(0.0, cycles, len(J_field2))

darray = np.load('./data/original/doublonarray2.npy')
breaktimes = np.load('./data/original/breaktimes.npy')
#
print(darray.shape)
t_array = t2 = np.linspace(0.0, cycles, len(darray[0, :]))
breakline = []
cmap = plt.get_cmap('jet_r')
# plt.plot(t_array, D)

colouring = np.linspace(0, 1, 11)
for xx in range(0, 11):
    # color = cmap((float(10 * xx) - 7) / 45)
    # color2 = cmap((float(10 * (xx + 1)) - 7) / 45)
    color = plt.cm.jet(colouring[xx])
    if xx == 0 or xx == 10:
        plt.plot(t_array, darray[xx, :], color=color, label='$\\frac{U}{t_0}=$%s' % (xx))
    else:
        plt.plot(t_array, darray[xx, :], color=color)
for xx in range(0,len(breaktimes)):
        breakindex = int(breaktimes[xx] * len(darray[0, :]) / cycles)
        # plt.plot(breaktimes[xx], darray[xx+1,breakindex],color=color2, marker='o', markersize='10')
        plt.plot(breaktimes[xx], darray[xx + 1, breakindex], color='black', marker='o', markersize='7')
        breakline.append(darray[xx + 1, breakindex])
plt.plot(breaktimes[0], darray[1, int(breaktimes[0] * len(darray[0, :]) / cycles)], linestyle='none', color='black', marker='o',
         markersize='7', label='$t_{th}$')
plt.plot(breaktimes, breakline, linestyle='dashed', color='black')

# plt.plot(t_array, darray[xx, :], color=color)

plt.xlabel('Time [cycles]')
plt.ylabel('$D(t)$')
plt.legend(loc='upper right')
plt.show()

N_old = int(cycles / (prop.freq * delta)) + 1
times = np.linspace(0, cycles / prop.freq, N_old)

D_grad = D
D_grad2 = D2
if Tracking:
    D_grad_track = D_track
    D_grad_track2 = D_track2
#
# D_func = interp1d(t, D_grad, fill_value=0, bounds_error=False)
# # D_grad_track = np.gradient(D_track, delta_track)

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(t, J_field, label='$\\frac{U}{t_0}=0$')
ax2.plot(t2, J_field2, label='$\\frac{U}{t_0}=7$')
ax1.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
ax2.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')

plt.xlabel('Time [cycles]')
plt.show()


plt.subplot(211)
plt.plot(t, J_field, label='$\\frac{U}{t_0}=0$')
if Tracking:
    plt.plot(t_track* prop_track.freq / prop.freq, J_field_track, linestyle='dashed',
         label='Tracked Current')
# plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
# plt.legend(loc='upper right')
plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.08), fontsize=25)

plt.subplot(212)
plt.plot(t2, J_field2, label='\\frac{U}{t_0}=7')
if Tracking:
    plt.plot(t_track2, J_field_track2, linestyle='dashed',
         label='Tracked current')
plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
plt.annotate('b)', xy=(0.3, np.max(J_field2) - 0.05), fontsize=25)
plt.show()

plt.subplot(211)
plt.plot(t, D_grad, label='original')
if Tracking:
    plt.plot(t, D_grad, label='tracked', linestyle='dashed')
    plt.plot(t2, D_grad2, label='tracked', linestyle='dashed')
    plt.plot(t_track, D_grad_track, label='tracked', linestyle='dashed')
    plt.plot(t_track2, D_grad_track2, label='tracked', linestyle='dashed')
plt.ylabel('$D(t)$')
# plt.annotate('a)', xy=(0.3,np.max(D_grad)-0.005),fontsize='16')

plt.subplot(212)
plt.plot(t, J_field.real, label='original')
if Tracking:
    plt.plot(t_track, J_field_track.real, label='tracked', linestyle='dashed')
# plt.annotate('b)', xy=(0.3, np.max(J_field) - 0.05), fontsize='16')
plt.legend()
plt.ylabel('$J(t)$')
plt.xlabel('Time [cycles]')
plt.show()

if Switch:
    t = t1
    switchparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
        nx, cycles, 1 * t, t, number, delta_switch, field, F0, ascale)
    J_field_switch3 = np.load('./data/switch/Jfield' + switchparameternames) / scalefactor
    phi_switch3 = np.load('./data/switch/phi' + switchparameternames)
    J_field_switch3 = J_field_switch3[:-cut]
    phi_switch3 = phi_switch3[:-cut]

    switchparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
        nx, cycles, 0.5 * t, t, number, delta_switch, field, F0, ascale)
    J_field_switch2 = np.load('./data/switch/Jfield' + switchparameternames) / scalefactor
    phi_switch2 = np.load('./data/switch/phi' + switchparameternames)
    J_field_switch2 = J_field_switch2[:-cut]
    phi_switch2 = phi_switch2[:-cut]

    switchparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
        nx, cycles, 1.5 * t, t, number, delta_switch, field, F0, ascale)
    J_field_switch4 = np.load('./data/switch/Jfield' + switchparameternames) / scalefactor
    phi_switch4 = np.load('./data/switch/phi' + switchparameternames)
    J_field_switch4 = J_field_switch4[:-cut]
    phi_switch4 = phi_switch4[:-cut]

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(t_switch, phi_switch.real)
    ax1.plot(t_switch, phi_switch2.real, linestyle=':')
    ax1.plot(t_switch, phi_switch3.real, linestyle='--')
    ax1.plot(t_switch, phi_switch4.real, linestyle='-.')

    # plt.annotate('b)', xy=(0.3, np.max(J_field) - 0.05), fontsize='16')
    ax1.set_ylabel('$\Phi_T(t)$')
    # plt.plot(t_switch, switch_function.real, label='Switch Function')
    ax2.plot(t_switch, J_field_switch.real, label='$\\frac{U}{t_0}=0$')
    # ax2.plot(t_switch, J_field_switch2.real, label='$\\frac{U}{t_0}=0.5$', linestyle=':')
    # ax2.plot(t_switch, J_field_switch3.real, label='$\\frac{U}{t_0}=1$', linestyle='--')
    ax2.plot(t_switch, J_field_switch2.real, linestyle=':')
    ax2.plot(t_switch, J_field_switch3.real, linestyle='--')
    ax2.plot(t_switch, J_field_switch4.real, label='$\\frac{U}{t_0}=1.5$', linestyle='-.')

    # plt.annotate('b)', xy=(0.3, np.max(J_field) - 0.05), fontsize='16')
    ax2.set_ylabel('$J_T(t)$')
    plt.xlabel('Time [cycles]')
    ax2.legend(loc='upper right')
    plt.xlabel('Time [cycles]')

    plt.show()
    # plt.subplot(211)
    # plt.plot(t_switch, phi_switch.real)
    # plt.plot(t_switch, phi_switch2.real, linestyle=':')
    # plt.plot(t_switch, phi_switch3.real, linestyle='--')
    # plt.plot(t_switch, phi_switch4.real, linestyle='-.')
    #
    # # plt.annotate('b)', xy=(0.3, np.max(J_field) - 0.05), fontsize='16')
    # plt.ylabel('$\Phi_T(t)$')
    #
    # plt.subplot(212)
    # # plt.plot(t_switch, switch_function.real, label='Switch Function')
    # plt.plot(t_switch, J_field_switch.real, label='$\\frac{U}{t_0}=0$')
    # plt.plot(t_switch, J_field_switch2.real, label='$\\frac{U}{t_0}=0.5$', linestyle=':')
    # plt.plot(t_switch, J_field_switch3.real, label='$\\frac{U}{t_0}=1$', linestyle='--')
    # plt.plot(t_switch, J_field_switch4.real, label='$\\frac{U}{t_0}=1.5$', linestyle='-.')
    #
    # # plt.annotate('b)', xy=(0.3, np.max(J_field) - 0.05), fontsize='16')
    # plt.legend()
    # plt.ylabel('$J_T(t)$')
    # plt.xlabel('Time [cycles]')
    # plt.show()

    two_body_switch = np.array(two_body_switch)
    extra_switch = 2. * np.real(np.exp(-1j * phi_switch) * two_body_switch)

    diff_switch = phi_switch - np.angle(neighbour_switch)
    J_grad_switch = -2. * prop_switch.a * prop_switch.t * np.gradient(phi_switch, delta_switch) * np.abs(
        neighbour_switch) * np.cos(diff_switch)
    exact_switch = np.gradient(J_field_switch, delta_switch)
    eq32_switch = (J_grad_switch - prop_switch.a * prop_switch.t * prop_switch.U * extra_switch) / scalefactor
    eq33_switch = J_grad_switch + 2. * prop_switch.a * prop_switch.t * (
            np.gradient(np.angle(neighbour_switch), delta_switch) * np.abs(neighbour_switch) * np.cos(
        diff_switch) - np.gradient(
        np.abs(neighbour_switch), delta_switch) * np.sin(diff_switch))

    plt.plot(t_switch, exact_switch, label='Numerical gradient')
    plt.plot(t_switch, eq32_switch, linestyle='dashed',
             label='Analytical gradient')
    # plt.xlabel('Time [cycles]')
    plt.ylabel('$\\dot{J}(t)$')
    plt.legend()
    plt.show()
#
#
"""Phi and theta plots"""
plt.plot(t, phi_original.real, label='original')
# plt.plot(t2, J_field2.real, label='swapped')
if Tracking:
    # plt.plot(t_track, phi_track.real-np.angle(neighbour_track), label='tracked', linestyle='dashed')
    plt.plot(t_track, phi_track.real, label='tracked', linestyle='dashed')
#
# # plt.legend()
# plt.xlabel('Time [cycles]')
# plt.ylabel('$\\Phi(t)$')
# plt.yticks(np.arange(-1 * np.pi, 1 * np.pi, 0.5 * np.pi),
#            [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1 * np.pi, .5 * np.pi)])
# plt.show()
# if Tracking:
#     phi_track_shift = np.copy(phi_track)
#     for j in range(1, int(len(phi_track_shift))):
#         k = phi_track_shift[j] - phi_track_shift[j - 1]
#         if k > 1.8 * np.pi:
#             phi_track_shift[j:] = phi_track_shift[j:] - 2 * np.pi
#         if k < -1.8* np.pi:
#             phi_track_shift[j:] = phi_track_shift[j:] + 2 * np.pi
#     plt.plot(t_track, phi_track_shift.real+2*np.pi, label='shifted phi')
#     plt.plot(t_track,np.zeros(len(t_track)))
#     print(np.sum(phi_track_shift.real)*delta_track+delta_track*len(phi_track_shift)*2.105*np.pi)
#     print(np.sum(phi_track.real)*delta_track)
#     print(np.sum(phi_track2))
#     print(np.sum(phi_original)*delta_track)
#     # plt.legend()
#     plt.xlabel('Time [cycles]')
#     plt.ylabel('$\\Phi(t)$')
#     plt.yticks(np.arange(-3 * np.pi, 3 * np.pi, 0.5 * np.pi),
#            [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-3 * np.pi, 3 * np.pi, .5 * np.pi)])
#     plt.show()
#
#     plt.plot(t_track,np.angle(neighbour_track), label='tracked', linestyle='dashed')
#
#     # plt.legend()
#     plt.xlabel('Time [cycles]')
#     plt.ylabel('$\\theta(t)$')
#     plt.yticks(np.arange(-1 * np.pi, 1 * np.pi, 0.5 * np.pi),
#                [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1 * np.pi, .5 * np.pi)])
#     plt.show()
#
#     plt.plot(t_track, phi_track.real-np.angle(neighbour_track), label='tracked', linestyle='dashed')
#
#     # plt.legend()
#     plt.xlabel('Time [cycles]')
#     plt.ylabel('$\\Phi(t)-\\theta(t)$')
#     plt.yticks(np.arange(-1 * np.pi, 1 * np.pi, 0.5 * np.pi),
#                [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1 * np.pi, .5 * np.pi)])
#     plt.show()
#
# plt.plot(t2, phi_original2.real, label='original')
# # plt.plot(t2, J_field2.real, label='swapped')
#
# if Tracking:
#     # plt.plot(t_track, phi_track2.real-np.angle(neighbour_track2), label='tracked', linestyle='dashed')
#     plt.plot(t_track2, phi_track2, label='tracked', linestyle='dashed')
# # plt.legend()
# plt.xlabel('Time [cycles]')
# plt.ylabel('$\\Phi(t)$')
# plt.yticks(np.arange(-1 * np.pi, 1 * np.pi, 0.5 * np.pi),
#            [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1 * np.pi, .5 * np.pi)])
# plt.show()
#
# if Tracking:
#     plt.plot(t_track2,np.angle(neighbour_track2), label='tracked', linestyle='dashed')
#
#     # plt.legend()
#     plt.xlabel('Time [cycles]')
#     plt.ylabel('$\\theta(t)$')
#     plt.yticks(np.arange(-1 * np.pi, 1 * np.pi, 0.5 * np.pi),
#                [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1 * np.pi, .5 * np.pi)])
#     plt.show()
#
#     plt.plot(t_track2, phi_track2.real-np.angle(neighbour_track2), label='tracked', linestyle='dashed')
#
#     plt.show()
#
#
# # plt.plot(t2, J_field2.real, label='swapped')
# if Tracking:
#     plt.plot(t_track, np.abs(neighbour_track), label='tracked', linestyle='dashed')
# plt.legend()
# plt.xlabel('Time [cycles]')
# plt.ylabel('$R(t)$')
# plt.show()
#
if Tracking:
    # plt.plot(t_track,0.25*(phi_track_shift+2*np.pi*np.ones(len(t_track))))
    # plt.plot(t_track2,10*phi_track2)
    # plt.plot(t,phi_original)
    # plt.plot(t,J_field, color='red')
    # plt.plot(t2,10*J_field2,color='red')
    plt.plot(t_track,0.25*phi_track,color='red')
    plt.axis('off')
    plt.show()
# #
two_body = np.array(two_body)
extra = 2. * np.real(np.exp(-1j * phi_original) * two_body)
diff = phi_original - np.angle(neighbour)
two_body2 = np.array(two_body2)
extra2 = 2. * np.real(np.exp(-1j * phi_original2) * two_body2)
diff2 = phi_original2 - np.angle(neighbour2)
J_grad = -2. * prop.a * prop.t * np.gradient(phi_original, delta) * np.abs(neighbour) * np.cos(diff)
J_grad2 = -2. * prop2.a * prop2.t * np.gradient(phi_original2, delta2) * np.abs(neighbour2) * np.cos(diff2)
exact = np.gradient(J_field, delta1)
exact2 = np.gradient(J_field2, delta2)
#
#
#
# # eq 32 should have a minus sign on the second term, but
eq32 = J_grad - prop.a * prop.t * prop.U * extra
# eq32= -prop.a * prop.t * prop.U * extra
eq33 = J_grad + 2. * prop.a * prop.t * (
        np.gradient(np.angle(neighbour), delta1) * np.abs(neighbour) * np.cos(diff) - np.gradient(
    np.abs(neighbour), delta1) * np.sin(diff))

# Just in case you want to plot from a second simulation

eq32_2 = J_grad2 - prop2.a * prop2.t * prop2.U * extra2
eq33_2 = J_grad2 + 2. * prop2.a * prop2.t * (
        np.gradient(np.angle(neighbour2), delta2) * np.abs(neighbour2) * np.cos(diff2) - np.gradient(
    np.abs(neighbour2), delta2) * np.sin(diff2))

# plt.plot(t, eq33-J_grad, label='Gradient calculated via expectations', linestyle='dashdot')
# plt.plot(t, eq32-J_grad, linestyle='dashed')
# plt.show()


# plot various gradient calculations
# plt.plot(t, eq33, label='Gradient calculated via expectations', linestyle='dashdot')
plt.subplot(211)
plt.plot(t, exact, label='Numerical gradient')
plt.plot(t, eq32, linestyle='dashed',
         label='Analytical gradient')
# plt.xlabel('Time [cycles]')
plt.ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
plt.legend()
plt.annotate('a)', xy=(0.3, np.max(exact) - 0.2), fontsize='25')

plt.subplot(212)
plt.plot(t2, exact2, label='Numerical gradient')
plt.plot(t2, eq32_2, linestyle='dashed',
         label='Analytical gradient')
plt.xlabel('Time [cycles]')
plt.ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
plt.annotate('b)', xy=(0.3, np.max(exact) - 0.1), fontsize='25')
plt.show()

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(t, exact, label='Numerical gradient')
ax1.plot(t, eq32, linestyle='dashed',
         label='Analytical gradient')
# plt.xlabel('Time [cycles]')
ax1.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
ax1.annotate('a)', xy=(0.3, np.max(exact) - 0.2), fontsize='28')

ax2.plot(t2, exact2, label='Numerical gradient')
ax2.plot(t2, eq32_2, linestyle='dashed',  label='Analytical gradient')
plt.xlabel('Time [cycles]')
ax2.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
ax2.annotate('b)', xy=(0.3, np.max(exact) - 0.1), fontsize='28')
plt.legend(loc='lower left')
plt.show()


if Tracking:
    phi_track_shift = np.copy(phi_track)
    theta=np.angle(neighbour_track)
    if Tracking:
        for j in range(1, int(len(phi_track_shift))):
            k = phi_track_shift[j] - phi_track_shift[j - 1]
            k2= theta[j] - theta[j - 1]

            if k > 1.8 * np.pi:
                phi_track_shift[j:] = phi_track_shift[j:] - 2 * np.pi
            if k < -1.8 * np.pi:
                phi_track_shift[j:] = phi_track_shift[j:] + 2 * np.pi
            if k2 > 1.8 * np.pi:
                theta[j:] = theta[j:] - 2 * np.pi
            if k2 < -1.8 * np.pi:
                theta[j:] = theta[j:] + 2 * np.pi
    phi_track=phi_track_shift
    two_body_track = np.array(two_body_track)
    plt.plot(two_body_track)
    plt.show()
    extra_track = 2. * np.real(np.exp(-1j * phi_track) * two_body_track)


    diff_track = phi_track - theta
    J_grad_track = -2. * prop_track.a * prop_track.t * np.gradient(phi_track, delta_track) * np.abs(
        neighbour_track) * np.cos(diff_track)
    plt.plot(J_grad_track)
    plt.show()
    exact_track = np.gradient(J_field_track, delta_track)
    exact_track2 = np.gradient(J_field_track2, delta_track2)

    print(max(exact) / max(exact_track))
    #
    #
    #
    # # eq 32 should have a minus sign on the second term, but
    eq32_track = (J_grad_track - prop_track.a * prop_track.t * prop_track.U * extra_track) / scalefactor
    print(max(eq32) / max(eq32_track))
    # eq32= -prop.a * prop.t * prop.U * extra
    eq33_track = (J_grad_track + 2. * prop_track.a * prop_track.t * (
            np.gradient(theta, delta_track) * np.abs(neighbour_track) * np.cos(
        diff_track) - np.gradient(
        np.abs(neighbour_track), delta_track) * np.sin(diff_track)))/scalefactor
    # for j in range(1,len(eq32_track)):
    #     if abs(eq32_track[j]-eq32_track[j-1]) > 100:
    #         eq32_track[j]= eq32_track[j - 1]

    # Just in case you want to plot from a second simulation

    # plt.plot(t, eq33-J_grad, label='Gradient calculated via expectations', linestyle='dashdot')
    # plt.plot(t, eq32-J_grad, linestyle='dashed')
    # plt.show()
    # #

    # plot various gradient calculations
    # plt.plot(t, eq33, label='Gradient calculated via expectations', linestyle='dashdot')
    plt.subplot(311)
    # plt.plot(t, eq32, label='original')
    plt.plot(t_track, exact_track,
             label='numerical')
    plt.plot(t_track[:-2], eq33_track[:-2], linestyle='dashed',
             label='analytical')
    plt.plot(t_track[:-2], eq32_track[:-2], linestyle='dashed',
             label='tracked')
    plt.ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
    plt.legend()

    plt.subplot(312)
    plt.plot(t, np.abs(neighbour), label='original')
    plt.plot(t_track, np.abs(neighbour_track), linestyle='dashed')
    plt.ylabel('$R(\psi)$')

    plt.subplot(313)
    plt.plot(t, np.abs(two_body), label='original')
    plt.plot(t_track, np.abs(two_body_track), linestyle='dashed')
    plt.ylabel('$C(\psi)$')
    plt.xlabel('Time [cycles]')

    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    ax1.plot(t_track, exact_track,
          label='Original')
    # ax1.plot(t_track[:-2], eq33_track[:-2], linestyle='dashed',   label='Tracking')
    ax1.plot(t_track[:-2], eq32_track[:-2], linestyle='dashed',
             label='Tracking')
    ax1.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
    ax1.legend(loc='upper right')

    ax2.plot(t, np.abs(neighbour), label='original')
    ax2.plot(t_track, np.abs(neighbour_track), linestyle='dashed')
    ax2.set_ylabel('$R(\psi)$')

    ax3.plot(t, np.abs(two_body), label='original')
    ax3.plot(t_track, np.abs(two_body_track), linestyle='dashed')
    ax3.set_ylabel('$C(\psi)$')

    plt.xlabel('Time [cycles]')

    plt.show()


    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(t_track, exact_track)
    # ax1.plot(t_track[:-2], eq33_track[:-2], linestyle='dashed',   label='Tracking')
    ax1.plot(t_track[:-2], eq32_track[:-2], linestyle='dashed',
             label='$\\Delta=10^{-2}$')
    ax1.set_ylabel('$\\frac{{\\rm d}J}{{\\rm d}t}$')
    ax1.legend(loc='upper right')

    J_field_track = np.load('./data/tracking/Jfield' + newparameternames) * Jscale / scalefactor



    ax2.plot(t, np.angle(neighbour), label='original')
    ax2.plot(t_track, np.abs(neighbour_track), linestyle='dashed')
    ax2.set_ylabel('$R(\psi)$')


    plt.xlabel('Time [cycles]')

    plt.show()

# w1, spec1 = har_spec.spectrum_welch(exact, delta1)
# w2, spec2 = har_spec.spectrum_welch(exact2, delta2)
# plt.semilogy(w1, spec1, label='$D^{(0)}(t)$')
# plt.semilogy(w2, spec2, label='$D^{(7)}(t)$')
# plt.show()



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
plot_spectra([0, 7], w, spec, min_spec, max_harm)

#
# """Cutting the spectrum in order to evolve it"""
# if Tracking:
#     cutoff=200
#     cutoff2=cutoff
#     w *= 2. * np.pi / prop.field
#     phi_f1=FT(phi_track)
#     phi_f2=FT(phi_track2)
#     # phi_f1=np.fft.fftshift(np.fft.fft(phi_track))
#     # phi_f2=np.fft.fft(phi_track2)
#     w_phi1= np.fft.fftshift(np.fft.fftfreq(len(phi_track),delta))
#     w_phi1*= 2. * np.pi / prop_track.field
#
#     w_phi2= np.fft.fftshift(np.fft.fftfreq(len(phi_track2),delta))
#     w_phi2*= 2. * np.pi / prop_track2.field
#     plt.plot(w_phi1[2:-2], np.log10(phi_f1)[2:-2])
#     plt.plot(w_phi2[2:-2], np.log10(phi_f2)[2:-2])
#     plt.plot(np.log10(phi_f1))
#     plt.plot(np.log10(phi_f2))
#     # axes = plt.gca()
#     # axes.set_xlim([0, 30])
#     # plt.xlim([-1,60])
#     plt.show()
#
#     a=np.nonzero(w_phi1>-cutoff)[0]
#     b=np.nonzero(w_phi1<cutoff)[-1]
#     print(a)
#     print(b)
#     a2=np.nonzero(w_phi2>-cutoff2)[0]
#     b2=np.nonzero(w_phi2<cutoff2)[-1]
#     # plt.plot(w_phi1[a[0]:b[-1]],np.log10(phi_f1)[a[0]:b[-1]])
#     # plt.plot(w_phi2[a2[0]:b2[-1]],np.log10(phi_f2)[a2[0]:b2[-1]])
#     # plt.show()
#     phi_f1[b[-1]:]=0
#     phi_f1[:a[0]]=0
#     phi_f2[b2[-1]:]=0
#     phi_f2[:a2[0]]=0
#     cutphi_1=iFT((phi_f1))
#     cutphi_2=iFT((phi_f2))
#
#     plt.plot(w_phi1, np.log10(phi_f1))
#     plt.show()
#     plt.plot(w_phi2, np.log10(phi_f2))
#     plt.show()
#
#     plt.plot(t_track,cutphi_1)
#     plt.plot(t_track2,cutphi_2)
#     plt.show()
#
#     t1=0.52
#     t2=0.52
#     cutparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
#         nx, cycles, U, t1, number, delta, field, F0, ascale,cutoff)
#     cutparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
#         nx2, cycles2, U2, t2, number2, delta2, field2, F0, ascale2,cutoff2)
#     np.save('./data/cutfreqs/phi'+cutparameternames,cutphi_1)
#     np.save('./data/cutfreqs/phi'+cutparameternames2,cutphi_2)
#


if Tracking:
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.plot(t_array, darray[0, :], label='$D^{(0)}(t)$')
    ax2.plot(t2, D_grad2, label='$D^{(7)}(t)$')
    ax1.plot(t_track2, D_grad_track2, label='$D_T^{(0)}(t)$', linestyle='dashed')
    ax2.plot(t_track, D_grad_track, label='$D_T^{(7)}(t)$', linestyle='dashed')
    # ax2.plot(t_track[1390], D_grad_track[1390], linestyle='none', color='black',
    ax2.plot(t_track[653], D_grad_track[653], linestyle='none', color='black',
             marker='o',
             markersize='7', label='$t_{th}$')
    ax1.set_ylabel('$D(t)$')
    ax2.set_ylabel('$D(t)$')
    plt.xlabel('Time [cycles]')
    ax1.legend()
    ax2.legend()
    plt.show()

    spec = np.zeros((FT_count(len(exact)), 4))
    print(len(exact_track))
    print(len(exact))
    print(delta_track)
    if method == 'welch':
        w, spec[:, 0] = har_spec.spectrum_welch(exact, delta1)
        w2, spec[:, 1] = har_spec.spectrum_welch(exact2, delta2)
        w3, spec[:, 2] = har_spec.spectrum_welch(exact_track2, delta_track2)
        w4, spec[:, 3] = har_spec.spectrum_welch(exact_track, delta_track)
        # w4, spec[:, 3] = har_spec.spectrum_welch(exact_track, delta_track)
    elif method == 'hann':
        w, spec[:, 0] = har_spec.spectrum_hanning(exact, delta1)
        w2, spec[:, 0] = har_spec.spectrum_hanning(exact2, delta2)
    elif method == 'none':
        w, spec[:, 0] = har_spec.spectrum(exact, delta1)
        w2, spec[:, 0] = har_spec.spectrum(exact2, delta2)
    else:
        print('Invalid spectrum method')
    w *= 2. * np.pi / prop.field
    plot_spectra_track(['$\mathcal{F}\left(\\frac{{\\rm d}J^{(0)}}{{\\rm d} t}\\right)$',
                        '$\mathcal{F}\left(\\frac{{\\rm d}J^{(7)}}{{\\rm d} t}\\right)$',
                        '$\mathcal{F}\left(\\frac{{\\rm d}J_T^{(0)}}{{\\rm d} t}\\right)$',
                        '$\mathcal{F}\left(\\frac{{\\rm d}J_T^{(7)}}{{\\rm d} t}\\right)$'], w, spec, min_spec,
                       max_harm)

if CutSpec:
    exact_track = np.gradient(J_cut_1/scalefactor, delta_cut)
    exact_track_alt = np.gradient(J_cut_alt_1.real/scalefactor, delta_cut)
    print(len(exact_track))
    print(len(exact_track_alt))
    exact_track2 = np.gradient(J_cut_2, delta_cut2)
    spec = np.zeros((FT_count(len(exact)), 4))
    cut_times=np.linspace(0,cycles,len(J_cut_1))
    plt.plot(cut_times[:-10],100*J_cut_alt_1[:-10])
    plt.plot(t, J_field, label='Original Current')
    # plt.xlim([0,9])
    plt.show()
    plt.plot(cut_times, phi_cut_1)
    # plt.plot(cut_times,phi_track_shift)
    plt.show()
    # plt.plot(J_cut_2)
    # plt.show()
    if method == 'welch':
        w, spec[:, 0] = har_spec.spectrum_welch(exact, delta1)
        w2, spec[:, 1] = har_spec.spectrum_welch(exact2, delta2)
        w3, spec[:, 2] = har_spec.spectrum_welch(exact_track2, delta_cut2)
        w3, spec[:, 3] = har_spec.spectrum_welch(exact_track_alt, delta_cut)
        # w4, spec[:, 3] = har_spec.spectrum_welch(exact_track, delta_track)
    elif method == 'hann':
        w, spec[:, 0] = har_spec.spectrum_hanning(exact, delta1)
        w2, spec[:, 0] = har_spec.spectrum_hanning(exact2, delta2)
    elif method == 'none':
        w, spec[:, 0] = har_spec.spectrum(exact, delta1)
        w2, spec[:, 0] = har_spec.spectrum(exact2, delta2)
    else:
        print('Invalid spectrum method')
    w *= 2. * np.pi / prop.field
    plot_spectra_track(['$\mathcal{F}\left(\\frac{{\\rm d}J^{(0)}}{{\\rm d} t}\\right)$',
                        '$\mathcal{F}\left(\\frac{{\\rm d}J^{(7)}}{{\\rm d} t}\\right)$',
                        '$\mathcal{F}\left(\\frac{{\\rm d}J_T^{(0)}}{{\\rm d} t}\\right)$',
                        '$\mathcal{F}\left(\\frac{{\\rm d}J_T^{(7)}}{{\\rm d} t}\\right)$'], w, spec, min_spec,
                       max_harm)

#     spec = np.log10(spec)
    for deltas in [0.05,0.005]:
        t = 0.52
        delta = deltas
        delta_cut = prop_cut.freq * delta / prop.freq
        cutoff = 200
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles, U, t, number, delta, field, F0, ascale, cutoff)
        J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
        J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
        new_cut_times=np.linspace(0,cycles,len(J_cut_alt_1))
        plt.plot(new_cut_times[:-10],J_cut_alt_1[:-10]/scalefactor,label='$\\Delta=$ %s' %(deltas))
    times=np.linspace(0,cycles,len(J_field))
    plt.plot(times,J_field,color='black',label='exact')
    plt.xlabel('time')
    plt.ylabel('$J(t)$')
    plt.legend()
    plt.show()

    colouring = np.linspace(0, 1,7)
    jk=6


    for cutoffs in [40,60,100]:
        color = plt.cm.jet(colouring[jk])
        t = 0.52
        delta = 0.005
        ascale=1
        scalefactor=1/100
        delta_cut = prop_cut.freq * delta / prop.freq
        # cutoff = 100
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles, U, t, number, delta, field, F0, ascale, cutoffs)
        J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
        J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
        new_cut_times=np.linspace(0,cycles,len(J_cut_alt_1))
        plt.plot(new_cut_times[:-10],J_cut_alt_1[:-10]/scalefactor,label='$\\omega_c=$%s$\\omega_o$' % (cutoffs),color=color)
        jk-=1
    times=np.linspace(0,cycles,len(J_field))
    plt.plot(times,J_field,label='exact',color='black')
    plt.legend(loc='upper right')
    plt.ylim([-3,3])
    # plt.title('$J_T^{(7)}$')
    plt.show()



    xlines = [2 * i - 1 for i in range(1, 6)]

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)


    for cutoffs in [10, 20, 40]:
        t = 0.52
        delta = 0.005
        ascale = 5
        scalefactor = 1
        delta_cut = prop_cut.freq * delta / prop.freq
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles, U, t, number, delta, field, F0, ascale, cutoffs)
        J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
        J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
        exact_track_alt = np.gradient(J_cut_alt_1.real / scalefactor, delta_cut)
        w, spec = har_spec.spectrum_welch(exact_track_alt, delta_cut)
        # spec = np.log10(spec)
        w *= 2. * np.pi / prop.field
        ax1.semilogy(w, spec, label='$\\omega_c=$%s$\\omega_o$' % (cutoffs))
    for xc in xlines:
        ax1.axvline(x=xc, color='black', linestyle='dashed')
    w, spec = har_spec.spectrum_welch(exact, delta1)
    w *= 2. * np.pi / prop.field
    ax1.semilogy(w, spec, color='black', linestyle='-.', label='Target')
    ax1.annotate('a)', xy=(10.5, 10**(0)), fontsize='35')
    ax1.set_ylabel('HHG spectra')
    ax1.legend(loc='upper right')
    ax1.set_xlim([0, max_harm])
    ax1.set_ylim([10**(-12), 10**2])
    # plt.title('$J_T^{(1)}$')

    for cutoffs in [10,20,40]:
        t=0.52
        delta=0.05
        ascale=1.001
        scalefactor=1
        delta_cut = prop_cut.freq * delta / prop.freq
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles, U2, t, number, delta, field, F0, ascale, cutoffs)
        J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
        J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
        exact_track_alt = np.gradient(J_cut_alt_1.real / scalefactor, delta_cut)
        w, spec = har_spec.spectrum_welch(exact_track_alt,delta_cut)
        # spec=np.log10(spec)
        w *= 2. * np.pi / prop.field
        ax2.semilogy(w,spec, label='$\\omega_c=$%s$\\omega_o$' % (cutoffs))
    for xc in xlines:
        ax2.axvline(x=xc, color='black', linestyle='dashed')
    w, spec = har_spec.spectrum_welch(exact2, delta1)
    # spec = np.log10(spec)
    w *= 2. * np.pi / prop.field
    ax2.semilogy(w, spec, color='black', linestyle='-.', label='Target')
    ax2.set_ylabel('HHG spectra')
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, max_harm])
    ax2.set_ylim([10**(-12), 10**2])
    ax2.annotate('b)', xy=(10.5, 10**(0)), fontsize='35')
    plt.xlabel('Harmonic Order')
    plt.savefig("newtest.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    fig, (ax2, ax1) = plt.subplots(nrows=2, sharex=True)

    for cutoffs in [10, 20, 40]:
        t = 0.52
        delta = 0.05
        ascale = 1
        scalefactor = 1
        delta_cut = prop_cut.freq * delta / prop.freq
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles, U2, t, number, delta, field, F0, ascale, cutoffs)
        J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
        J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
        exact_track_alt = np.gradient(J_cut_alt_1.real / scalefactor, delta_cut)
        w, spec = har_spec.spectrum_welch(exact_track_alt, delta_cut)
        # spec = np.log10(spec)
        w *= 2. * np.pi / prop.field
        ax1.semilogy(w, spec, label='$\\omega_c=$%s$\\omega_o$' % (cutoffs))
    for xc in xlines:
        ax1.axvline(x=xc, color='black', linestyle='dashed')
    w, spec = har_spec.spectrum_welch(exact2, delta1)
    w *= 2. * np.pi / prop.field
    ax1.semilogy(w, spec, color='black', linestyle='-.', label='Target')
    ax1.set_ylabel('HHG spectra')
    ax1.legend(loc='upper right')
    ax1.annotate('b)', xy=(10.5, 10**(0)), fontsize='35')
    ax1.set_xlim([0, max_harm])
    ax1.set_ylim([10 ** (-12), 10 ** 2])
    # plt.title('$J_T^{(1)}$')

    for cutoffs in [40, 60, 100]:
        t = 0.52
        delta = 0.005
        ascale = 1
        scalefactor = 1 / 100
        delta_cut = prop_cut.freq * delta / prop.freq
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles, U, t, number, delta, field, F0, ascale, cutoffs)
        J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
        J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
        exact_track_alt = np.gradient(J_cut_alt_1.real / scalefactor, delta_cut)
        w, spec = har_spec.spectrum_welch(exact_track_alt, delta_cut)
        # spec=np.log10(spec)
        w *= 2. * np.pi / prop.field
        ax2.semilogy(w, spec, label='$\\omega_c=$%s$\\omega_o$' % (cutoffs))
    for xc in xlines:
        ax2.axvline(x=xc, color='black', linestyle='dashed')
    w, spec = har_spec.spectrum_welch(exact, delta1)
    # spec = np.log10(spec)
    w *= 2. * np.pi / prop.field
    ax2.semilogy(w, spec, color='black', linestyle='-.', label='Target')
    ax2.annotate('a)', xy=(10.5, 10**(0)), fontsize='35')
    ax2.set_ylabel('HHG spectra')
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, max_harm])
    ax2.set_ylim([10 ** (-12), 10 ** 2])
    plt.xlabel('Harmonic Order')
    plt.savefig("newtest.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    for cutoffs in [10, 20, 50]:
        t = 0.52
        delta = 0.005
        ascale = 5
        scalefactor = 1
        delta_cut = prop_cut.freq * delta / prop.freq
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles, U, t, number, delta, field, F0, ascale, cutoffs)
        J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
        J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
        exact_track_alt = np.gradient(J_cut_alt_1.real / scalefactor, delta_cut)
        w, spec = har_spec.spectrum_welch(exact_track_alt, delta_cut)
        # spec = np.log10(spec)
        w *= 2. * np.pi / prop.field
        plt.semilogy(w, spec, label='$\\omega_c=$%s$\\omega_o$' % (cutoffs))
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
    w, spec = har_spec.spectrum_welch(exact, delta1)
    w *= 2. * np.pi / prop.field
    plt.semilogy(w, spec, color='black', label='Exact')
    plt.xlabel('Harmonic Order')
    plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    axes = plt.gca()
    axes.set_xlim([0, max_harm])
    axes.set_ylim([10**(-10), 10**1])
    # plt.title('$J_T^{(1)}$')
    plt.show()

    xlines = [2 * i - 1 for i in range(1, 6)]
    for cutoffs in [40,60,100]:
        t=0.52
        delta=0.005
        ascale=1
        scalefactor=1/100
        delta_cut = prop_cut.freq * delta / prop.freq
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles, U, t, number, delta, field, F0, ascale, cutoffs)
        J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
        J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
        exact_track_alt = np.gradient(J_cut_alt_1.real / scalefactor, delta_cut)
        w, spec = har_spec.spectrum_welch(exact_track_alt,delta_cut)
        # spec=np.log10(spec)
        w *= 2. * np.pi / prop.field
        plt.semilogy(w,spec, label='$\\omega_c=$%s$\\omega_o$' % (cutoffs))
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
    w, spec = har_spec.spectrum_welch(exact, delta1)
    # spec = np.log10(spec)
    w *= 2. * np.pi / prop.field
    plt.semilogy(w, spec, color='black', label='Exact')
    plt.xlabel('Harmonic Order')
    plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    axes = plt.gca()
    axes.set_xlim([0, max_harm])
    axes.set_ylim([10**(-10), 10**1])
    # plt.title('$J_T^{(7)}$')
    plt.savefig("test.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    for delta in [0.05,0.02,0.005]:
        t=0.52
        cutoffs=100
        delta_cut = prop_cut.freq * delta / prop.freq
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles, U, t, number, delta, field, F0, ascale, cutoffs)
        J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
        J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
        exact_track_alt = np.gradient(J_cut_alt_1.real / scalefactor, delta_cut)
        w, spec = har_spec.spectrum_welch(exact_track_alt,delta_cut)
        spec=np.log10(spec)
        w *= 2. * np.pi / prop.field
        plt.plot(w,spec, label='$\\Delta=$%s' % (delta))
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
    plt.xlabel('Harmonic Order')
    plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    axes = plt.gca()
    axes.set_xlim([0, max_harm])
    axes.set_ylim([-10, 2])
    plt.show()

    for cutoffs in [60,100,200]:
        t = 0.52
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles, U, t, number, delta, field, F0, ascale, cutoffs)
        D_cut = np.load('./data/cutfreqs/doublon' + cutparameternames1)
        plt.plot(new_cut_times, D_cut, label='$\\omega_c=$%s$\\omega_o$' % (cutoffs))
    plt.legend(loc='upper right')
    plt.show()

    spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    w, spec = har_spec.spectrum_welch(exact, delta1)
    spec = np.log10(spec)
    w *= 2. * np.pi / prop.field
    plt.plot(w, spec, label='Exact')
    for delta in [0.05,0.02,0.005]:
        t = 0.52
        cutoffs=200
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles, U, t, number, delta, field, F0, ascale, cutoffs)
        J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
        J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
        delta_cut = prop_cut.freq * delta / prop.freq
        exact_track_alt = np.gradient(J_cut_alt_1.real / scalefactor, delta_cut)
        w, spec = har_spec.spectrum_welch(exact_track_alt, delta_cut)
        spec = np.log10(spec)
        w *= 2. * np.pi / prop.field
        plt.plot(w, spec, label='$\\Delta=$%s' % (delta))
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
    plt.xlabel('Harmonic Order')
    plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    axes = plt.gca()
    axes.set_xlim([0, max_harm])
    axes.set_ylim([-16, -1])
    plt.show()


    xlines = [2 * i - 1 for i in range(1, 6)]
    for cutoffs in [5,10,20,50]:
        t=0.52
        delta=0.05
        ascale=1
        scalefactor=1
        delta_cut = prop_cut.freq * delta / prop.freq
        cutparameternames1 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale-%s-cutoff.npy' % (
            nx, cycles2, U2, t, number2, delta2, field, F0, ascale, cutoffs)
        J_cut_1 = np.load('./data/cutfreqs/Jfield' + cutparameternames1)
        J_cut_alt_1 = np.load('./data/cutfreqs/Jfieldalt' + cutparameternames1)
        exact_track_alt = np.gradient(J_cut_alt_1.real / scalefactor, delta_cut)
        w, spec = har_spec.spectrum_welch(exact_track_alt,delta_cut)
        # spec=np.log10(spec)
        w *= 2. * np.pi / prop.field
        plt.semilogy(w,spec, label='$\\omega_c=$%s$\\omega_o$' % (cutoffs))
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
    w, spec = har_spec.spectrum_welch(exact2, delta2)
    # spec = np.log10(spec)
    w *= 2. * np.pi / prop.field
    plt.semilogy(w, spec, color='black', label='Exact')
    plt.xlabel('Harmonic Order')
    plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    axes = plt.gca()
    axes.set_xlim([0, max_harm])
    axes.set_ylim([10**(-15), 10**1])
    # plt.title('$J_T^{(7)}$')
    plt.savefig("test.pdf", format="pdf", bbox_inches="tight")
    plt.show()


# plot_spectrogram(t_switch, w, spec[:,0], min_spec=11, max_harm=60)
if Switch:
    spec = np.zeros((FT_count(len(phi_switch)), 2))
    if method == 'welch':
        w, spec[:, 0] = har_spec.spectrum_welch(phi_switch, delta_switch)
        # w2, spec[:,1] = har_spec.spectrum_welch(exact_track, delta_track)
        w2, spec[:, 1] = har_spec.spectrum_welch(J_field_switch, delta_switch)
    elif method == 'hann':
        w, spec[:, 0] = har_spec.spectrum_hanning(exact, delta1)
        w2, spec[:, 0] = har_spec.spectrum_hanning(exact2, delta2)
    elif method == 'none':
        w, spec[:, 0] = har_spec.spectrum(exact, delta1)
        w2, spec[:, 0] = har_spec.spectrum(exact2, delta2)
    else:
        print('Invalid spectrum method')
    w *= 2. * np.pi / prop.field
    plot_spectra_switch(['a', 'b'], w, spec, min_spec, max_harm)
# converts frequencies to harmonics


if Switch:
    omegas = omeaga_switch
    Y, X, Z1 = stft(phi_switch.real, 1, nperseg=len(t_switch) / 100, window=('gaussian', 2 / prop_switch.field))
    Z1 = np.abs(Z1) ** 2
    spec = np.log10(spec[:len(w)])
    specn = ma.masked_where(spec < -min_spec, spec)
    cm.RdYlBu_r.set_bad(color='white', alpha=None)
    # plt.pcolormesh(X*delta, Y*omegas.max()/Y.max(), Z1, norm=colors.LogNorm())
    plt.pcolormesh(X * delta, Y * omegas.max() / Y.max(), Z1, cmap='RdYlBu_r')

    plt.title('STFT Magnitude-Tracking field')
    plt.ylim([0, 10])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time(cycles)')
    plt.show()
