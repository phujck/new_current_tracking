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
    A = np.array(A)
    k = A.size
    A = np.pad(A, (0, 4 * k), 'constant')
    minus_one = (-1) ** np.arange(A.size)
    # result = np.fft.fft(minus_one * A)
    result = np.fft.fft(minus_one * A, n=k)
    minus_one = (-1) ** np.arange(result.size)
    result *= minus_one
    result *= np.exp(-1j * np.pi * A.size / 2)
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
    spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        plt.plot(w, spec[:, i], label='U/t= %.1f' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([-min_spec, spec.max()])
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
    spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 6)]
    for i, j in enumerate(U):
        print(i)
        print(i % 2)
        if i <2:
            plt.plot(w, spec[:, i], label='%s' % (j))
        else:
            plt.plot(w, spec[:, i], linestyle='dashed', label='%s' % (j))
        axes = plt.gca()
        axes.set_xlim([0, max_harm])
        axes.set_ylim([-min_spec, spec.max()])
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
# nx = 4
# ny = 0
# t = 0.191
# U = 0.1 * t
# delta = 2
# cycles = 10
params = {
    'axes.labelsize': 30,
    'legend.fontsize': 19,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'figure.figsize': [6, 6],
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
U = 0*t
U2 = 7* t
delta = 0.05
delta2 = 0.05
cycles = 10
cycles2 = 10
# field= 32.9
field = 32.9
field2=32.9
F0 = 10
a = 4
scalefactor = 1
ascale = 1
ascale2 = 1
Jscale=1

Tracking = False
Switch = False
prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
prop2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U2, t=t2, F0=F0, a=a, bc='pbc')
print(prop.field)
print(prop2.field)
if Tracking:
    prop_track = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=7 * t, t=t, F0=F0, a=ascale * a, bc='pbc')
    prop_track2 = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=0 * t, t=t, F0=F0, a=ascale2 * a,
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

    J_field_track = np.load('./data/tracking/Jfield' + newparameternames)*Jscale / scalefactor
    phi_track = np.load('./data/tracking/phi' + newparameternames)
    # phi_reconstruct = np.load('./data/tracking/phirecon' + parameternames)
    neighbour_track = np.load('./data/tracking/neighbour' + newparameternames)
    two_body_track = np.load('./data/tracking/twobody' + newparameternames)
    t_track = np.linspace(0.0, cycles, len(J_field_track))
    D_track = np.load('./data/tracking/double' + newparameternames)

    newparameternames2 = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
        nx, cycles2, U2, t2, number2, delta2, field, F0, ascale2)

    J_field_track2 = np.load('./data/tracking/Jfield' + newparameternames2) / scalefactor
    phi_track2 = np.load('./data/tracking/phi' + newparameternames2)
    # phi_reconstruct = np.load('./data/tracking/phirecon' + parameternames)
    neighbour_track2 = np.load('./data/tracking/neighbour' + newparameternames2)
    two_body_track2 = np.load('./data/tracking/twobody' + newparameternames2)
    t_track2 = np.linspace(0.0, cycles, len(J_field_track))
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
# print(darray.shape)
# t_array = t2 = np.linspace(0.0, cycles, len(darray[0, :]))
# breakline = []
# cmap = plt.get_cmap('jet_r')
# plt.plot(t_array,D)
# for xx in range(0, 11):
#     color = cmap((float(10 * xx) - 7) / 45)
#     color2 = cmap((float(10 * (xx + 1)) - 7) / 45)
#     if xx == 0 or xx == 10:
#         plt.plot(t_array, darray[xx, :], color=color, label='$\\frac{U}{t_0}=$%s' % (xx))
#     else:
#         plt.plot(t_array, darray[xx, :], color=color)
#     if xx < len(breaktimes):
#         breakindex = int(breaktimes[xx] * len(darray[0, :]) / cycles)
#         # plt.plot(breaktimes[xx], darray[xx+1,breakindex],color=color2, marker='o', markersize='10')
#         plt.plot(breaktimes[xx], darray[xx + 1, breakindex], color='black', marker='o', markersize='10')
#         breakline.append(darray[xx + 1, breakindex])
# plt.plot(breaktimes[0], darray[1, int(breaktimes[0] * len(darray[0, :]) / cycles)], color='black', marker='o',
#          markersize='10', label='$t_{th}$')
# plt.plot(breaktimes, breakline, linestyle='dashed', color='black')

# plt.plot(t_array, darray[xx,:], color=color)

plt.xlabel('Time [cycles]')
plt.ylabel('$D(t)$')
plt.legend()
plt.show()

N_old = int(cycles / (prop.freq * delta)) + 1
times = np.linspace(0, cycles / prop.freq, N_old)

D_grad = D
D_grad2=D2
if Tracking:
    D_grad_track = D_track
    D_grad_track2 = D_track2
#
# D_func = interp1d(t, D_grad, fill_value=0, bounds_error=False)
# # D_grad_track = np.gradient(D_track, delta_track)

#

plt.subplot(211)
plt.plot(t, J_field, label='Original Current')
plt.plot(t, J_field, linestyle='dashed',
         label='Tracked Current')
# plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
plt.legend()
plt.annotate('a)', xy=(0.3, np.max(J_field) - 0.05), fontsize=20)

plt.subplot(212)
plt.plot(t2, J_field2, label='Original current')
plt.plot(t2, J_field2, linestyle='dashed',
         label='Tracked current')
plt.xlabel('Time [cycles]')
plt.ylabel('$J(t)$')
plt.annotate('b)', xy=(0.3, np.max(J_field2) - 0.05), fontsize=20)
plt.show()

plt.subplot(211)
plt.plot(t, D_grad, label='original')
if Tracking:
    plt.plot(t, D_grad, label='tracked', linestyle='dashed')
    plt.plot(t, D_grad2, label='tracked', linestyle='dashed')
    plt.plot(t_track, D_grad_track, label='tracked', linestyle='dashed')
    plt.plot(t_track, D_grad_track2, label='tracked', linestyle='dashed')
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
    t=t1
    switchparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
        nx, cycles, 1*t, t, number, delta_switch, field, F0, ascale)
    J_field_switch3 = np.load('./data/switch/Jfield' + switchparameternames) / scalefactor
    phi_switch3 = np.load('./data/switch/phi' + switchparameternames)
    J_field_switch3 = J_field_switch3[:-cut]
    phi_switch3 = phi_switch3[:-cut]

    switchparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
        nx, cycles, 0.5*t, t, number, delta_switch, field, F0, ascale)
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

    plt.subplot(211)
    plt.plot(t_switch, phi_switch.real)
    plt.plot(t_switch, phi_switch2.real, linestyle=':')
    plt.plot(t_switch, phi_switch3.real, linestyle='--')
    plt.plot(t_switch, phi_switch4.real, linestyle='-.')



    # plt.annotate('b)', xy=(0.3, np.max(J_field) - 0.05), fontsize='16')
    plt.ylabel('$\Phi_J(t)$')

    plt.subplot(212)
    # plt.plot(t_switch, switch_function.real, label='Switch Function')
    plt.plot(t_switch, J_field_switch.real, label='U/t=0')
    plt.plot(t_switch, J_field_switch2.real, label='U/t=0.5', linestyle=':')
    plt.plot(t_switch, J_field_switch3.real, label='U/t=1',linestyle='--')
    plt.plot(t_switch, J_field_switch4.real, label='U/t=1.5',linestyle='-.')


    # plt.annotate('b)', xy=(0.3, np.max(J_field) - 0.05), fontsize='16')
    plt.legend()
    plt.ylabel('$J_T(t)$')
    plt.xlabel('Time [cycles]')
    plt.show()

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


plt.plot(t, phi_original.real, label='original')
# plt.plot(t2, J_field2.real, label='swapped')
if Tracking:
    plt.plot(t_track, phi_track.real, label='tracked', linestyle='dashed')
# plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('$\\Phi(t)$')
plt.yticks(np.arange(-1 * np.pi, 1 * np.pi, 0.5 * np.pi),
           [r"$" + format(r / np.pi, ".2g") + r"\pi$" for r in np.arange(-1 * np.pi, 1 * np.pi, .5 * np.pi)])
plt.show()
#
# plt.plot(t, np.abs(neighbour), label='original')
# # plt.plot(t2, J_field2.real, label='swapped')
# if Tracking:
#     plt.plot(t_track, np.abs(neighbour_track), label='tracked', linestyle='dashed')
# plt.legend()
# plt.xlabel('Time [cycles]')
# plt.ylabel('R(t)')
# plt.show()
#
# plt.plot(t, np.angle(neighbour), label='original')
# # plt.plot(t2, J_field2.real, label='swapped')
# if Tracking:
#     plt.plot(t_track, np.angle(neighbour_track), label='tracked', linestyle='dashed')
# plt.legend()
# plt.xlabel('Time [cycles]')
# plt.ylabel('$\\theta(t)$')
# plt.show()
#
# plt.plot(t, np.abs(two_body), label='original')
# # plt.plot(t2, J_field2.real, label='swapped')
# if Tracking:
#     plt.plot(t_track, np.abs(two_body_track), label='tracked', linestyle='dashed')
# plt.legend()
# plt.xlabel('Time [cycles]')
# plt.ylabel('C(t)')
# plt.show()
#
# plt.plot(t, np.angle(two_body), label='original')
# # plt.plot(t2, J_field2.real, label='swapped')
# if Tracking:
#     plt.plot(t_track, np.angle(two_body_track), label='tracked', linestyle='dashed')
# plt.legend()
# plt.xlabel('Time [cycles]')
# plt.ylabel('$\\kappa(t)$')
# plt.show()

# plt.plot(t,np.gradient(J_field,delta1))
# plt.plot(t2,np.gradient(J_field2,delta2))
# plt.show()
#
# # Phi field
# cross_times_up=[]
# cross_times_down=[]
# plt.plot(t, phi_original, label='original',linestyle='dashed')
# # for k in range (1,2):
# #     if k != 0:
# #         line=k*np.ones(len(t)) * np.pi / 2
# #         idx_pos = np.argwhere(np.diff(np.sign(phi_original - line))).flatten()
# #         idx_neg = np.argwhere(np.diff(np.sign(phi_original + line))).flatten()
# #         idx_up=min(idx_pos[0],idx_neg[0])
# #         idx_down=max(idx_pos[-1],idx_neg[-1])
# #         # idx_up=idx_up[0]
# #         # idx_down=idx_down[-1]
# #         # plt.plot(t, line, color='red')
# #         # plt.plot(t[idx],line[idx], 'ro')
# #         cross_times_up.append(idx_up)
# #         cross_times_down.append(idx_down)
# # # cross_times_up=np.concatenate(cross_times).ravel()
# # plt.plot(t[cross_times_up],phi_original[cross_times_up],'go')
# # plt.plot(t[cross_times_down],phi_original[cross_times_down],'ro')
# # for xc in cross_times_up:
# #     plt.hlines(phi_original[xc],0,t[xc],color='green', linestyle='dashed')
# # for xc in cross_times_down:
# #     plt.hlines(phi_original[xc],t[xc],t[-1],color='red', linestyle='dashed')
# # cross_times_up=(t[cross_times_up])
# # cross_times_down=(t[cross_times_down])
# # if Tracking:
# #     plt.plot(t[:J_field_track.size], phi_track, label='Tracking', linestyle='dashdot')
# # if Track_Branch:
# #     plt.plot(t[:phi_track_branch.size], phi_track_branch, label='Tracking with Branches', linestyle='dotted',color='yellow')
# # plt.plot(t, np.ones(len(t)) * np.pi / 2, color='red')
# # plt.plot(t, np.ones(len(t)) * -1 * np.pi / 2, color='red')
# # plt.yticks(np.arange(-1.5*np.pi, 2*np.pi, 0.5*np.pi),[r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(-1.5*np.pi, 2*np.pi, .5*np.pi)])
# plt.legend()
# plt.xlabel('Time [cycles]')
# plt.ylabel('$\\phi$')
# plt.show()
#
#
#
# # Double occupancy plot
# # plt.plot(t, D)
# # plt.xlabel('Time [cycles]')
# # plt.ylabel('Double occupancy')
# # plt.show()
#
#
#
# # Current gradients
two_body = np.array(two_body)

# plt.plot(two_body.real)
# plt.plot(two_body.imag)
# plt.show()
#
# plt.plot(t, np.angle(-two_body / prop.nsites))
# plt.plot(t, np.pi * np.ones(t.size))
# plt.plot(t, -np.pi * np.ones(t.size))
# plt.show()

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

if Tracking:
    two_body_track = np.array(two_body_track)
    extra_track = 2. * np.real(np.exp(-1j * phi_track) * two_body_track)

    diff_track = phi_track - np.angle(neighbour_track)
    J_grad_track = -2. * prop_track.a * prop_track.t * np.gradient(phi_track, delta_track) * np.abs(
        neighbour_track) * np.cos(diff_track)
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
    eq33_track = J_grad_track + 2. * prop_track.a * prop_track.t * (
            np.gradient(np.angle(neighbour_track), delta_track) * np.abs(neighbour_track) * np.cos(
        diff_track) - np.gradient(
        np.abs(neighbour_track), delta_track) * np.sin(diff_track))

    # Just in case you want to plot from a second simulation

    # plt.plot(t, eq33-J_grad, label='Gradient calculated via expectations', linestyle='dashdot')
    # plt.plot(t, eq32-J_grad, linestyle='dashed')
    # plt.show()
    # #

    # plot various gradient calculations
    # plt.plot(t, eq33, label='Gradient calculated via expectations', linestyle='dashdot')
    plt.subplot(311)
    plt.plot(t, eq32, label='original')
    plt.plot(t_track[:-2], eq32_track[:-2], linestyle='dashed',
             label='tracked')
    plt.ylabel('$\\dot{J}(t)$')
    plt.legend()

    plt.subplot(312)
    plt.plot(t, np.abs(neighbour), label='original')
    plt.plot(t_track, np.abs(neighbour_track), linestyle='dashed')
    plt.ylabel('$R(t)$')

    plt.subplot(313)
    plt.plot(t, np.abs(two_body), label='original')
    plt.plot(t_track, np.abs(two_body_track), linestyle='dashed')
    plt.ylabel('$C(t)$')
    plt.xlabel('Time [cycles]')

    plt.show()

#
# # error plot

#
#
# # gradient deviations
# plt.plot(t, abs(exact-eq33), label='expectation gradient deviation')
# plt.plot(t, abs(exact-eq32), label='commutator gradient deviation')
# # scaling error to see it on the same axis as the gradient deviations
# # plt.plot(t,(error-error[0])*max(abs(exact)-abs(eq32))/max(error-error[0]),label='propagator error estimation')
# # plt.plot(t,error)
# plt.legend()
# plt.show()
#
# print("average deviation from gradient when calculated via expectations")
# print(np.sqrt(np.mean((exact-eq33)**2)))
#
# print("average deviation from gradient when calculated via commutators")
# print(np.sqrt(np.mean((exact-eq32)**2)))
#
#
#
#
# # different windowing functions.
#
# # epsilon=int(t.size/30)
# # window=np.ones(t.size)
# # window[:epsilon]=np.sin(np.pi * t[:epsilon] / (2.*t_delta*epsilon)) ** 2.
# # window[:-epsilon]=np.sin(np.pi * (t[-1]-t[:-epsilon]) / (2.*t_delta*epsilon)) ** 2.
#

# window=blackman(len(J_field))
# window2=blackman(len(J_field2))
#
# n_time=J_field_track.size
# # plot the spectrum.
# xlines=[2*i-1 for i in range(1,15)]
# plt.semilogy(omegas, (abs(FT(np.gradient(J_field,delta1) * window)) ** 2), label='$\\frac{U}{t_0}=5$')
# plt.semilogy(omegas2, (abs(FT(np.gradient(J_field2,delta2) * window2)) ** 2), label='$\\frac{U}{t_0}=0.1$')
#
# for xc in xlines:
#         plt.axvline(x=xc, color='black', linestyle='dashed')
# if Tracking:
#     plt.semilogy(omegas, abs(FT(np.gradient(J_field_track[:n_time],delta_track) * blackman(n_time))) ** 2,
#                  label='Tracking')
# if Track_Branch:
#     plt.semilogy(omegas, abs(FT(np.gradient(J_field_track_branch[:prop.n_time]) * blackman(prop.n_time))) ** 2,
#                  label='Tracking With Branches')
# plt.legend()
# plt.title("output dipole acceleration")
# plt.xlim([0, 60])
# plt.xlabel('$\\frac{\omega}{\omega_0}$')
# plt.ylim([1e-8,1e5])
# plt.show()

method = 'welch'

min_spec = 14
max_harm = 60
gabor = 'fL'
spec = np.zeros((FT_count(len(J_field)), 2))

w1,spec1=har_spec.spectrum_welch(exact, delta1)
w2,spec2=har_spec.spectrum_welch(exact2, delta2)
plt.semilogy(w1, spec1, label='$D^{(0)}(t)$')
plt.semilogy(w2, spec2, label='$D^{(7)}(t)$')
plt.show()


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
plot_spectra([2 * U, int(2 * U2)], w, spec, min_spec, max_harm)

if Tracking:
    plt.plot(t_array, darray[0, :], label='$D^{(0)}(t)$')
    plt.plot(t, D_grad2, label='$D^{(7)}(t)$')
    plt.plot(t_track, D_grad_track2, label='$D_T^{(0)}(t)$', linestyle='dashed')
    plt.plot(t_track, D_grad_track, label='$D_T^{(7)}(t)$', linestyle='dashed')
    plt.ylabel('$D(t)$')
    plt.xlabel('Time [cycles]')
    plt.legend()
    plt.show()


    spec = np.zeros((FT_count(len(exact)), 4))
    if method == 'welch':
        w, spec[:, 0] = har_spec.spectrum_welch(exact, delta1)
        w2, spec[:, 1] = har_spec.spectrum_welch(exact2, delta2)
        w3, spec[:, 2] = har_spec.spectrum_welch(exact_track2, delta_track2)
        w4, spec[:, 3] = har_spec.spectrum_welch(exact_track, delta_track)
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
                        '$\mathcal{F}\left(\\frac{{\\rm d}J_T^{(7)}}{{\\rm d} t}\\right)$'], w, spec, min_spec, max_harm)

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


"""Commented stuff not necessary atm"""

# spec = np.log10(spec)
# xlines = [2 * i - 1 for i in range(1, 15)]
# for i, j in enumerate(U):
#     plt.plot(w, spec[:, i], label='U/t= %.1f' % (j))
#     axes = plt.gca()
#     axes.set_xlim([0, max_harm])
#     axes.set_ylim([-min_spec, spec.max()])
# for xc in xlines:
#     plt.axvline(x=xc, color='black', linestyle='dashed')
#     plt.xlabel('Harmonic Order')
#     plt.ylabel('HHG spectra')
# plt.legend(loc='upper right')
# plt.show()



# plt.semilogy(omegas, abs(FT(phi_original[:prop.n_time] * blackman(prop.n_time))) ** 2, label='original')
# if Tracking:
#     plt.semilogy(omegas, abs(FT(phi_track[:prop.n_time]* blackman(prop.n_time))) ** 2, label='Tracking')
#     for xc in xlines:
#         plt.axvline(x=xc, color='black', linestyle='dashed')
# if Track_Branch:
#     plt.semilogy(omegas, abs(FT(phi_track_branch[:prop.n_time] * blackman(prop.n_time))) ** 2,
#                  label='Tracking With Branch')
# plt.legend()
# plt.title("input-field")
# plt.xlim([0, 30])
#
# plt.show()
#
# Y, X, Z1 = stft(phi_original.real, 1, nperseg=prop.n_time/10, window=('gaussian', 2/prop.field))
# Z1 = np.abs(Z1) ** 2
# # plt.pcolormesh(X*delta, Y*omegas.max()/Y.max(), Z1, norm=colors.LogNorm())
# plt.pcolormesh(X * delta, Y * omegas.max() / Y.max(), Z1)
#
# plt.title('STFT Magnitude-Tracking field')
# plt.ylim([0, 10])
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time(cycles)')
# plt.show()
#
#
# if Tracking:
#     Y, X, Z1 = stft(phi_track.real, 1, nperseg=100, nfft=omegas.size/2, window=('gaussian', 2/(prop.field)))
#     print(Y)
#     Z1 = np.abs(Z1)**2
#     plt.pcolormesh(X*delta, Y*omegas.max()/Y.max(), Z1, norm=colors.PowerNorm(gamma=0.85))
#     plt.ylim(0,8)
#     # plt.pcolormesh(X * delta, Y * omegas.max() / Y.max(), Z1,cmap='plasma')
#     plt.colorbar()
#     for xc in cross_times_up:
#         plt.axvline(x=xc, color='green', linestyle='dashed')
#     for xc in cross_times_down:
#         plt.axvline(x=xc, color='red', linestyle='dashed')
#     plt.title('STFT Magnitude-Tracking field')
#     plt.ylim([0, 10])
#     plt.ylabel('Frequency/$\\omega_0$')
#     plt.xlabel('Time(cycles)')
#     plt.show()
#
# if Track_Branch:
#     Y, X, Z1 = stft((phi_track_branch).real, 1, nperseg=100, nfft=omegas.size/2, window=('gaussian', 2/(prop.field)))
#     Z1 = np.abs(Z1)**2
#     plt.pcolormesh(X*delta, Y*omegas.max()/Y.max(), Z1, norm=colors.PowerNorm(gamma=0.6))
#     # plt.pcolormesh(X * delta, Y * omegas.max() / Y.max(), Z1,cmap='plasma')
#     plt.colorbar()
#     plt.axvline(x=time1, color='black', linestyle='dashed')
#     for xc in cross_times_up:
#         if xc > time1:
#             plt.axvline(x=xc, color='green', linestyle='dashed')
#     for xc in cross_times_down:
#         if xc >time1:
#             plt.axvline(x=xc, color='red', linestyle='dashed')
#
#     plt.title('STFT Magnitude-Tracking with branch cut')
#     plt.ylim([0, 15])
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time(cycles)')
#     plt.show()

# if Tracking:
#     alist = [20,25,30]
#     for j in alist:
#         k = j * a
#         print(k)
#         prop_track = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=5 * t1, t=t1, F0=F0, a=k,
#                               bc='pbc')
#         newparameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
#             nx, cycles, U, t1, number, delta, field, F0, j)
#         D_track = np.load('./data/tracking/double' + newparameternames)
#         phi_track = np.load('./data/tracking/phi' + newparameternames)
#         J_field_track = np.load('./data/tracking/Jfield' + newparameternames) / scalefactor
#         # phi_reconstruct = np.load('./data/tracking/phirecon' + parameternames)
#         neighbour_track = np.load('./data/tracking/neighbour' + newparameternames)
#         two_body_track = np.load('./data/tracking/twobody' + newparameternames)
#         t_track = np.linspace(0.0, cycles, len(J_field_track))
#
#         extra_track = 2. * np.real(np.exp(-1j * phi_track) * two_body_track)
#
#         diff_track = phi_track - np.angle(neighbour_track)
#         J_grad_track = -2. * prop_track.a * prop_track.t * np.gradient(phi_track, delta_track) * np.abs(
#             neighbour_track) * np.cos(diff_track)
#         exact_track = np.gradient(J_field_track, delta_track)
#         #
#         #
#         #
#         eq32_track = (J_grad_track - prop_track.a * prop_track.t * prop_track.U * extra_track) / scalefactor
#         print('lattice constant')
#         print(prop_track.U)
#         plt.subplot(3, 1, 1)
#         plt.plot(t_track, D_track, label='$a_s$= %s $a$' % (j))
#         plt.ylabel('$D(t)$')
#         plt.subplot(3, 1, 2)
#         plt.plot(t_track, phi_track, label='a= %s' % (j))
#         plt.ylabel('$\\Phi(t)$')
#         plt.subplot(3, 1, 3)
#         plt.plot(t_track, 100*(exact_track - eq32_track)/np.max(exact_track),
#                  label='$a_s$= %s$a$' % (j))
#         plt.xlabel('Time [cycles]')
#         plt.legend()
#
#         # plt.ylim([-1,1])
#         plt.ylabel('E. Div. \%')
#
#     plt.show()
#
# prop_track = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=5 * t1, t=t1, F0=F0, a=ascale * a,
#                       bc='pbc')
# if Tracking:
#     # parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
#     # nx, cycles, U, t, number, delta, field, F0)
#     parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude-%s-ascale.npy' % (
#         nx, cycles, U, t, number, delta, field, F0, ascale)
#
#     J_field_track = np.load('./data/tracking/Jfield' + newparameternames) / scalefactor
#     phi_track = np.load('./data/tracking/phi' + newparameternames)
#     # phi_reconstruct = np.load('./data/tracking/phirecon' + parameternames)
#     neighbour_track = np.load('./data/tracking/neighbour' + newparameternames)
#     two_body_track = np.load('./data/tracking/twobody' + newparameternames)
#     t_track = np.linspace(0.0, cycles, len(J_field_track))
#     D_track = np.load('./data/tracking/double' + newparameternames)

#
# plt.plot(t, D, label='$L=6$')
# plt.plot(t2, D2, label='$L=10$')
# plt.ylabel('$D(t)$')
# plt.xlabel('Time [cycles]')
# plt.legend()
# # plt.show()


"""Calculating energy expectations"""
# plt.plot(t, energy.real, label='$\\frac{U}{t_0}=0$')
# plt.plot([0,t[-1]],[energy.real[0],energy.real[-1]])
# plt.plot(t, energy2.real, label='$\\frac{U}{t_0}=6$')
# plt.plot(t,-8*prop.t*np.cos(-phi_original), linestyle='dashed', label='Guess')
# plt.ylabel('$H(t)$')
# plt.xlabel('Time [cycles]')
# plt.legend()
# plt.show()
#
#
# plt.plot(t, energy.real/4, label='4 sites')
# plt.plot(t, energy2.real/6, label='6 sites')
# plt.plot(t,-4*prop.t*np.cos(-phi_original)/4, linestyle='dashed', label='Guess (4 sites)')
# plt.plot(t,-8*prop.t*np.cos(-phi_original)/6, linestyle='dashed', label='Guess (6 sites)')
# plt.ylabel('$H(t)$')
# plt.xlabel('Time [cycles]')
# plt.legend()
# plt.show()
#
# plt.subplot(311)
# plt.plot(t, doublon_energy_L.real-energy.real, label='$\\frac{U}{t_0}=0$')
# plt.plot(t, doublon_energy_L2.real-energy2.real, label='$\\frac{U}{t_0}=6$')
# plt.legend()
# plt.ylabel('$\Delta_{M}(t)$')
#
#
# plt.subplot(312)
# plt.plot(t, doublon_energy.real-energy.real, label='$\\frac{U}{t_0}=0$')
# plt.plot(t, doublon_energy2.real-energy2.real, label='$\\frac{U}{t_0}=6$')
# plt.ylabel('$\Delta_{D}(t)$')
#
# plt.subplot(313)
# plt.plot(t, singlon_energy.real-energy.real, label='$\\frac{U}{t_0}=0$')
# plt.plot(t, singlon_energy2.real-energy2.real, label='$\\frac{U}{t_0}=6$')
# plt.ylabel('$\Delta_{S}(t)$')
# plt.xlabel('Time [cycles]')
# plt.show()
#
#
#
# plt.subplot(311)
# plt.plot(t, doublon_energy_L.real, label='$\\frac{U}{t_0}=0$')
# plt.plot(t, doublon_energy_L2.real, label='$\\frac{U}{t_0}=6$')
# plt.legend()
# plt.ylabel('$\Delta_{M}(t)+H(t)$')
#
#
# plt.subplot(312)
# plt.plot(t, doublon_energy.real, label='$\\frac{U}{t_0}=0$')
# plt.plot(t, doublon_energy2.real, label='$\\frac{U}{t_0}=6$')
# plt.ylabel('$\Delta_{D}(t)+H(t)$')
#
# plt.subplot(313)
# plt.plot(t, singlon_energy.real, label='$\\frac{U}{t_0}=0$')
# plt.plot(t, singlon_energy2.real, label='$\\frac{U}{t_0}=6$')
# plt.ylabel('$\Delta_{S}(t)+H(t)$')
# plt.xlabel('Time [cycles]')
# plt.show()
# print(np.mean(doublon_energy_L.real-energy.real))
# print(np.mean(doublon_energy.real-energy.real))
# print(np.mean(singlon_energy.real-energy.real))
#
#
# print(np.mean(doublon_energy_L2.real-energy2.real))
# print(np.mean(doublon_energy2.real-energy2.real))
# print(np.mean(singlon_energy2.real-energy2.real))
#
#
#
#
# plt.plot(t, doublon_energy.real-energy.real, label='$\\frac{U}{t_0}=0$')
# plt.plot(t, doublon_energy2.real-energy2.real, label='$\\frac{U}{t_0}=6$')
#
# plt.ylabel('$\Delta_{L^2}(t)$')
# plt.xlabel('Time [cycles]')
# plt.legend()
# plt.show()
