import numpy as np
import matplotlib.pyplot as plt
import definition as hams
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
    k=A.size
    A=np.pad(A, (0, 4*k), 'constant')
    minus_one = (-1) ** np.arange(A.size)
    # result = np.fft.fft(minus_one * A)
    result = np.fft.fft(minus_one * A,n=k)
    minus_one = (-1) ** np.arange(result.size)
    result *= minus_one
    result *= np.exp(-1j * np.pi * A.size / 2)
    return result

def smoothing(A, b=1,c=5,d=0):
    if b==1:
       b= int(A.size /50)
    if b % 2 == 0:
        b=b+1
    j=savgol_filter(A, b, c,deriv=d)
    return j

def current(sys, phi, neighbour):
    conjugator = np.exp(-1j * phi) * neighbour
    c =sys.a * sys.t * 2*np.imag(conjugator)
    return c

Tracking=False
Track_Branch=False

def plot_spectra(U,w,spec,min_spec,max_harm):
    spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 15)]
    for i,j in enumerate(U):
        plt.plot(w,spec[:,i],label='U/t= %.1f' %(j))
        axes=plt.gca()
        axes.set_xlim([0,max_harm])
        axes.set_ylim([-min_spec,spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()

def FT_count(N):
    if N%2==0:
        return int(1+N/2)
    else:
        return int((N+1)/2)

# These are Parameters I'm using
# number=2
# nelec = (number, number)
# nx = 4
# ny = 0
# t = 0.191
# U = 0.1 * t
# delta = 2
# cycles = 10

# Load parameters and data. 2 suffix is for loading in a different simulation for comparison
number=3
number2=3
nelec = (number, number)
nx = 6
nx2=6
ny = 0
t = 0.52
t2=0.52
U = 5*t
U2= 5*t
delta = 0.05
delta2=0.05
cycles = 1
cycles2=1
# field= 32.9
field=32.9
F0=10
a=4

Tracking=True
prop = hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
prop_track=hams.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
# factor=prop.factor
delta1=delta
delta_track=prop_track.freq*delta/prop.freq


# load files
parameternames='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (nx,cycles,U,t,number,delta,field,F0)
J_field=np.load('./data/original/Jfield'+parameternames)
phi_original=np.load('./data/original/phi'+parameternames)
phi_reconstruct = np.load('./data/original/phirecon'+parameternames)
neighbour=np.load('./data/original/neighbour'+parameternames)
two_body=np.load('./data/original/twobody'+parameternames)
# two_body_old=np.load('./data/original/twobodyold'+parameternames)
D=np.load('./data/original/double'+parameternames)

error=np.load('./data/original/error'+parameternames)

parameternames2='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (nx2,cycles2,U2,t2,number2,delta2,field,F0)

J_field2=np.load('./data/original/Jfield'+parameternames2)
two_body2=np.load('./data/original/twobody'+parameternames2)
neighbour2=np.load('./data/original/neighbour'+parameternames2)
phi_original2=np.load('./data/original/phi'+parameternames2)
error2=np.load('./data/original/error'+parameternames2)

if Tracking:
    parameternames = '-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude.npy' % (
    nx, cycles, U, t, number, delta, field, F0)
    J_field_track = np.load('./data/tracking/Jfield' + parameternames)
    phi_track = np.load('./data/tracking/phi' + parameternames)
    # phi_reconstruct = np.load('./data/tracking/phirecon' + parameternames)
    neighbour_track = np.load('./data/tracking/neighbour' + parameternames)
    two_body_track = np.load('./data/tracking/twobody' + parameternames)
    t_track = np.linspace(0.0, cycles, len(J_field_track))
    D_track = np.load('./data/tracking/double' + parameternames)


omegas = (np.arange(len(J_field)) - len(J_field) / 2) /cycles
omegas2 = (np.arange(len(J_field2)) - len(J_field2) / 2) /cycles

# delta2=delta2*factor

t = np.linspace(0.0, cycles, len(J_field))
t2=np.linspace(0.0, cycles, len(J_field2))

# smoothing- don't use.
# J_field=smoothing(J_field)
# J_field2=smoothing(J_field2)
# neighbour_real=smoothing(neighbour.real)
# neighbour_imag=smoothing(neighbour.imag)
# two_body_imag=smoothing(two_body.imag)
# two_body_real=smoothing(two_body.real)
# neighbour=np.array(neighbour_real+1j*neighbour_imag)
# two_body=np.array(two_body_real+1j*two_body_imag)



# Plot the current expectation
plt.plot(t, J_field.real, label='$\\frac{U}{t_0}=5$')
plt.plot(t2, J_field2.real, label='$\\frac{U}{t_0}=0.1$')
plt.show()

#
# # # Double occupancy plot
# plt.plot(t, D.real)
# plt.plot(t, D.imag)
# plt.xlabel('Time [cycles]')
# plt.ylabel('Double occupancy')
# plt.show()
N_old = int(cycles/(prop.freq*delta))+1
times=np.linspace(0,cycles/prop.freq,N_old)

# plt.plot(t,two_body.real,label='Original')
# if Tracking:
#     plt.plot(t_track,two_body_track.real,label='Tracking')
# plt.title('two body expectation real')
# plt.legend()
# plt.show()
#
#
# plt.plot(t,two_body.imag,label='Original')
# if Tracking:
#     plt.plot(t_track,two_body_track.imag,label='Tracking')
# plt.title('two body expectation imaginary')
# plt.legend()
# plt.show()


D_grad=np.gradient(D,delta)

D_func = interp1d(t, D_grad, fill_value=0, bounds_error=False)
#
# D_grad=D
# D_grad_track=D_track
#
# D_func = interp1d(t, D_grad, fill_value=0, bounds_error=False)



plt.plot(t,D_grad, label='original')
plt.plot(t,D_func(t), label='original')
if Tracking:
    D_grad_track = np.gradient(D_track, delta_track)
    plt.plot(t_track,D_grad_track,label='tracking')
plt.xlabel('Time [cycles]')
plt.ylabel('Double occupancy gradient')
plt.legend()
plt.show()

plt.plot(t, J_field.real, label='original')
# plt.plot(t2, J_field2.real, label='swapped')
if Tracking:
    plt.plot(t_track,J_field_track.real,label='tracked',linestyle='dashed')
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('Current expectation')
plt.show()

plt.plot(t, np.gradient(J_field.real,delta1), label='original')
# plt.plot(t2, J_field2.real, label='swapped')
if Tracking:
    plt.plot(t_track,np.gradient(J_field_track.real,delta_track),label='tracked',linestyle='dashed')
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('Current expectation gradients')
plt.show()


plt.plot(t, phi_original.real, label='original')
# plt.plot(t2, J_field2.real, label='swapped')
if Tracking:
    plt.plot(t_track,phi_track.real,label='tracked',linestyle='dashed')
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('$\\phi$')
plt.show()

plt.plot(t, neighbour.real, label='original')
# plt.plot(t2, J_field2.real, label='swapped')
if Tracking:
    plt.plot(t_track,neighbour_track.real,label='tracked',linestyle='dashed')
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('nearest-neighbour expectations(real)')
plt.show()


plt.plot(t, neighbour.imag, label='original')
# plt.plot(t2, J_field2.real, label='swapped')
if Tracking:
    plt.plot(t_track,neighbour_track.imag,label='tracked',linestyle='dashed')
plt.legend()
plt.xlabel('Time [cycles]')
plt.ylabel('nearest-neighbour expectations(imaginary)')
plt.show()



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
extra = 2. * np.real(np.exp(-1j * phi_original)*two_body)


diff = phi_original - np.angle(neighbour)
two_body2 = np.array(two_body2)
extra2 = 2. * np.real(np.exp(-1j * phi_original2)*two_body2)
diff2 = phi_original2 - np.angle(neighbour2)
J_grad = -2. * prop.a * prop.t * np.gradient(phi_original, delta) * np.abs(neighbour) * np.cos(diff)
J_grad2 = -2. * prop.a * prop.t * np.gradient(phi_original2, delta2) * np.abs(neighbour2) * np.cos(diff2)
exact=np.gradient(J_field,delta1)
exact2=np.gradient(J_field2,delta2)
#
#
#
# # eq 32 should have a minus sign on the second term, but
eq32 = J_grad -prop.a * prop.t *prop.U* extra
# eq32= -prop.a * prop.t * prop.U * extra
eq33 = J_grad + 2. * prop.a * prop.t * (np.gradient(np.angle(neighbour), delta1) * np.abs(neighbour) * np.cos(diff) - np.gradient(
    np.abs(neighbour), delta1) * np.sin(diff))

# Just in case you want to plot from a second simulation

eq32_2= J_grad2 + prop.a * t2*U2 * extra2/prop.factor
eq33_2 = J_grad2 + 2. * prop.a * prop.t * (np.gradient(np.angle(neighbour2), delta2) * np.abs(neighbour2) * np.cos(diff2) - np.gradient(
    np.abs(neighbour2), delta2) * np.sin(diff2))

# plt.plot(t, eq33-J_grad, label='Gradient calculated via expectations', linestyle='dashdot')
# plt.plot(t, eq32-J_grad, linestyle='dashed')
# plt.show()
# #

# plot various gradient calculations
# plt.plot(t, eq33, label='Gradient calculated via expectations', linestyle='dashdot')
plt.plot(t, exact, label='Numerical current gradient')
plt.plot(t,eq32, linestyle='dashed',
         label='Gradient using commutators')
plt.xlabel('Time [cycles]')
plt.ylabel('Current Expectation Gradient')
plt.legend()
plt.show()


if Tracking:
    two_body_track = np.array(two_body_track)
    extra_track = 2. * np.real(np.exp(-1j * phi_track) * two_body_track)

    diff_track = phi_track - np.angle(neighbour_track)
    J_grad_track = -2. * prop_track.a * prop_track.t * np.gradient(phi_track, delta_track) * np.abs(neighbour_track) * np.cos(diff_track)
    exact_track = np.gradient(J_field_track, delta_track)
    #
    #
    #
    # # eq 32 should have a minus sign on the second term, but
    eq32_track = J_grad_track - prop_track.a * prop_track.t * prop_track.U * extra_track
    # eq32= -prop.a * prop.t * prop.U * extra
    eq33_track = J_grad_track + 2. * prop_track.a * prop_track.t * (
                np.gradient(np.angle(neighbour_track), delta_track) * np.abs(neighbour_track) * np.cos(diff_track) - np.gradient(
            np.abs(neighbour_track), delta_track) * np.sin(diff_track))

    # Just in case you want to plot from a second simulation



    # plt.plot(t, eq33-J_grad, label='Gradient calculated via expectations', linestyle='dashdot')
    # plt.plot(t, eq32-J_grad, linestyle='dashed')
    # plt.show()
    # #

    # plot various gradient calculations
    # plt.plot(t, eq33, label='Gradient calculated via expectations', linestyle='dashdot')
    plt.plot(t_track, exact_track, label='Numerical current gradient')
    plt.plot(t_track, eq32_track, linestyle='dashed',
             label='Gradient using commutators')
    plt.xlabel('Time [cycles]')
    plt.ylabel('Current Expectation Gradient-tracking')
    plt.legend()
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

method='welch'

min_spec=9
max_harm=50
gabor='fL'
spec = np.zeros((FT_count(len(J_field)), 2))
if method=='welch':
    w, spec[:,0] = har_spec.spectrum_welch(exact, delta1)
    w2, spec[:,1] = har_spec.spectrum_welch(exact2, delta2)
elif method=='hann':
    w, spec[:,0] = har_spec.spectrum_hanning(exact, delta1)
    w2, spec[:,0] = har_spec.spectrum_hanning(exact2, delta2)
elif method=='none':
    w, spec[:,0]=har_spec.spectrum(exact, delta1)
    w2, spec[:,0]=har_spec.spectrum(exact2, delta2)
else:
    print('Invalid spectrum method')
#converts frequencies to harmonics
w *= 2.*np.pi/prop.field
plot_spectra([2*U,int(2*U2)], w, spec, min_spec, max_harm)


# This stuff isn't needed right now


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

