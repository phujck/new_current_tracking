import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import evolve as evolve 
import observable as observable
import definition as harmonic 
import hub_lats as hub
import harmonic as har_spec
import des_cre as dc
from matplotlib import cm as cm
from scipy.integrate import ode
import des_cre as dc

#NOTE: time is inputted and plotted in terms of cycles, but the actual propagation happens in 'normal' time

#this plots double occupancy, spin-correlation, overlap probability with initial state and energy
#K is the fraction of timesteps in which the observables are calculated
def plot_observables(lat, delta=0.01, time=5., K=0.1):
    N = int(time/(lat.freq*delta)) + 1
    h = hub.create_1e_ham(lat,True)
    D = []
    eta = []
    over = []
    energies = []
    s = harmonic.hubbard(lat)[1]
    psi = np.copy(s)
    print('\n')
    for i in range(N):
        harmonic.progress(N, i)
        psi = evolve.RK4(lat, h, delta, i*delta, psi,time)
        if i%(1/K)==0:
            D.append(observable.DHP(lat, psi))
            eta.append(observable.spin(lat, psi))
            over.append(observable.overlap(lat, s ,psi)[0])
            energies.append(observable.energy(psi, lat, h, i*delta, time))
            
    t = np.linspace(0.0, time, len(D))
    plt.plot(t, D)
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Doublon-Hole Pairs')
    plt.show()
    plt.plot(t, eta)
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Spin-Spin Correlation')
    plt.show()
    plt.plot(t, over)
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Ground-State Population')
    plt.show()
    plt.plot(t, energies)
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Energy')
    plt.show()
    return

#plots the spectra with U/t values U = [...]
def plot_spectra(U,w,spec,min_spec,max_harm):
    spec = np.log10(spec)
    xlines = [2 * i - 1 for i in range(1, 15)]
    for i,j in enumerate(U):
        plt.plot(w,spec[:,i],label='U/t='+str(j))
        axes=plt.gca()
        axes.set_xlim([0,max_harm])
        axes.set_ylim([-min_spec,spec.max()])
    for xc in xlines:
        plt.axvline(x=xc, color='black', linestyle='dashed')
        plt.xlabel('Harmonic Order')
        plt.ylabel('HHG spectra')
    plt.legend(loc='upper right')
    plt.show()
    return

def plot_spectrogram(t,w,spec,min_spec=11,max_harm=60):
    w = w[w<=max_harm]
    t, w = np.meshgrid(t,w)
    spec = np.log10(spec[:len(w)])
    specn = ma.masked_where(spec<-min_spec,spec)
    cm.RdYlBu_r.set_bad(color='white', alpha=None)
    plt.pcolormesh(t,w,specn,cmap='RdYlBu_r')
    plt.colorbar()
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Harmonic Order')
    plt.title('Time-Resolved Emission')
    plt.show()
    return

def FT_count(N):
    if N%2==0:
        return int(1+N/2)
    else:
        return int((N+1)/2)
    
#this plots the current, dipole acceleration, spectrum and spectrogram
def spectra(lat, initial=None, delta=0.01, time=5., method='welch', min_spec=7, max_harm=50, gabor='wL'):
    #window() generates the gaussian window used in the Gabor transform 
    def window(tau,tau2):
        if gabor=='wL':
            return np.exp(-9.*lat.field**2*(tau-tau2)**2.)
        elif gabor=='fL':
            return np.exp(-9.*lat.freq**2*(tau-tau2)**2.)
        
    N = int(time/(lat.freq*delta))+1 
    h = hub.create_1e_ham(lat,True)
    
    if initial==None:
        psi_temp = harmonic.hubbard(lat)[1].astype(complex)
    else:
        psi_temp = initial
        
    eJ = []
    eJ2= []
    nforward=[]
    nbackward=[]
    noriginal=[]
    two_body=[]
    two_body_old=[]
    print('\nCalculating Spectrum...')
    for k in range(N):
        harmonic.progress(N,k)
        psi_temp = evolve.RK4(lat, h, delta, k*delta, psi_temp, time)
        # eJ.append(har_spec.J_expectation(lat, h, psi_temp, k*delta, time))
        # neighbour = har_spec.nearest_neighbour(lat, psi_temp)

        # this calculates the nearest neighbour expectation using
        neighbour_old=har_spec.nearest_neighbour(lat, psi_temp).conj()
        # neighbour_old=har_spec.nearest_neighbour(lat, psi_temp)
        noriginal.append(neighbour_old)
        neighbour = har_spec.nearest_neighbour_new(lat, h, psi_temp)
        nforward.append(neighbour)
        neighbour_2=har_spec.nearest_neighbour_new_2(lat, h, psi_temp)
        nbackward.append(neighbour_2)
        two_body.append(har_spec.two_body(lat, h, psi_temp.real, psi_temp.imag))
        two_body_old.append(har_spec.two_body_old(lat, psi_temp))
        eJ.append(har_spec.J_expectation2(lat, h, psi_temp, k*delta, time, neighbour_old))
        eJ2.append(har_spec.J_expectation(lat, h, psi_temp, k*delta, time))
        # eJ.append(har_spec.J_expectation3(lat, h, psi_temp, k*delta, time, neighbour,neighbour_2))


    at = np.gradient(eJ, delta)
    nforward=np.array(nforward)
    nbackward=np.array(nbackward)
    noriginal=np.array(noriginal)
    two_body=np.array(two_body)
    two_body_old=np.array(two_body_old)
    
    #plots current
    t = np.linspace(delta, time, len(eJ))
    plt.plot(t, eJ)
    plt.plot(t, eJ2)
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Current')
    plt.show()

    plt.plot(t, two_body.real)
    plt.plot(t, two_body_old.real)
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Current')
    plt.show()

    plt.plot(t, two_body.imag)
    plt.plot(t, two_body_old.imag)
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Current')
    plt.show()

    plt.plot(t, nforward.real, label='$h_1$')
    plt.plot(t, nbackward.real,label='$h_2$')
    plt.plot(t, noriginal.real, label='$h_3$', linestyle='dashed')
    plt.plot(t, ((nforward+nbackward)/2).real, label='$(h_1+h_2)/2$',linestyle='dotted')
    plt.xlabel('Time [Cycles]')
    plt.ylabel('$\\rm{Re}(\left<c^\dagger_j c_{j+1}\\right>$')
    plt.legend()
    plt.show()

    plt.plot(t, nforward.imag,label='$h_1$')
    plt.plot(t, nbackward.imag, label='$h_2^\dagger$')
    plt.plot(t, noriginal.imag, label='$h_3$',linestyle='dashed')
    plt.plot(t, ((nforward+nbackward)/2).imag, label='$(h_1+h_2^\dagger)/2$',linestyle='dotted')
    plt.plot(t, ((nforward-nbackward)/2).imag, label='$(h_1-h_2^\dagger)/2$',linestyle='dotted')
    plt.legend()
    plt.xlabel('Time [Cycles]')
    plt.ylabel('$\\rm{Im}(\left<c^\dagger_j c_{j+1}\\right>$')
    plt.show()
    #plots dipole acceleration
    t = np.linspace(delta, time, len(at))
    plt.plot(t, at)
    plt.xlabel('Time [Cycles]')
    plt.ylabel('Dipole Acceleration')
    plt.show()
    
    spec = np.zeros((FT_count(len(eJ)), 1))
    if method=='welch':
        w, spec[:,0] = har_spec.spectrum_welch(at, delta)
    elif method=='hann':
        w, spec[:,0] = har_spec.spectrum_hanning(at, delta)
    elif method=='none':
        w, spec[:,0]=har_spec.spectrum(at, delta)
    else:
        print('Invalid spectrum method')
    #converts frequencies to harmonics
    w *= 2.*np.pi/lat.field
    plot_spectra([lat.U], w, spec, min_spec, max_harm)

    
    # spec = np.zeros((FT_count(len(at)), len(at)))
    # t = np.linspace(delta, time/lat.freq, len(at))
    # print('\nProducing Spectrogram...')
    # for j,k in enumerate(t):
    #     #a(w,tau)=FT[a(t)*g(t-tau)]
    #     nat = at*np.array([window(k,i) for i in t])
    #     if method=='welch':
    #         w, spec[:,j] = har_spec.spectrum_welch(nat, delta)
    #     elif method=='hann':
    #         w, spec[:,j] = har_spec.spectrum_hanning(nat, delta)
    #     elif method=='none':
    #         w, spec[:,j] = har_spec.spectrum(nat, delta)
    #     else:
    #         print('Invalid spectrum method')
    #
    # t = np.linspace(delta, time, len(at))
    # #converts frequencies to harmonics
    # w *= 2.*np.pi/lat.field
    # if method=='welch':
    #     w = w[2:]
    #     spec = spec[2:]
    # plot_spectrogram(t, w, spec, min_spec, max_harm)
    return 

#input units: THz (field), eV (t, U), MV/cm (peak amplitude, F0), Angstroms (lattice cst, a) 
#they're then converted to t-normalised atomic units. bc='pbc' for periodic and 'abc' for antiperiodic 


neighbour = []
neighbour_check=[]
energy=[]
doublon_energy=[]
phi_original = []
J_field = []
phi_reconstruct = [0., 0.]
boundary_1 = []
boundary_2 = []
two_body = []
two_body_old=[]
error=[]
D=[]
X=[]

number=3
nelec = (number, number)
nx = 6
ny = 0
t = 0.52
# t=1.91
# t=1
U = 0.1*t
delta = 0.05
cycles = 10
# field= 32.9
field=32.9
F0=10
a=4


parameternames='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude' % (nx,cycles,U,t,number,delta,field,F0)

lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
print('\n')
print(vars(lat))
time=cycles
psi_temp = harmonic.hubbard(lat)[1].astype(complex)
init=psi_temp
h= hub.create_1e_ham(lat,True)
N = int(time/(lat.freq*delta))+1
print(N)
# for k in range(N):
#     harmonic.progress(N, k)
#     psi_old=psi_temp
#     psierror=evolve.f(lat,evolve.ham1(lat,h,k*delta,time),psi_old)
#     neighbour.append(har_spec.nearest_neighbour_new(lat, h, psi_temp))
#     J_field.append(har_spec.J_expectation(lat, h, psi_temp, k * delta, time))
#     phi_original.append(har_spec.phi(lat,k*delta,time))
#     two_body.append(har_spec.two_body_old(lat, psi_temp))
#     two_body_old.append(har_spec.two_body_old(lat,psi_temp))
#     D.append(observable.DHP(lat, psi_temp))
#
#     psi_temp = evolve.RK4(lat, h, delta, k * delta, psi_temp, time)
#     diff = (psi_temp - psi_old) / delta
#     newerror = np.linalg.norm(diff + 1j * psierror)
#     error.append(newerror)
# #
#
# np.save('./data/original/Jfield'+parameternames,J_field)
# np.save('./data/original/phi'+parameternames,phi_original)
# np.save('./data/original/phirecon'+parameternames,phi_reconstruct)
# # np.save('./data/original/boundary1'+parameternames,boundary_1)
# # np.save('./data/original/boundary2'+parameternames,boundary_2)
# np.save('./data/original/neighbour'+parameternames,neighbour)
# np.save('./data/original/twobody'+parameternames,two_body)
# np.save('./data/original/twobodyold'+parameternames,two_body_old)
# np.save('./data/original/error'+parameternames,error)
# np.save('./data/original/double'+parameternames,D)



prop=lat
r = ode(evolve.integrate_f).set_integrator('zvode', method='bdf')
r.set_initial_value(psi_temp, 0).set_f_params(lat,time,h)
branch = 0
delta=delta
while r.successful() and r.t < time/lat.freq:
    oldpsi=psi_temp
    r.integrate(r.t + delta)
    psi_temp = r.y
    newtime = r.t
    # add to expectations

    # double occupancy fails for anything other than half filling.
    # D.append(evolve.DHP(prop,psi))
    harmonic.progress(N, int(newtime / delta))
    # psierror=evolve.f(lat,evolve.ham1(lat,h,newtime,time),oldpsi)
    # diff = (psi_temp - oldpsi) / delta
    # newerror = np.linalg.norm(diff + 1j * psierror)
    # error.append(newerror)
    neighbour.append(har_spec.nearest_neighbour_new(lat, h, psi_temp))
    # neighbour_check.append(har_spec.nearest_neighbour(lat, psi_temp))
    # X.append(observable.overlap(lat, psi_temp)[1])
    J_field.append(har_spec.J_expectation(lat, h, psi_temp, newtime, time))
    phi_original.append(har_spec.phi(lat,newtime,time))
    two_body.append(har_spec.two_body_old(lat, psi_temp))
    D.append(observable.DHP(lat, psi_temp))
    new_e=har_spec.one_energy(lat,psi_temp,phi_original[-1])+har_spec.two_energy(lat,psi_temp)
    energy.append(new_e)
    new_e_doublon=har_spec.doublon_one_energy(lat,psi_temp,phi_original[-1])+har_spec.doublon_two_energy(lat,psi_temp)
    doublon_energy.append(new_e_doublon)

    # alternate way for calculating energy+ doublon added energy
    # energy.append(np.dot(psi_temp.conj().flatten(), evolve.f(lat,evolve.ham1(lat,h,newtime,time),psi_temp).flatten()))
    # for k in range(lat.nsites):
    #     g=0
    #     neleca, nelecb = lat.nup,lat.ndown
    #     norbs=lat.nsites
    #     civec=dc.cre_a(psi_temp,norbs,(neleca,nelecb),k)
    #     lat.nup+=1
    #     civec=dc.cre_b(civec,norbs,(neleca,nelecb),k)
    #     lat.ndown+=1
    #     lat.ne+=2
    #     print(evolve.f(lat,evolve.ham1(lat,h,newtime,time),civec).flatten().shape())
    #     g+=np.dot(civec.conj().flatten(), evolve.f(lat,evolve.ham1(lat,h,newtime,time),civec).flatten())
    #     lat.nup = number
    #     lat.ndown = number
    #     lat.ne=2*number
    # doublon_energy.append(g/lat.nsites)
    #





del phi_reconstruct[0:2]

np.save('./data/original/Jfield'+parameternames,J_field)
np.save('./data/original/phi'+parameternames,phi_original)
np.save('./data/original/phirecon'+parameternames,phi_reconstruct)
np.save('./data/original/neighbour'+parameternames,neighbour)
# np.save('./data/original/neighbour_check'+parameternames,neighbour_check)
np.save('./data/original/twobody'+parameternames,two_body)
np.save('./data/original/error'+parameternames,error)
np.save('./data/original/double'+parameternames,D)
np.save('./data/original/energy2'+parameternames,energy)
np.save('./data/original/doublonenergy2'+parameternames,doublon_energy)


# np.save('./data/original/position'+parameternames,X)



#plot_observables(lat, delta=0.02, time=5., K=.1)
# spectra(lat, initial=None, delta=delta, time=cycles, method='welch', min_spec=7, max_harm=50, gabor='fL')
