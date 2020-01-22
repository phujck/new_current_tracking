import os
threads = 35
print("threads =%s" %threads)
os.environ["OMP_NUM_THREADS"] = "%s" % threads
import numpy as np
from tqdm import tqdm

# This contains the stuff needed to calculate some expectations. Generally contains stuff
# that applies operators to the wave function
import evolve as evolve

# Contains lots of important functions.
import definition as definition
# Sets up the lattice for the system
import hub_lats as hub

# These also contain various important observable calculators
import harmonic as har_spec
import observable as observable

# Not strictly necessary, but nice to have.
from matplotlib import cm as cm
from scipy.integrate import ode
import des_cre as dc

import timeit
import psutil

import resource
import platform
import sys


def memory_limit(percentage: float):
    if platform.system() != "Linux":
        print('Only works on linux!')
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * percentage, hard))


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def memory(percentage=0.8):
    def decorator(function):
        def wrapper(*args, **kwargs):
            memory_limit(percentage)
            try:
                function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 / 1024
                print('Remain: %.2f GB' % mem)
                sys.stderr.write('\n\nERROR: Memory Exception\n')
                sys.exit(1)

        return wrapper

    return decorator


@memory(percentage=0.8)
def main():
    print('My memory is limited to 80%.')


start = timeit.default_timer()

print(psutil.virtual_memory())
print('available memory: %s GB' % (get_memory() / 1024 / 1024))

"""Number of electrons"""
number = 9
# this specifically enforces spin up number being equal to spin down
nelec = (number, number)
# nelec = (number, 0)

"""number of sites"""
nx = int(2 * number)
# nx=10
ny = 0

"""System Parameters"""
t = 0.52
U = 1*t
field= 32.9
F0=10
# F0=0
a = 4
cycles = 0.2

"""Timestep used"""
delta = 0.05

"""these lists get popuproped with the appropriate expectations"""
neighbour = []
energy=[]
doublon_energy=[]
phi_original = []
# This is just to check the phi reconstruction does what it's supposed to.
phi_reconstruct=[0.,0.]
J_field = []
two_body = []
D=[]

"""used for saving expectations after simulation"""
parameternames='-%s-nsites-%s-cycles-%s-U-%s-t-%s-n-%s-delta-%s-field-%s-amplitude' % (nx,cycles,U,t,number,delta,field,F0)

"""class that contains all the essential parameters+scales them. V. IMPORTANT"""
prop = definition.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
# prop = definition.hhg(field=field, nup=number, ndown=0, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
print('\n')
print(vars(prop))
print(psutil.virtual_memory())


time=cycles

"""This sets initial wavefunction as the ground state"""
psi_temp = definition.hubbard(prop)[1].astype(complex)
init=psi_temp

h= hub.create_1e_ham(prop,True)
N = int(time/(prop.freq*delta))+1
print(N)


"""Set up the ode. Look at the scipy.integrate_ode page for more info."""
# r = ode(evolve.integrate_f).set_integrator('zvode', method='bdf')
# r.set_initial_value(psi_temp, 0).set_f_params(prop,time,h)
# delta=delta
# while r.successful() and r.t < time/prop.freq:
#     oldpsi=psi_temp
#     r.integrate(r.t + delta)
#     psi_temp = r.y
#     newtime = r.t
#
#     definition.progress(N, int(newtime / delta))
#     neighbour.append(har_spec.nearest_neighbour_new(prop, h, psi_temp))
#     J_field.append(har_spec.J_expectation(prop, h, psi_temp, newtime, time))
#     phi_original.append(har_spec.phi(prop,newtime,time))
#     two_body.append(har_spec.two_body_old(prop, psi_temp))
#     D.append(observable.DHP(prop, psi_temp))
#
"""RK4 method"""
# for k in tqdm(range(N),desc="Nuclear Countdown:"):
#         newtime=k*delta
#         psi_temp = evolve.RK4(prop, h, delta, newtime, psi_temp, time)
#         # eJ.append(har_spec.J_expectation(lat, h, psi_temp, k*delta, time))
#         # neighbour = har_spec.nearest_neighbour(lat, psi_temp)
#
#         neighbour.append(har_spec.nearest_neighbour_new(prop, h, psi_temp))
#         J_field.append(har_spec.J_expectation(prop, h, psi_temp, newtime, time))
#         phi_original.append(har_spec.phi(prop, newtime, time))
#         two_body.append(har_spec.two_body_old(prop, psi_temp))
#         D.append(observable.DHP(prop, psi_temp))

# del phi_reconstruct[0:2]
# np.save('./data/original/Jfield'+parameternames,J_field)
# np.save('./data/original/phi'+parameternames,phi_original)
# np.save('./data/original/phirecon'+parameternames,phi_reconstruct)
# np.save('./data/original/neighbour'+parameternames,neighbour)
# # np.save('./data/original/neighbour_check'+parameternames,neighbour_check)
# np.save('./data/original/twobody'+parameternames,two_body)
# np.save('./data/original/double'+parameternames,D)

"""rank testing"""
for k in tqdm(range(N)):
    # definition.progress(N,k)
    newtime = k * delta
    if k == 0:
        # psi_temp = evolve.RK4_max(prop, h, delta, psi_temp, time)
        psi_temp = evolve.RK4_constant(prop, h, delta, 120, psi_temp, time)
        J_field.append(har_spec.J_expectation_track(prop, h, psi_temp, har_spec.phi(prop, 120, time)))


    else:
        psi_temp = evolve.RK4_constant(prop, h, delta, 0, psi_temp, time)
        J_field.append(har_spec.J_expectation_track(prop, h, psi_temp, har_spec.phi(prop, 0, time)))
#         # eJ.append(har_spec.J_expectation(lat, h, psi_temp, k*delta, time))
#         # neighbour = har_spec.nearest_neighbour(lat, psi_temp)
#
#         neighbour.append(har_spec.nearest_neighbour_new(prop, h, psi_temp))
#         # J_field.append(har_spec.J_expectation(prop, h, psi_temp, newtime, time))
#         phi_original.append(har_spec.phi(prop, newtime, time))
#         two_body.append(har_spec.two_body_old(prop, psi_temp))
#         D.append(observable.DHP(prop, psi_temp))
np.save('./data/original/ranktestcurrentshorttime' + parameternames, J_field)

print("threads =%s" %threads)
stop = timeit.default_timer()

print('Time: ', stop - start)


