import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpmath import *
import evolve as evolve
import observable as observable
import definition as harmonic
import hub_lats as hub
import harmonic as har_spec
import des_cre as dc
from matplotlib import cm as cm
from scipy.integrate import ode
import des_cre as dc
import scipy
def phi(lat,current_time,cycles):
    if lat.field==0.:
        return 0.
    else:
        return (lat.a*lat.F0/lat.field)*(np.sin(lat.field*current_time/(2.*cycles))**2.)*np.sin(lat.field*current_time)

number = 5
nelec = (number, number)
nx = 10
ny = 0
t = 0.52
field = 32.9
F0 = 10
a = 4
cycles=10
gaps=[]
list=[]
corrs=[]
Ulist=[]



for f in range(1,21):
    list.append(1*f)
    U=1*f*t
    lat = harmonic.hhg(field=field, nup=number, ndown=number, nx=nx, ny=0, U=U, t=t, F0=F0, a=a, bc='pbc')
    Ulist.append(lat.U)
    # lat.U=f
    # gap_sum=nsum(lambda n: ((1+0.25*(n*lat.U)**2)**0.5 -0.5*n*lat.U)*((-1)**n), [1, inf])
    # print(gap_sum)
    # gap=(lat.U-4+8*gap_sum)
    # print(gap)

    int= lambda x: np.log(x+(x**2-1)**0.5)/np.cosh(2*np.pi*x/lat.U)
    corr_inverse=scipy.integrate.quad(int,1,np.inf)[0]
    corrs.append(lat.U/(4*corr_inverse))


    chem_integrand= lambda x: scipy.special.jv(1,x)/(x*(1+np.exp(x*lat.U/2)))
    # chem_integrand= lambda x: scipy.special.jv(1,x)/x
    chem=scipy.integrate.quad(chem_integrand,0,np.inf,limit=240)

    gap=lat.U-2*(2-4*chem[0])
    gaps.append(gap)

print(type(gaps))
print(type(corrs))
print(lat.a)
print(lat.F0)
print(lat.field)
breakdown=[a/(0.4*b*lat.F0) for a,b in zip(gaps,corrs)]
x=[n for n,i in enumerate(breakdown) if i>1][0]
print(float(x))
# plt.plot(list,gaps)
# plt.show()
#
# plt.plot(list[10:],corrs[10:])
# plt.show()

# plt.plot(list,breakdown)
# plt.show()
#
def phi_unit(lat,current_time,cycles):
    if lat.field==0.:
        return 0.
    else:
        return (np.sin(current_time*np.pi/(cycles))**2.)*np.sin(lat.field*current_time)
        # return (np.sin(current_time*np.pi/(cycles))**2.)

N=3000
times=np.linspace(0.0, cycles, N)
phi_list=[phi_unit(lat,t,cycles) for t in times]

plt.plot(times,phi_list)
plt.show()
breaktimes=[]
for a in breakdown:
    for j in times:
        if abs(phi_unit(lat,j,cycles))>a:
            breaktimes.append(j)
            break
plt.plot(breaktimes,Ulist[:len(breaktimes)])
plt.show()

print(breaktimes)

np.save('./data/original/breaktimes',breaktimes)# print(Ulist)

