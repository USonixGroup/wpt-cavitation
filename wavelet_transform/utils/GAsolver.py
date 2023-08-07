##
##
##
##
#==============================================================================
#  This code calculates the cavitation threshold charts using the MAch number 1 criterion for inertial cavitation.
#  It uses the Gilmore bubble model and Akulichev method to estimate radiated pressure (GA model)
#==============================================================================
##
##
##
# <codecell>

import numpy as np
from   scipy.integrate import ode
import scipy.interpolate
#import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


# <codecell> ODE Function (GA Model)
def GAmodel(tau,y):   
    global nH
    R = y[0]
    Rdot = y[1]
#    nHdot = y[0]
    
    # Excitation Pressure   
    nPac = -npac*np.sin(nomega*tau)
    
    # Enthalpy Calculation
    nPinf = nP0 + nPac
    nCinf = nc
    
    nPi = (nP0 + 2.*nsigma/nR0)*(nR0/R)**(3*kappa)   
    nPR = nPi - 2.*nsigma/R - 4.*nmu*Rdot/R
    coeff = n*nA**(1./n)/(n-1.)/nrho
    nH = coeff*((nPR + nB)**(1.-1./n) - (nPinf + nB)**(1.-1./n))                                                   # this is to calculate radiated pressure from oscillating bubble, this is used in the following line
    # Speed of sound at the wall of bubble
    nCR = np.sqrt(nCinf**2 + (n-1)*nH)
    # calculating dH/dR ar bubble wall
    dnPi_dR = -3*kappa/R * nPi
    dnPR_dR = dnPi_dR + 2.*nsigma/R**2 + 4.*nmu*Rdot/R**2
    dH_dR = (nA**(1./n)/nrho)*(nPR + nB)**(-1./n) * dnPR_dR
    
    X1 = Rdot
    X2 = (1./R) * ( nH*(nCR + Rdot)/(nCR - Rdot) + dH_dR*Rdot*R/nCR - 0.5*Rdot**2*(3.*nCR - Rdot)/(nCR - Rdot) )
    
    return [X1,X2]

# <codecell>  Dormulating the solver for GA model and perform post-processing

def GA_Solver(omega,pac,R0,p0,t,tstep,A,B):
    # PARAMETERS
    global nc,nrho, nP0, nsigma, nR0, nmu, nomega, npac, nA, nB
    # Non-dimensionalisation of parameters
    nc = c * (R0*omega)**(-1)
    nrho = rho * (R0*omega)**2/p0
    nP0 = p0/p0
    nsigma = sigma/(p0*R0)
    nR0 = R0/R0
    nmu = mu*omega/p0
    nomega = omega/omega
    npac = pac/p0
    tau = omega*t
    nA = A/p0
    nB = B/p0
    dtau = omega*tstep    
    A_RecDiff_coeff = 2*nsigma/nP0
    B_RecDiff_coeff = nomega**2 * nrho/(3*kappa*nP0)
    
    
    Y0 = np.array([nR0,0])
    Yint = ode(GAmodel).set_integrator('vode', method='BDF',atol=1E-16)
    Yint.set_initial_value(Y0,tau[0])
    Ysol = np.ndarray(shape=(len(tau),2),order='F') 
    H_int = np.ndarray(shape=(len(tau),1))
    nR_Sim = np.ndarray(shape=(len(tau),1))
    nRdot_Sim = np.ndarray(shape=(len(tau),1))    
    
    
    idx_int = 0
    while Yint.successful () and Yint.t < tau[-1] :
        Ysol[idx_int:] = Yint.integrate(Yint.t + dtau)
        global nH
        H_int[idx_int,0] = nH
        idx_int = idx_int +1
        
    Y = Ysol.real
    if Y[::].imag.any(): print 'IMAGINARY solution errorrrr.....'
    nR_Sim[:,0] = Y[:,0]
    nRdot_Sim[:,0] = Y[:,1]    
    
    # <codecell>
    # Post processing cell
#    nG_bub = Y[:,1]*(Y[:,0] + 0.5*Y[:,2]**2)        # invariant of bubble motion, see Akulichev, High intensity ultrasound fields
#    nP_rad = nA*( 2./(n+1.) + (n-1.)/(n+1.)*np.sqrt(1. + nG_bub*(n+1.)/nr_dist/(nc**2) ) )**(2*n/(n-1.)) - nB
    nomega_res = np.sqrt( (1/nrho/nR0**2) * (3*kappa*nP0 + 2*nsigma/nR0*(3*kappa - 1)) )
    damp_factor = 4*nmu/(nrho*nomega_res*nR0**2)
    Rec_Diff_THR = np.sqrt( 1.5*(A_RecDiff_coeff*nP0**2/(A_RecDiff_coeff + nR0)) * ((1 - nR0**2*B_RecDiff_coeff)**2 + nR0**2*B_RecDiff_coeff*damp_factor) )    
#    
    nC_R = np.sqrt(nc**2 + (n-1)*H_int)
    Mach_max = (np.abs(nRdot_Sim)).max()/nc
    speed_ratio_max = nC_R.max()/nc
    
    #    Mach_max = Mach.max() 
#    print nC_R.shape,nRdot_Sim.shape 
#    print Rec_Diff_THR, nomega_res    
    
    del nc,nrho, nP0, nsigma, nR0, nmu, nomega, npac, A_RecDiff_coeff, B_RecDiff_coeff
   
    return Mach_max, speed_ratio_max , nR_Sim.max(), H_int.max(), nomega_res, Rec_Diff_THR 

# <codecell>
  
plt.close('all')    
 
# Excitation Parameters
f = 26.5*1E3                     #insonation frequency (Hz)
no_cycle = 20                   #no. of cycles in burst
r_dist = 10.E-6               # distance from the centre of bubble which radiated pressure (acoustic emission from bubble) is calculated 
omega = 2*np.pi*f

# Ambient condition
p0 = 1.0E5                     #ambient pressure (Pa)

# Medium properties
sigma = 72.5E-3                #equlibrium gas/shell surface tension (N/m)
rho = 1000.0                   #surrounding fluid density (kg/m^3)
mu = 1.0E-3                    #surrounding fluid viscosity (Pa.s)
c = 1481.0                     #surrounding fluid sound velocity (m/s)
pv = 3.2718E3                  #vapor pressure (Pa)
gamma = 1.4                    #adiabatic gas constant, gas solved in the water  (adiabatic index)
kappa = gamma  
A = 304.E6                     #constant in the pressure-density state equation of water (adiabatic condition)     , the equatios is p = A(rho/rho_0)**n - B   
B = 303.9E6                    #constant in the pressure-density state equation of water (adiabatic condition)     , the equatios is p = A(rho/rho_0)**n - B   , Church used the B =A-1 relationship
n = 7.                         #constant in the pressure-density state equation of water (adiabatic condition)     , the equatios is p = A(rho/rho_0)**n - B   
idx = 0

fs = f*2E3
tstep = 1./fs
tmax = no_cycle/f
t = np.arange(0.,tmax,tstep)   # time array
 
R0_THR_range = np.arange(0.1E-6,4E-6,0.1E-6)
pac_THR_range = np.arange(2*p0,10*p0,0.5*p0)

Mach_MaxSim = np.ndarray(shape=(len(pac_THR_range),len(R0_THR_range)),order='F') 
C_ratioMax = np.ndarray(shape=(len(pac_THR_range),len(R0_THR_range)),order='F') 
nR_MaxSim = np.ndarray(shape=(len(pac_THR_range),len(R0_THR_range)),order='F') 
H_MaxSim = np.ndarray(shape=(len(pac_THR_range),len(R0_THR_range)),order='F') 
nP_RecDiff_THR = np.ndarray(shape=(len(pac_THR_range),len(R0_THR_range)),order='F') 
nomega_res_R0_THR = np.ndarray(shape=(len(pac_THR_range),len(R0_THR_range)),order='F') 

N_rows = Mach_MaxSim.shape [0]
N_cols = Mach_MaxSim.shape [1]

for R0_THR in R0_THR_range:
    idx2 = 0
    for pac_THR in pac_THR_range:
        GA_out = GA_Solver(omega,pac_THR,R0_THR,p0,t,tstep,A,B)
        Mach_MaxSim[idx2,idx] = GA_out[0]
        C_ratioMax[idx2,idx] = GA_out[1]
        nR_MaxSim[idx2,idx] = GA_out[2]
        H_MaxSim[idx2,idx] = GA_out[3]
        nomega_res_R0_THR[idx2,idx] = GA_out[4]
        nP_RecDiff_THR[idx2,idx] = GA_out[5]
        print N_rows-idx2 , N_cols-idx
        idx2 = idx2 +1
    idx = idx+1 
        
# <codecell>   Plotting cell


title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'weight':'normal','size':'18'}
tick_size = 16
figsavedirectory = 'D:/Anaconda/Work/Figs/'


t_plot = t*1E6        # change the time axis unit to micro second
figsize_inches1 = (16,9)      #(width,height) in inches
figsize_inches2 = (16,6.75)      #(width,height) in inches

Xdata_1D = R0_THR_range*1E6
Ydata_1D = pac_THR_range*1E-3
Xdata, Ydata = np.meshgrid(Xdata_1D, Ydata_1D)
Zdata = Mach_MaxSim
Zdata_lev = 10*np.log(Mach_MaxSim)     # just to stretch the graph and make it more visible

# 3D surface plot
fig = plt.figure(figsize=figsize_inches1,facecolor='w')
ax = fig.gca(projection='3d')
surfplt = ax.plot_surface(Xdata, Ydata, Zdata_lev, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased = False )
ax.contourf(Xdata, Ydata, Zdata_lev, zdir='z', offset = Zdata_lev.min()-50, cmap=cm.jet)
ax.contourf(Xdata, Ydata, Zdata_lev, zdir='x', offset = Xdata_1D.min()-50, cmap=cm.jet)
ax.contourf(Xdata, Ydata, Zdata_lev, zdir='y', offset = Ydata_1D.max()+800, cmap=cm.jet)
cBAR = fig.colorbar(surfplt, shrink=0.8, extend='both') #, label='Logarithm of Mach number')
cBAR.set_label('Logarithm of Mach number')
ax.set_xlabel('$R_0 (\mu m)$', axis_font)
ax.set_xlim(Xdata_1D.min()-50, Xdata_1D.max()+50)
ax.set_ylabel('Acoustic pressure (kPa)',axis_font)
ax.set_ylim(Ydata_1D.min() - 800, Ydata_1D.max() + 500)
ax.set_zlabel('Logarithm of acoustic Mach number',axis_font)
ax.set_zlim(Zdata_lev.min() - 50, Zdata_lev.max() + 10)
ax.tick_params(axis='both', which='major', labelsize=tick_size)

plt.show()

figname1 = "3DCavTHR%dkHz.%s" % (f*1E-3,'pdf')             #'BD{0d}mic{1d}kPa{2d}kHz.{3}'.format(R0*1E6,pac*1E-3,f*1E-3,'pdf')
plt.savefig(figsavedirectory+figname1)

# <codecell>

#2D contour plot

fig2 = plt.figure(figsize=figsize_inches1,facecolor='w')
ax2 = fig2.gca()

#im = plt.imshow(Zdata_lev, interpolation='bilinear', cmap=cm.gray, extent=(Xdata_1D.min(),Xdata_1D.max(),Ydata_1D.min()*1E-2,Ydata_1D.max()*1E-2) )
levels = np.arange(Zdata_lev.min(), Zdata_lev.max(), 5)
ContPlt = plt.contour(Xdata,Ydata,Zdata_lev,levels,colors='k',linewidths=1.5, origin='lower', extent=(Xdata_1D.min(),Xdata_1D.max(),Ydata_1D.min(),Ydata_1D.max()) )   #extent=(Xdata_1D.min(),Xdata_1D.max(),Ydata_1D.min(),Ydata_1D.max())

#Thicken the zero contour.
zerocontindx = np.array((levels<0).nonzero()).max() 
zc = ContPlt.collections[zerocontindx]
plt.setp(zc, linewidth=5)

plt.clabel(ContPlt, levels[1::5],  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=18,
           colors='k')

plt.xlabel('$R_0 (\mu m)$',axis_font)
plt.xlim(Xdata_1D.min()-1, Xdata_1D.max()+1)
plt.ylabel('Acoustic pressure (kPa)',axis_font)
plt.ylim(Ydata_1D.min() - 80, Ydata_1D.max() + 50)
ax2.tick_params(axis='both', which='major', labelsize=tick_size)
ax2.annotate('Rec_Diff Thrd', xy=(0.95, 0.04),  xycoords='axes fraction',xytext=(0.6, 0.20),textcoords='figure fraction', fontsize=16, fontweight='bold',
             arrowprops=dict(facecolor='black', shrink=0.12))          # ...,horizontalalignment='left', verticalalignment='bottom'
# Drawing a filled contour on top of the previous one
ContFPlt = plt.contourf(Xdata,Ydata,Zdata_lev,levels, origin='lower', extent=(Xdata_1D.min(),Xdata_1D.max(),Ydata_1D.min(),Ydata_1D.max()) )   #extent=(Xdata_1D.min(),Xdata_1D.max(),Ydata_1D.min(),Ydata_1D.max())

# make a colorbar for the contour lines
ContCBAR = plt.colorbar(ContFPlt, shrink=0.8, extend='both')
ContCBAR.set_label('Logarithm of Mach number')
plt.flag()
plt.jet()  # Now change the colormap for the contour lines and colorbar
# We can still add a colorbar for the image, too.
#CBI = plt.colorbar(im, orientation='horizontal', shrink=0.8)
# This makes the original colorbar look a bit out of place,
# so let's improve its position.
#l,b,w,h = plt.gca().get_position().bounds
#ll,bb,ww,hh = ContCBAR.ax.get_position().bounds
#ContCBAR.ax.set_position([ll, b+0.1*h, ww, h*0.8])

# Plotting rectified diffusion threshold
plt.plot(Xdata[0,:],nP_RecDiff_THR[0,:]*p0*1E-3,color='r',linewidth=3.5)

plt.show()

figname2 = "ContCavTHR%dkHz.%s" % (f*1E-3,'pdf')             #'BD{0d}mic{1d}kPa{2d}kHz.{3}'.format(R0*1E6,pac*1E-3,f*1E-3,'pdf')
plt.savefig(figsavedirectory+figname2)





# <codecell>
zoomindx = 70
XdataZoomed = Xdata[:,0:zoomindx]
YdataZoomed = Ydata[:,0:zoomindx]
ZdataZoomed = Zdata_lev[:,0:zoomindx]

#2D contour plot, zoomed in low R0

fig3 = plt.figure(figsize=figsize_inches2,facecolor='w')
ax3 = fig3.gca()

#im = plt.imshow(Zdata_lev, interpolation='bilinear', cmap=cm.gray, extent=(Xdata_1D.min(),Xdata_1D.max(),Ydata_1D.min()*1E-2,Ydata_1D.max()*1E-2) )
levels3 = np.arange(ZdataZoomed.min(), ZdataZoomed.max(), 5)
ContPlt3 = plt.contour(XdataZoomed,YdataZoomed,ZdataZoomed,levels3,colors='k',linewidths=1.5, origin='lower', extent=(XdataZoomed.min(),XdataZoomed.max(),YdataZoomed.min(),YdataZoomed.max()) )   #extent=(Xdata_1D.min(),Xdata_1D.max(),Ydata_1D.min(),Ydata_1D.max())

#Thicken the zero contour.
zerocontindx = np.array((levels3<0).nonzero()).max() 
zc = ContPlt.collections[zerocontindx]
plt.setp(zc, linewidth=4)

plt.clabel(ContPlt3, levels3[1::4],  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=18,
           colors='k')

plt.xlabel('$R_0 (\mu m)$',axis_font)
plt.xlim(XdataZoomed.min(), XdataZoomed.max())
plt.ylabel('Acoustic pressure (kPa)',axis_font)
plt.ylim(YdataZoomed.min() - 80, YdataZoomed.max() + 50)
ax3.tick_params(axis='both', which='major', labelsize=tick_size)
ax3.annotate('Rec_Diff Thrd', xy=(0.61, 0.65),  xycoords='figure fraction',xytext=(0.65, 0.5),textcoords='figure fraction',
             arrowprops=dict(facecolor='black', shrink=0.12))          # ...,horizontalalignment='left', verticalalignment='bottom'

# Drawing a filled contour on top of the previous one
ContFPlt3 = plt.contourf(XdataZoomed,YdataZoomed,ZdataZoomed,levels3, origin='lower', extent=(XdataZoomed.min(),XdataZoomed.max(),YdataZoomed.min(),YdataZoomed.max()) )   #extent=(Xdata_1D.min(),Xdata_1D.max(),Ydata_1D.min(),Ydata_1D.max())

# make a colorbar for the contour lines
ContCBAR3 = plt.colorbar(ContFPlt3, shrink=0.8, extend='both')
ContCBAR3.set_label('Logarithm of Mach number')
plt.flag()
plt.jet()  # Now change the colormap for the contour lines and colorbar
plt.show()

# Plotting rectified diffusion threshold
plt.plot(Xdata[0,0:zoomindx],nP_RecDiff_THR[0,0:zoomindx]*p0*1E-3,color='r',linewidth=5)

figname3 = "ContCavTHRZoomedR0%dkHz.%s" % (f*1E-3,'pdf')             #'BD{0d}mic{1d}kPa{2d}kHz.{3}'.format(R0*1E6,pac*1E-3,f*1E-3,'pdf')
plt.savefig(figsavedirectory+figname3)


# <codecell>
Xdatalog = 10*np.log10(Xdata)
#2D contour plot

fig4 = plt.figure(figsize=figsize_inches2,facecolor='w')
ax4 = fig4.gca()

#im = plt.imshow(Zdata_lev, interpolation='bilinear', cmap=cm.gray, extent=(Xdata_1D.min(),Xdata_1D.max(),Ydata_1D.min()*1E-2,Ydata_1D.max()*1E-2) )
levels = np.arange(Zdata_lev.min(), Zdata_lev.max(), 5)
ContPlt4 = plt.contour(Xdatalog,Ydata,Zdata_lev,levels,colors='k',linewidths=1.5, origin='lower', extent=(Xdatalog.min(),Xdatalog.max(),Ydata_1D.min(),Ydata_1D.max()) )   #extent=(Xdata_1D.min(),Xdata_1D.max(),Ydata_1D.min(),Ydata_1D.max())

#Thicken the zero contour.
zerocontindx = np.array((levels<0).nonzero()).max() 
zc = ContPlt4.collections[zerocontindx]
plt.setp(zc, linewidth=6)

plt.clabel(ContPlt4, levels[1::2],  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=18,
           colors='k')

plt.xlabel('$10log_{10}R_0 (R_0  \mathrm{in}  \mu m)$',axis_font)
plt.xlim(Xdatalog.min(), Xdatalog.max())
plt.ylabel('Acoustic pressure (kPa)',axis_font)
plt.ylim(Ydata_1D.min() - 80, Ydata_1D.max() + 50)
ax4.tick_params(axis='both', which='major', labelsize=tick_size)


# Drawing a filled contour on top of the previous one
ContFPlt4 = plt.contourf(Xdatalog,Ydata,Zdata_lev,levels, origin='lower', extent=(Xdatalog.min(),Xdatalog.max(),Ydata_1D.min(),Ydata_1D.max()) )   #extent=(Xdata_1D.min(),Xdata_1D.max(),Ydata_1D.min(),Ydata_1D.max())

# make a colorbar for the contour lines
ContCBAR4 = plt.colorbar(ContFPlt, shrink=0.8, extend='both')
ContCBAR4.set_label('Logarithm of Mach number')
plt.flag()
plt.jet()  # Now change the colormap for the contour lines and colorbar

plt.show()
