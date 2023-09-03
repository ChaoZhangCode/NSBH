import autograd.numpy as np
from autograd import elementwise_grad as grad
from autograd.numpy import sin, cos, pi, sqrt, tan, arctan, arccos, log10, log, exp, sinc
import matplotlib.pyplot as plt
from matplotlib import ticker
from multiprocessing import Pool
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from  scipy.optimize import minimize

coefs = np.array([
    [ 3.916,    4.078,    4.857],
    [ 7.701,    7.587,    6.981],
    [ 0.00858,  0.00839,  0.00706],
    [ 0.22114,  0.21695,  0.19351],
    [ 3.269,    3.614,    4.085],
    [ 11.964,   11.942,   12.065],
    [ 13.349,   13.751,   10.521],
    [ 1.3683,   1.3373,   1.5905],
    [ 3.254,    3.606,    4.104],
    [-12.953,  -22.996,  -28.726],
    [ 0.9237,   1.6229,   2.0845],
    [ 6.20,     4.88,     4.89],
    [ 14.383,   14.274,   14.302],
    [ 16.693,   23.560,   22.881],
    [-1.0514,  -1.5564,  -1.7690],
    [ 2.486,    2.095,    0.989],
    [ 15.362,   15.294,   15.313],
    [ 0.085,    0.084,    0.091],
    [ 6.23,     6.36,     4.68],
    [ 11.68,    11.67,    11.65],
    [-0.029,   -0.042,   -0.086],
    [ 20.1,     14.8,     10.0],
    [ 14.19,    14.18,    14.15]])

Names = ["BSk19", "BSk20", "BSk21"]

G = 6.67430e-11; # N m^2 / kg^2, m^3 / kg / s^2
C = 299792458e0 # m/s
MSUN = 1.988409902147041e30 # kg
NSMass = 1.4*MSUN # kg
Mpc = 3.08567758149e22 # Megaparsec, m
# R = np.array([1.074e4, 1.174e4, 1.257e4]) # m
R = np.array([1.0737658e4, 1.1742791e4, 1.2589295e4]) # m

plt.rc('text', usetex=True)
plt.rc('font', family='calibri', weight='bold')

def presure(rho, NSmodel): # rho, kg/m^3, = 10^-3 g/cm^3
    """
    NSmodel = 0: BSk19, 1: BSk20, 2: BSk21
    """
    x = log10(rho*1e-3) # to g/cm^3
    
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23 = coefs[:, NSmodel]
    logp = ((a1 + a2*x + a3*x**3)/(1+a4*x)/(exp(a5*(x-a6)) + 1)
            + (a7 + a8*x)/(exp(a9*(a6-x)) + 1)
            + (a10 + a11*x)/(exp(a12*(a13-x)) + 1)
            + (a14 + a15*x)/(exp(a16*(a17-x)) + 1)
            + a18/(1+(a19*(x-a20))**2) + a21/(1+(a22*(x-a23))**2))
    
    return 10**logp*0.1 # kg/m/s^2, N/m^2, = 10 dyn/cm^2


# Ref[6] Fig. 2
def plotprho():
    rhos = 10**np.linspace(6, 17, 10000)  # g/cm^3
    for i in range(3):
        ps = presure(rhos * 1e3, i) * 10 # dyn/cm^2
        np.savetxt("model%dprho.txt"%i, np.vstack((rhos, ps)).T)
        ys = ps / rhos**1.4
        plt.loglog(rhos, ys, label=Names[i])
    plt.xlabel(r"$\log \rho\ \mathrm{(g\ cm^{-3})}$")
    plt.ylabel(r"$\log P\ \mathrm{(dyn/cm^2)} - 1.4\ \log\rho\ \mathrm{(g\ cm^{-3})}$")
    plt.legend()
    plt.title("Ref.[6] Fig. 2")
    # plt.show()
# plotprho()

dpdrho = grad(presure) # m^2 / s^2

# function that returns dz/dt

def Getdmdr(r, rho):
    return 4*pi*r**2*rho

def Getdalphadr(m, r, p):
    res = (G*m/C**2 + 4*pi*G*r**3*p/C**4) / (r*(r-2*G*m/C**2)) #  1 / m
    return res
def Getdrhodr(m, r, rho, NSmodel):
    p = presure(rho, NSmodel)
    return -(rho + p/C**2)*Getdalphadr(m, r, p)/(dpdrho(rho, NSmodel)/C**2)

# def Getdalphadr(m, r, p):
#     res = (G*m/C**2 ) / (r*(r)) #  1 / m
#     return res
# def Getdrhodr(m, r, rho, NSmodel):
#     p = presure(rho, NSmodel)
#     return -(rho)*Getdalphadr(m, r, p)/(dpdrho(rho, NSmodel)/C**2)

def model(x, r, NSmodel):
    alpha, m, rho = x

    p = presure(rho, NSmodel)

    dalphadr = Getdalphadr(m, r, p)
    dmdr = Getdmdr(r, rho)
    drhodr = Getdrhodr(m, r, rho, NSmodel)

    return [dalphadr, dmdr, drhodr]

class MajLogLocator(ticker.LogLocator):
    def __init__(self, base, vmin, vmax, numticks=None, **kwargs):
        super().__init__(base=base, numticks=numticks, **kwargs)
        self.vmin = vmin
        self.vmax = vmax
    def nonsingular(self, a, b):
        return (self.vmin, self.vmax)

class MinLogLocator(ticker.LogLocator):
    def __init__(self, base, vmin, vmax, **kwargs):
        super().__init__(base=base, subs=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], numticks=9, **kwargs)
        self.vmin = vmin
        self.vmax = vmax
    def nonsingular(self, a, b):
        return (self.vmin, self.vmax)
    
def SolveMetric():
    size = 25
    legsize=16
    labelpad = 3
    # plt.rc('xtick', top=True, labeltop=False, labelsize=16)
    # plt.rc('ytick', right=True, labelright=False, labelsize=16)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    labels = [r'$\alpha$', r'$m[\mathrm{kg}]$', r'$\rho\left[\mathrm{kg/m^3}\right]$', r'$p\left[\mathrm{N/m^2}\right]$']  
    labels2 = [r'$\frac{d\alpha}{dr}$', r'$\frac{dm}{dr}$', r'$\frac{d\rho}{dr}$', r'$\frac{dp}{dr}$']  
    colors = ['firebrick', 'goldenrod', 'royalblue']   
    
    # solve    
    N = round(1e6)
    rmin = 1e-3#1e1     
    xmin = rmin
    lss = ["-", "--", ":"]
    xticks = [[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4], [r"$10^{-2}$", "", r"$10^{0}$", "",  r"$10^{2}$", "", r"$10^{4}$"]]
    xMajLocator = MajLogLocator(10, rmin, 2e4, 10)
    xMinLogLocator = MinLogLocator(10, rmin, 2e4)

    # alpha(r), m(r), rho(r) at r = 0.01 m
    inv_ini = [ [-6.148564833233587379e-01, 1.652836497791889286e+24, 1.837432761244823808e+18],
                [-5.675155338847596154e-01, 1.834122008910400424e+24, 1.327752917272534784e+18],
                [-5.746914018921139844e-01, 2.268974248938643892e+24, 1.215639592606444544e+18]]

    # for NSmodel in range(3):
    for NSmodel in [0]:
        # initial condition
        ini = [-G*NSMass/C**2/R[NSmodel], NSMass, 7.86e3] # 1, kg, kg/m^3
        # print(ini, 2*G*NSMass/C**2)

        # rarray = np.linspace(rmin, R[NSmodel], N)       
        rarray = 10**np.linspace(np.log10(rmin), np.log10(R[NSmodel]), N) 
        # rarray = np.hstack((10**np.linspace(np.log10(rmin), np.log10(0.1), N), 10**np.linspace(np.log10(0.1), np.log10(R[NSmodel]), N)))
        # print(rarray)
    
        # solve ODE
        res = odeint(model, ini, rarray[::-1], args=(NSmodel,))[::-1, :]
        print(res.shape)


        # step = round(N/1e1)
        step = 1
        
        end = None # round(N/1000)
        
        lw = 0.5
        r = rarray[::step][:end]

        alpha = res[::step, 0][:end]
        m = res[::step, 1][:end]
        rho = res[::step, 2][:end]
        p = presure(rho, NSmodel)

        # np.savetxt("model%dalpha.txt"%NSmodel, np.vstack((r, res[::step, 0], Getdalphadr(m, r, p))).T)
        # np.savetxt("model%dm.txt"%NSmodel, np.vstack((r, m, Getdmdr(r, rho))).T)
        # np.savetxt("model%drho.txt"%NSmodel, np.vstack((r, rho, Getdrhodr(m, r, rho, NSmodel))).T)

        # print(NSmodel, res[0, 1])
        axes[0, 0].semilogx(r, alpha, label=Names[NSmodel], color=colors[NSmodel], lw=lw, ls=lss[NSmodel])
        # axes[0, 0].plot(r, Getdalphadr(m, r, p), label=Names[NSmodel]+', ' + labels2[0], ls="--", color=colors[NSmodel], lw=lw)
        
        axes[0, 1].loglog(r, m, label=Names[NSmodel], color=colors[NSmodel], lw=lw, ls=lss[NSmodel])
        # axes[0, 1].loglog(r, Getdmdr(r, rho), label=Names[NSmodel]+', ' + labels2[1], ls="--", color=colors[NSmodel], lw=lw)
        
        axes[1, 0].loglog(r, rho, label=Names[NSmodel], color=colors[NSmodel], lw=lw, ls=lss[NSmodel])
        # axes[1, 0].loglog(r, Getdrhodr(m, r, rho, NSmodel), label=Names[NSmodel]+', ' + labels2[2], ls="--", color=colors[NSmodel], lw=lw)
        
        axes[1, 1].loglog(r, p, label=Names[NSmodel], color=colors[NSmodel], lw=lw, ls=lss[NSmodel])
        # axes[1, 1].loglog(r, dpdrho(rho, NSmodel)*Getdrhodr(m, r, rho, NSmodel), label=Names[NSmodel]+', ' + labels2[3], ls="--", color=colors[NSmodel], lw=lw)


        # # solve ODE from core
        # res2 = odeint(model, inv_ini[NSmodel], rarray, args=(NSmodel,))
        # print(res2.shape)

        # alpha2 = res2[::step, 0][:end]
        # m2 = res2[::step, 1][:end]
        # rho2 = res2[::step, 2][:end]
        # p2 = presure(rho2, NSmodel)

        # # print(NSmodel, res[0, 1])
        # axes[0, 0].semilogx(r, alpha2, color=colors[NSmodel], lw=lw, ls=lss[NSmodel])
        # # axes[0, 0].plot(r, Getdalphadr(m, r, p), label=Names[NSmodel]+', ' + labels2[0], ls="--", color=colors[NSmodel], lw=lw)
        
        # axes[0, 1].loglog(r, m2, color=colors[NSmodel], lw=lw, ls=lss[NSmodel])
        # # axes[0, 1].loglog(r, Getdmdr(r, rho), label=Names[NSmodel]+', ' + labels2[1], ls="--", color=colors[NSmodel], lw=lw)
        
        # axes[1, 0].loglog(r, rho2, color=colors[NSmodel], lw=lw, ls=lss[NSmodel])
        # # axes[1, 0].loglog(r, Getdrhodr(m, r, rho, NSmodel), label=Names[NSmodel]+', ' + labels2[2], ls="--", color=colors[NSmodel], lw=lw)
        
        # axes[1, 1].loglog(r, p2, color=colors[NSmodel], lw=lw, ls=lss[NSmodel])
        # # axes[1, 1].loglog(r, dpdrho(rho, NSmodel)*Getdrhodr(m, r, rho, NSmodel), label=Names[NSmodel]+', ' + labels2[3], ls="--", color=colors[NSmodel], lw=lw)


    axes[0, 0].set_xlabel(r'$r[\mathrm{m}]$', size=size, labelpad=labelpad)  
    axes[0, 0].set_ylabel(labels[0], size=size)  
    axes[0, 0].legend(ncol=1, fontsize=legsize)
    axes[0, 0].tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    axes[0, 0].set_yticks([-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0], ["-0.6", "", "-0.4", "", "-0.2", "", "0"])
    # axes[0, 0].yaxis.set_major_locator(ticker.MaxNLocator(3))
    # axes[0, 0].yaxis.set_minor_locator(ticker.MaxNLocator(12))
    axes[0, 0].xaxis.set_major_locator(xMajLocator)
    axes[0, 0].xaxis.set_minor_locator(xMinLogLocator)
    axes[0, 0].set_xticks(*xticks)
    axes[0, 0].set_xlim(xmin, R[NSmodel]) 
    
    axes[0, 1].set_xlabel(r'$r[\mathrm{m}]$', size=size, labelpad=labelpad) 
    axes[0, 1].set_ylabel(labels[1], size=size) 
    axes[0, 1].legend(ncol=1, fontsize=legsize)
    axes[0, 1].tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    axes[0, 1].yaxis.set_major_locator(MajLogLocator(10, 1e24, 4e30, 8))
    axes[0, 1].yaxis.set_minor_locator(MinLogLocator(10, 1e24, 4e30))
    axes[0, 1].xaxis.set_major_locator(xMajLocator)
    axes[0, 1].xaxis.set_minor_locator(xMinLogLocator)
    axes[0, 1].set_xticks(*xticks)
    axes[0, 1].set_yticks([1e24, 1e25, 1e26, 1e27, 1e28, 1e29, 1e30], [r"$10^{24}$", "", r"$10^{26}$", "", r"$10^{28}$", "", r"$10^{30}$"])
    axes[0, 1].set_xlim(xmin, R[NSmodel])          
    axes[0, 1].set_ylim(1e24, 4e30)
        
    axes[1, 0].set_xlabel(r'$r[\mathrm{m}]$', size=size, labelpad=labelpad) 
    axes[1, 0].set_ylabel(labels[2], size=size)
    axes[1, 0].legend(ncol=1, fontsize=legsize)
    axes[1, 0].tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    axes[1, 0].yaxis.set_major_locator(MajLogLocator(10, 1e16, 1e19, 6))
    axes[1, 0].yaxis.set_minor_locator(MinLogLocator(10, 1e16, 1e19))
    axes[1, 0].xaxis.set_major_locator(xMajLocator)
    axes[1, 0].xaxis.set_minor_locator(xMinLogLocator)
    axes[1, 0].set_xticks(*xticks)
    axes[1, 0].set_xlim(xmin, R[NSmodel])     
    axes[1, 0].set_ylim(1e16, 1e19)
    
    axes[1, 1].set_xlabel(r'$r[\mathrm{m}]$', size=size, labelpad=labelpad)
    axes[1, 1].set_ylabel(labels[3], size=size)
    axes[1, 1].legend(ncol=1, fontsize=legsize)
    axes[1, 1].tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    axes[1, 1].yaxis.set_major_locator(MajLogLocator(10, 1e31, 4e34, 6))
    axes[1, 1].yaxis.set_minor_locator(MinLogLocator(10, 1e31, 4e34))
    axes[1, 1].xaxis.set_major_locator(xMajLocator)
    axes[1, 1].xaxis.set_minor_locator(xMinLogLocator)    
    axes[1, 1].set_xticks(*xticks)
    axes[1, 1].set_xlim(xmin, R[NSmodel])     
    axes[1, 1].set_ylim(1e31, 4e34)

        
    # fig.suptitle(r"$ds^2=-e^{2\alpha}dt^2+[1-2Gm/r]^{-1}dr^2+r^2d\Omega^2$")
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.95, hspace=0.5, wspace=0.5)
    plt.show()
    # fig.savefig("NSMetric.pdf")

# SolveMetric()


def Evolution():
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    size = 25
    legsize=16
    labelpad = 2.5
    lss = ["-", "--", ":"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # data = []
    # data = np.load("OrbitEvolution.npy", allow_pickle=True)
    
    for model in range(3):
        # dat = []
        for i in range(4):
            # da = data[model, i]
            mass_magnitude = i+3
            filename = "Mm%d_orbit_model%d.txt" % (mass_magnitude, model)
            da = np.loadtxt(filename)

            print(filename, da.shape)
            # dat.append(da)
            axes[0, 0].loglog(da[:, 0], da[:, 1], color=colors[i], ls=lss[model], label=(r"$10^{-%d}$"%mass_magnitude) if model == 0 else None)
            axes[0, 1].loglog(da[:, 0], da[:, 3], color=colors[i], ls=lss[model], label=Names[model] if i==0 else None)
            axes[1, 0].loglog(da[:, 0], da[:, 2]/MSUN, color=colors[i], ls=lss[model])
            axes[1, 1].loglog(da[:, 0], da[:, 4]/pi, color=colors[i], ls=lss[model])
    #     data.append(np.array(dat, dtype=object))
    # data = np.array(data)
    # print(data.shape)
    # np.save("OrbitEvolution.npy", np.array(data, dtype=object))
    
    axes[0, 0].set_ylabel(r"$r[m]$", size=size, labelpad=labelpad)
    axes[0, 0].set_xlabel(r"$t[s]$", size=size, labelpad=labelpad)
    axes[0, 0].yaxis.set_major_locator(MajLogLocator(10, 1e1, 2e4, 5))
    axes[0, 0].yaxis.set_minor_locator(MinLogLocator(10, 1e1, 2e4))
    axes[0, 0].set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], [r"$10^{-4}$", "", r"$10^{-2}$", "",  r"$10^{0}$", "", r"$10^{2}$"])
    axes[0, 0].xaxis.set_minor_locator(MinLogLocator(10, 1e-4, 1e2))
    axes[0, 0].tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    axes[0, 0].legend(ncol=1, fontsize=legsize+5, title=r"   $m_D[M_\odot]$", labelspacing=0.3, handletextpad=0.5, framealpha=0,
                      title_fontsize=legsize+5, handlelength=1.5, bbox_to_anchor=(-0.08, -0.12), loc="lower left")#, bbox_to_anchor=(1.25, 1.18), loc="center")

    
    axes[0, 1].set_ylabel(r"$\varphi$", size=size, labelpad=labelpad)
    axes[0, 1].set_xlabel(r"$t[s]$", size=size, labelpad=labelpad)
    axes[0, 1].yaxis.set_major_locator(MajLogLocator(10, 1e0, 1e6, 7))
    axes[0, 1].yaxis.set_minor_locator(MinLogLocator(10, 1e0, 1e6))
    axes[0, 1].set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], [r"$10^{-4}$", "", r"$10^{-2}$", "",  r"$10^{0}$", "", r"$10^{2}$"])
    axes[0, 1].xaxis.set_minor_locator(MinLogLocator(10, 1e-4, 1e2))
    axes[0, 1].tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    # axes[0, 1].legend(ncol=1, fontsize=legsize)
    axes[0, 1].legend(ncol=3, fontsize=legsize, bbox_to_anchor=(-0.25, 1.18), loc="center")


    axes[1, 0].set_ylabel(r"$m_D[M_\odot]$", size=size, labelpad=labelpad)
    axes[1, 0].set_xlabel(r"$t[s]$", size=size, labelpad=labelpad)
    axes[1, 0].yaxis.set_major_locator(MajLogLocator(10, 8e-7, 3e-2, 6))
    axes[1, 0].yaxis.set_minor_locator(MinLogLocator(10, 8e-7, 3e-2))
    axes[1, 0].set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], [r"$10^{-4}$", "", r"$10^{-2}$", "",  r"$10^{0}$", "", r"$10^{2}$"])
    axes[1, 0].xaxis.set_minor_locator(MinLogLocator(10, 1e-4, 1e2))
    axes[1, 0].tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    # axes[1, 0].legend(ncol=1, fontsize=legsize)


    axes[1, 1].set_ylabel(r"$f_{\textrm{GW}}[\textrm{Hz}]$", size=size, labelpad=labelpad)
    axes[1, 1].set_xlabel(r"$t[s]$", size=size, labelpad=labelpad)
    # axes[1, 1].yaxis.set_major_locator(MajLogLocator(10, 1e4, 2e4))
    # axes[1, 1].yaxis.set_minor_locator(MinLogLocator(10, 1e4, 2e4))
    axes[1, 1].set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], [r"$10^{-4}$", "", r"$10^{-2}$", "",  r"$10^{0}$", "", r"$10^{2}$"])
    axes[1, 1].xaxis.set_minor_locator(MinLogLocator(10, 1e-4, 1e2))
    axes[1, 1].tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    # axes[1, 1].legend(ncol=1, fontsize=legsize)
    axes[1, 1].set_ylim(3.5e3, 6e3)
    axes[1, 1].set_yticks([4e3, 5e3, 6e3], [4, 5, 6])
    axes[1, 1].text(1.1e-4, 6.1e3, r"$\times10^3$", size=size)
    
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.91, hspace=0.5, wspace=0.5)
    # plt.show()
    fig.savefig("OrbitEvolution.pdf")
    
# Evolution()

def Plotfgw():
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    size = 25
    legsize=16
    labelpad = 2.5
    lss = ["-", "--", ":"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # data = []
    # data = np.load("OrbitEvolution.npy", allow_pickle=True)
    
    for model in range(3):
        # dat = []
        for i in range(4):
            # da = data[model, i]
            mass_magnitude = i+3
            filename = "Mm%d_orbit_model%d.txt" % (mass_magnitude, model)
            da = np.loadtxt(filename)
            t = da[:, 0]
 
            f1 = interp1d(da[:, 0], da[:, 1], fill_value="extrapolate")
            f2 = interp1d(da[:, 0], da[:, 2]/MSUN, fill_value="extrapolate")
            f3 = interp1d(da[:, 0], da[:, 3], fill_value="extrapolate")
            f4 = interp1d(da[:, 0], da[:, 4]/pi, fill_value="extrapolate")

            tr = interp1d(da[:, 1], da[:, 0], fill_value="extrapolate")

            def tempf(t):
                return (f2(t) - 0.01)**2
            tend = minimize(tempf, t[-1]).x[0]
            rend = f1(tend)
            # r = np.hstack((r, tend))
            r = 10**np.linspace(log10(da[0, 1]), log10(da[-1, 1]), 1000)
            print(filename, da.shape)
            # dat.append(da)
            ax.loglog(r, f4(tr(r)), color=colors[i], ls=lss[model], label=(r"$10^{-%d}$"%mass_magnitude) if model == 0 else None)
    #     data.append(np.array(dat, dtype=object))
    # data = np.array(data)
    # print(data.shape)
    # np.save("OrbitEvolution.npy", np.array(data, dtype=object))
    
    ax.set_ylabel(r"$f_\mathrm{GW[Hz]}$", size=size, labelpad=labelpad)
    ax.set_xlabel(r"$r[m]$", size=size, labelpad=labelpad)
    ax.xaxis.set_major_locator(MajLogLocator(10, 1e1, 1e4))
    ax.xaxis.set_minor_locator(MinLogLocator(10, 1e1, 1e4))
    ax.set_ylim(3.5e3, 6e3)
    ax.set_yticks([4e3, 5e3, 6e3], [4, 5, 6])
    ax.text(1.1e-4, 6.1e3, r"$\times10^3$", size=size)
    ax.tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    ax.legend(ncol=1, fontsize=legsize+5, title=r"   $m_D[M_\odot]$", labelspacing=0.3, handletextpad=0.5, framealpha=0,
                      title_fontsize=legsize+5, handlelength=1.5, bbox_to_anchor=(-0.08, -0.12), loc="lower left")#, bbox_to_anchor=(1.25, 1.18), loc="center")
    plt.show()

# Plotfgw()

def PlotV():
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    size = 25
    legsize=16
    labelpad = 2.5
    lss = ["-", "--", ":"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # data = []
    # data = np.load("OrbitEvolution.npy", allow_pickle=True)
    for model in range(3):
        # dat = []
        for i in range(4):
            # da = data[model, i]
            mass_magnitude = i+3
            filename = "Mm%d_orbit_model%d.txt" % (mass_magnitude, model)
            da = np.loadtxt(filename)
            # dat.append(da)
            ax.semilogx(da[:, 0], da[:, 4]*da[:, 1]/C, color=colors[i], ls=lss[model])
    #     data.append(np.array(dat, dtype=object))
    # data = np.array(data)
    # print(data.shape)
    # np.save("OrbitEvolution.npy", np.array(data, dtype=object))
    # axes[1, 0].legend(ncol=1, fontsize=legsize)

    ax.set_ylabel(r"$v/c$", size=size, labelpad=labelpad)
    ax.set_xlabel(r"$t[s]$", size=size, labelpad=labelpad)
    # axes[1, 1].yaxis.set_major_locator(MajLogLocator(10, 1e4, 2e4))
    # axes[1, 1].yaxis.set_minor_locator(MinLogLocator(10, 1e4, 2e4))
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], ["", "", r"$10^{-2}$", "",  r"$10^{0}$", "", r"$10^{2}$"])
    ax.xaxis.set_minor_locator(MinLogLocator(10, 1e-4, 1e2))
    ax.tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    # axes[1, 1].legend(ncol=1, fontsize=legsize)
    # ax.set_ylim(3.5e3, 6e3)
    # ax.set_yticks([4e3, 5e3, 6e3], [4, 5, 6])
    # ax.text(1.1e-4, 6.1e3, r"$\times10^3$", size=size)
    
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.91, hspace=0.5, wspace=0.5)
    plt.show()
    # fig.savefig("PBHVelocity.pdf")
    
# PlotV()

def GetAmp(md, r, dl, iota, phi, omega):
    # amp = 4*G*md*MSUN/(dl*C**2)*(G*m/C**3*omega)**(2/3)
    # hp = amp*cos(2*phi)*(1+cos(iota)**2)/2
    # hc = -amp*sin(2*phi)*cos(iota)
    # return (hp**2+hc**2)**0.5
    amp = 4*G*md*MSUN/(dl*C**2)*(omega*r/C)**2
    return amp*( (0.5*(1+cos(iota)**2))**2 + cos(iota)**2 )**0.5


def plotGW(iota, dl):
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    size = 30
    legsize=20
    labelpad = 2.5
    lss = ["-", "--", ":"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # data = []
    # data = np.load("OrbitEvolution.npy", allow_pickle=True)
    for model in range(3):
        # dat = []
        mrdata = np.loadtxt("model%dm.txt"%model)
        f1 = interp1d(mrdata[:, 0], mrdata[:, 1])
        for i in range(4):
            # da = data[model, i]
            mass_magnitude = i+3
            filename = "Mm%d_orbit_model%d.txt" % (mass_magnitude, model)
            print(filename)
            da = np.loadtxt(filename)
            amp = GetAmp(da[:, 2]/MSUN, da[:, 1], dl, iota, da[:, 3], da[:, 4])
            # dat.append(da)
            
            axes[0].loglog(da[:, 0], amp, color=colors[i], ls=lss[model], label=Names[model] if i==0 else None)
            axes[1].loglog(da[:, 1], amp, color=colors[i], ls=lss[model], label=Names[model] if i==0 else None)
            # dat.append(da)

        # data.append(np.array(dat, dtype=object))
    # data = np.array(data)
    # print(data.shape)
    # np.save("OrbitEvolution.npy", np.array(data, dtype=object))\

    rtest = np.array([100, 10000])
    axes[1].loglog(rtest, 1.5e-26*rtest**0.5, color='k', label=r'$\sqrt{r}$')

    axes[0].set_ylabel(r"$|h|$", size=size, labelpad=labelpad)
    axes[0].set_xlabel(r"$t[s]$", size=size, labelpad=labelpad)
    axes[0].legend(ncol=1, fontsize=legsize, bbox_to_anchor=(0.8, 0.75), loc="center", framealpha=0.4)
    axes[0].yaxis.set_major_locator(MajLogLocator(10, 2e-25, 3e-20, 10))
    axes[0].yaxis.set_minor_locator(MinLogLocator(10, 2e-25, 3e-20))
    axes[0].xaxis.set_major_locator(MajLogLocator(10, 1e-4, 2e2, 10))
    axes[0].set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], [r"$10^{-4}$", "", r"$10^{-2}$", "",  r"$10^{0}$", "", r"$10^{2}$"])
    axes[0].xaxis.set_minor_locator(MinLogLocator(10, 1e-4, 2e2))
    axes[0].tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    axes[0].tick_params(axis='both', which='major', length=5)
    axes[0].tick_params(axis='both', which='minor', length=3)
    
    axes[0].text(2e-4, 5e-21, r"$10^{-3}\ M_\odot$", size=size, color=colors[0])
    axes[0].text(2e-4, 5e-22, r"$10^{-4}\ M_\odot$", size=size, color=colors[1])
    axes[0].text(2e-4, 5e-23, r"$10^{-5}\ M_\odot$", size=size, color=colors[2])
    axes[0].text(2e-4, 5e-24, r"$10^{-6}\ M_\odot$", size=size, color=colors[3])

    axes[1].set_ylabel(r"$|h|$", size=size, labelpad=labelpad)
    axes[1].set_xlabel(r"$r[m]$", size=size, labelpad=labelpad)
    axes[1].legend(ncol=1, fontsize=legsize, framealpha=0.4) #, bbox_to_anchor=(0.5, 0.95), loc="center")
    axes[1].yaxis.set_major_locator(MajLogLocator(10, 1e-25, 3e-20, 10))
    axes[1].yaxis.set_minor_locator(MinLogLocator(10, 1e-25, 3e-20))
    axes[1].xaxis.set_major_locator(MajLogLocator(10, 6e1, 1.1e4, 10))
    # ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], [r"$10^{-4}$", "", r"$10^{-2}$", "",  r"$10^{0}$", "", r"$10^{2}$"])
    axes[1].xaxis.set_minor_locator(MinLogLocator(10, 6e1, 1.1e4))
    axes[1].tick_params(axis='both', which='both', top=True, labeltop=False, right=True, labelright=False, labelsize=size)
    axes[1].tick_params(axis='both', which='major', length=5)
    axes[1].tick_params(axis='both', which='minor', length=3)
    
    axes[1].text(1e3, 1e-21, r"$10^{-3}\ M_\odot$", size=size, color=colors[0], rotation=12)
    axes[1].text(1e3, 1e-22, r"$10^{-4}\ M_\odot$", size=size, color=colors[1], rotation=12)
    axes[1].text(1e3, 1e-23, r"$10^{-5}\ M_\odot$", size=size, color=colors[2], rotation=12)
    axes[1].text(1e3, 1e-24, r"$10^{-6}\ M_\odot$", size=size, color=colors[3], rotation=12)
    fig.subplots_adjust(left=0.25, bottom=0.13, right=0.97, top=0.97, hspace=0.4)
    # plt.show()
    fig.savefig("GWAmp.pdf")

# plotGW(pi/3, 0.01*Mpc)