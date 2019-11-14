"""Take in correlation functions and a true mass and compute DeltaSigma profiles.
"""
#from catalog import *
import numpy as np
import cluster_toolkit as ct
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import aemulus_extras
import matplotlib.pyplot as plt

#Redshift of aemulus snapshots
zs = np.array([3.        , 2.000003  , 1.        , 0.84999843, 0.70000085,
               0.5500007 , 0.39999944, 0.25      , 0.09999989, 0.        ])

class cf2ds_converter(object):
    def __init__(self, r, hmcf, Mtrue):
        """Constructor for the converter.

        Args:
            r (array like): 3D radi of the correlation function; Mpc/h comoving
            hmcf (array like): Halo-matter correlation function
            Mtrue (float): True mean mass of the stack; Msun/h

        """
        self.r = np.ascontiguousarray(r, dtype='float64')
        self.hmcf = np.ascontiguousarray(hmcf, dtype='float64')
        self.Mtrue = Mtrue

    def set_cosmology(self, index, zi):
        e = aemulus_extras.Extras(index)
        self.k = e.k
        self.Plin = e.P_lin[zi]
        self.Pnl = e.P_nl[zi]
        cos = e.cosmology
        h = cos[-3]/100.
        self.ns = cos[3]
        self.Omega_b = cos[0]/h**2
        self.Omega_m = (cos[0] + cos[1])/h**2
        self.h = h
        return

    def calc_concentration(self):
        M = self.Mtrue
        c = ct.concentration.concentration_at_M(M, self.k, self.Plin,
                                                self.ns, self.Omega_b,
                                                self.Omega_m, self.h,
                                                Mass_type='mean')
        self.conc = c
        return c

        
    def calc_xihm_model(self, rmodel):
        M = self.Mtrue
        xi_mm = ct.xi.xi_mm_at_r(rmodel, self.k, self.Pnl, exact=True)
        xi_nfw = ct.xi.xi_nfw_at_r(rmodel, M, self.conc, self.Omega_m)
        bias = ct.bias.bias_at_M(M, self.k, self.Plin, self.Omega_m)
        xi_2halo = ct.xi.xi_2halo(bias, xi_mm)
        xi_hm = ct.xi.xi_hm(xi_nfw, xi_2halo)
        self.rmodel = rmodel
        self.xim = xi_hm
        return
        
    def make_fixed_hmcf(self, lowcut=0.2, highcut=70.):
        """Clip the data at the ends and insert it into a model curve.
        """
        rm = self.rmodel
        xim = self.xim
        r = self.r
        xi = self.hmcf
        lowcut = np.max([lowcut, r[0]])
        highcut = np.min([highcut, r[-1]])
        inds = (r > lowcut)*(xi>1e-3)*(r < highcut) #note we shave off negatives
        r = r[inds]
        xi = xi[inds]
        ind = 0
        low = rm < r[0]
        rout = np.copy(r)
        xiout = np.copy(xi)
        rout = np.concatenate((rm[low], rout))
        xiout = np.concatenate((xim[low], xiout))
        high = rm > r[-1]
        rout = np.concatenate((rout, rm[high]))
        xiout = np.concatenate((xiout, xim[high]))
        #xiout_spline = IUS(rout, xiout)
        #newrout = np.logspace(np.log10(rm[0]), np.log10(rm[-1]), len(rm)*10)
        #newxiout = xiout_spline(newrout)
        self.r_fixed = rout
        self.xi_fixed = xiout
        return
    
    def calc_DS(self, r, hmcf, R):
        """Convert a correlation function to a DeltaSigma.

        Returns:
            (array like): DeltaSigma profile in hMsun/pc^2 comoving.
        """
        M = self.Mtrue
        Sigma = ct.deltasigma.Sigma_at_R(R, r, hmcf, M, self.conc, self.Omega_m)
        return Sigma, ct.deltasigma.DeltaSigma_at_R(R, R, Sigma, M,
                                                    self.conc, self.Omega_m)

if __name__ == "__main__":
    #Set up the model
    rmodel = np.logspace(-3, 3, num=1000) #Mpc/h comoving
    Rp = np.logspace(-2, 2.4, num=1000) #Mpc/h comoving
    r = np.load("xihm_data/r.npy")

    #Load everything in
    for zi in range(0, 10):
        masses = np.load("xihm_data/masses_Box34_Z%d.npy"%zi)
        xihms = np.load("xihm_data/xihm_Box34_Z%d.npy"%zi)
        z = zs[zi]

        concentrations = np.zeros_like(masses)
        Sout = np.zeros((len(masses), len(Rp)))
        DSout = np.zeros((len(masses), len(Rp)))

        print("Starting Z%d"%zi)
        for i in range(len(masses)):
            xi = xihms[i]

            
            conv = cf2ds_converter(r, xi, masses[i])
            conv.set_cosmology(34, zi)
            concentrations[i] = conv.calc_concentration()
            conv.calc_xihm_model(rmodel)
            conv.make_fixed_hmcf()
            rf = conv.r_fixed
            xif = conv.xi_fixed
            S, DS = conv.calc_DS(rf, xif, Rp)
            Sout[i] = S
            DSout[i] = DS
            #plt.loglog(Rp, DS, '--')
        #plt.show()
        np.save("output_data/Sigmas_Z%d_Box34"%zi, Sout)
        np.save("output_data/DeltaSigmas_Z%d_Box34"%zi, DSout)
        np.save("output_data/Rp_Mpch_comoving", Rp)
        np.save("output_data/concentrations_Z%d_Box34"%zi, concentrations)
        print("\tDone with Z%d"%zi)
        
