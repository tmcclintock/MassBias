"""Take in correlation functions and a true mass and compute DeltaSigma profiles.
"""
from catalog import *
import numpy as np
import cluster_toolkit as ct

class cf2ds_converter(object):
    def __init__(self, r, hmcf, Mtrue):
        """Constructor for the converter.

        Args:
            r (array like): 3D radi of the correlation function; Mpc/h
            hmcf (array like): Halo-matter correlation function
            Mtrue (float): True mean mass of the stack

        """
        self.r = np.ascontiguousarray(r, dtype='float64')
        self.hmcf = np.ascontiguousarray(hmcf, dtype='float64')
        self.Mtrue = Mtrue

    def set_cosmology(self, k, Plin, Pnl, ns, Omega_b, Omega_m, h):
        self.k = k
        self.Plin = Plin
        self.Pnl = Pnl
        self.ns = ns
        self.Omega_b = Omega_b
        self.Omega_m = Omega_m
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
        xi_mm = ct.xi.xi_mm_at_R(rmodel, self.k, self.Pnl, exact=True)
        xi_nfw = ct.xi.xi_nfw_at_R(rmodel, M, self.conc, self.Omega_m)
        bias = ct.bias.bias_at_M(M, self.k, self.Plin, self.Omega_m)
        xi_2halo = ct.xi.xi_2halo(bias, xi_mm)
        xi_hm = ct.xi.xi_hm(xi_nfw, xi_2halo)
        self.rmodel = rmodel
        self.xim = xi_hm
        return

        
    def make_fixed_hmcf(self, lowcut=0.2, highcut=80.):
        """Clip the data at the ends and insert it into a model curve.
        """
        rm = self.rmodel
        xim = self.xim
        r = self.r
        xi = self.hmcf
        lowcut = np.max([lowcut, r[0]])
        highcut = np.min([highcut, r[-1]])
        rout, xiout = [], []
        xiout = []
        inds = (Rd > lowcut)*(xi>1e-3)*(Rd < highcut)
        r = r[inds]
        xi = xi[inds]
        ind = 0
        for i in range(len(rm)):
            if ind >= len(r):
                rout.append(rm[i])
                xiout.append(xim[i])
                continue
            if rm[i] < r[ind]:
                rout.append(rm[i])
                xiout.append(xim[i])
                continue
            else:
                rout.append(r[ind])
                xioiut.append(xi[ind])
                ind += 1
                continue
            continue
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
        
    def calc_DSmis(self, r, hmcf, R,
                   Rmis, kernel="exponential"):
        M = self.Mtrue
        Sigma = ct.deltasigma.Sigma_at_R(R, r, hmcf, M, self.conc, self.Omega_m)
        Sigma_mis = ct.miscentering.Sigma_mis_at_R(R, R, Sigma, M, self.conc, self.Omega_m, Rmis, kernel)
        return Sigma_mis, ct.miscentering.Deltasigma_mis_at_R(R, R, Sigma_mis)

    def apply_systematics(self, r, hmcf, R, A=None,
                          boostpars = None,
                          Rmis = None, fmis = None,
                          Sigma_crit_inv = None):
        S, DS = self.calc_DS(r, hmcf, R)
        if (Rmis is not None) and (fmis is not None):
            Sm, DSm = self.calc_DSm(r, hmcf, R, Rmis)
            S = (1-fmis)*S + fmis*Sm
            DS = (1-fmis)*DS + fmis*DSm
        if A is not None:
            DS *= A
        if boostpars is not None:
            B0, Rs = boostpars
            B = ct.boostfactors.boost_nfw_at_R(R, B0, Rs)
            DS /= B
        if Sigma_crit_inv is not None:
            kappa = Sigma_crit_inv * S
            DS /= (1-kappa)
        #DS now has all systematics applied
        return DS

    def average_in_bins(self, R, DS, R_edges):
        return ct.averaging.average_profile_in_bins(R_edges, R, DS)

if __name__ == "__main__":
    #Load in the halo catalog
    data = np.load("testdata/reduced_halos_lamobs_009.npy")
    bins = np.array([20,30,45,60,999])
    cat = halo_catalog(data, bins)
    masses = cat.mean_masses

    #Fox cosmology
    Om = 0.318
    h = 0.6704
    Ob = 0.049
    ns = 0.962

    #load in some test data here
    k = np.loadtxt("testdata/k.txt")
    Plin = np.loadtxt("testdata/plin_z3.txt")
    Pnl = np.loadtxt("testdata/pnl_z3.txt") #z=0
    
    #Load in some hmcfs.
    r = np.loadtxt("testdata/r.txt")
    hmcfs = np.load("testdata/hmcfs_z009.npy")
    rmodel = np.logspace(-3, 3, num=1000) #Mpc/h comoving
    Rp = np.logspace(-3, 2.4, num=1000) #Mpc/h comoving

    import matplotlib.pyplot as plt
    for i in range(len(masses)):
        xi = hmcfs[i]
        conv = cf2ds_converter(r, xi, masses[i])
        conv.set_cosmology(k, Plin, Pnl, ns, Ob, Om, h)
        conv.calc_concentration()
        xim = conv.calc_xihm_model(rmodel)
        plt.loglog(r, xi)
        plt.loglog(rmodel, xim)
        plt.show()
        exit()
