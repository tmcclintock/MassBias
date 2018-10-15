"""Take in correlation functions and a true mass and compute DeltaSigma profiles.
"""
from catalog import *
import numpy as np
import cluster_toolkit as ct
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

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

        
    def make_fixed_hmcf(self, lowcut=0.2, highcut=70.):
        """Clip the data at the ends and insert it into a model curve.
        """
        rm = self.rmodel
        xim = self.xim
        r = self.r
        xi = self.hmcf
        lowcut = np.max([lowcut, r[0]])
        highcut = np.min([highcut, r[-1]])
        inds = (r > lowcut)*(xi>1e-3)*(r < highcut)
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
        
    def calc_DSmis(self, r, hmcf, R, Rmis, kernel="exponential"):
        M = self.Mtrue
        Sigma = ct.deltasigma.Sigma_at_R(R, r, hmcf, M, self.conc, self.Omega_m)
        Sigma_mis = ct.miscentering.Sigma_mis_at_R(R, R, Sigma, M, self.conc,
                                                   self.Omega_m, Rmis,
                                                   kernel="exponential")
        return Sigma_mis, ct.miscentering.DeltaSigma_mis_at_R(R, R, Sigma_mis)

    def apply_systematics(self, r, hmcf, R, A=None,
                          boostpars = None,
                          Rmis = None, fmis = None,
                          Sigma_crit_inv = None):
        S, DS = self.calc_DS(r, hmcf, R)
        if (Rmis is not None) and (fmis is not None):
            Sm, DSm = self.calc_DSmis(r, hmcf, R, Rmis)
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
    #data = np.load("testdata/reduced_halos_lamobs_0.20sigintr_009.npy")
    sigs = np.arange(0.05, 0.45, step=0.05)
    inds = [6,7,8,9]
    bins = np.array([20,30,45,60,999])
    zs = [1.0, 0.5, 0.25, 0.0]
    for sig in sigs:
        for ind in inds:
            data = np.load("/Users/tmcclintock/Data/halo_catalogs/reduced_halos_lamobs_%.2fsigintr_%03d.npy"%(sig,ind))
            bins = np.array([20,30,45,60,999])
            cat = halo_catalog(data, bins)
            masses = cat.mean_masses
            lams = cat.mean_observable

            #Fox cosmology
            Om = 0.318
            h = 0.6704
            Ob = 0.049
            ns = 0.962

            #load in some test data here
            k = np.loadtxt("testdata/k.txt")
            Plin = np.loadtxt("testdata/plin_z%d.txt"%(ind-6))
            Pnl = np.loadtxt("testdata/pnl_z%d.txt"%(ind-6))
    
            #Load in some hmcfs.
            r = np.loadtxt("testdata/r.txt")
            hmcfs = np.load("testdata/hmcfs_z%03d_%.2fsigintr.npy"%(ind,sig))

            #Set up the model
            rmodel = np.logspace(-3, 3, num=1000) #Mpc/h comoving
            Rp = np.logspace(-2, 2.4, num=1000) #Mpc/h comoving
            
            DSout = np.zeros((len(masses)-1, len(Rp)))
            for i in range(len(masses)):
                if i <1: continue
                xi = hmcfs[i]
                conv = cf2ds_converter(r, xi, masses[i])
                conv.set_cosmology(k, Plin, Pnl, ns, Ob, Om, h)
                conv.calc_concentration()
                conv.calc_xihm_model(rmodel)
                conv.make_fixed_hmcf()
                rf = conv.r_fixed
                xif = conv.xi_fixed

                #Sytematic parameters
                tau = 0.17 #Y1 prior
                fmis = 0.25
                Rlam = (lams[i]/100.)**0.2 #Mpc/h comoving

                Rmis = tau*Rlam
                zmap = [2,1,0,0] #Map from fox zi to data zi
                zid = zmap[ind-6]
                boostpars = np.load("boost_params.npy")
                B0, Rs = np.load("boost_params.npy")[zid,i-1]
                Rs *= h*(1+zs[ind-6]) #convert
                dp1 = np.loadtxt("Y1_deltap1.txt")
                delta_plus_1 = np.loadtxt("Y1_deltap1.txt")[zid, i+2]
                Am = 0.012 + delta_plus_1
                SCI = np.loadtxt("sigma_crit_inv.txt")[zid,i+2]

                """
                _, DS = conv.calc_DS(rf, xif, Rp)
                _, DSm = conv.calc_DSmis(rf, xif, Rp, Rmis)
                import matplotlib.pyplot as plt
                plt.loglog(Rp, DS)
                plt.loglog(Rp, DSm)
                plt.show()
                exit()
                """

                #Apply systematics and compute
                DSs = conv.apply_systematics(rf, xif, Rp, A=Am,
                                             boostpars = [B0, Rs],
                                             Rmis = Rmis, fmis = fmis,
                                             Sigma_crit_inv = SCI)

                DSout[i-1] = DSs


                continue
            print("Done with sig%.2f ind%d"%(sig,ind))
            np.savetxt("ds_testdata/Rp.txt",Rp, header="Mpc/h h=%.4f"%h)
            np.save("ds_testdata/DSs_z%03d_%.2fsigintr"%(ind,sig), DSout)
