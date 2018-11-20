"""Fit the averaged delta sigma profiles.
"""
from catalog import *
import numpy as np
import cluster_toolkit as ct
import scipy.optimize as op
import matplotlib.pyplot as plt

def get_model(M, args):
    Redges = args['Redges']
    Rlam = args['Rlam']
    h = args['h']
    Om = args['Om']
    z = args['z']
    r = args['r3d'] #Mpc/h comoving
    Rperp = args['Rperp'] #Mpc/h comoving
    SCI = args['SCI']
    k = args['k']
    Plin = args['Plin']
    xi_mm = args['xi_mm']
    c,tau,fmis,Am,B0,Rs = args['params']
    #c = ct.concentration.concentration_at_M(M, k, Plin, ns, Ob, Om, h, Mass_type='mean')
    xi_nfw = ct.xi.xi_nfw_at_R(r, M, c, Om)
    bias = ct.bias.bias_at_M(M,k,Plin,Om)
    xi_2halo = ct.xi.xi_2halo(bias, xi_mm)
    xi_hm = ct.xi.xi_hm(xi_nfw, xi_2halo)

    #print("%3e"%M)
    #print(bias)
    #rdata = np.loadtxt("testdata/r.txt")
    #xid = np.load("testdata/hmcfs_z006_0.05sigintr.npy")
    #print(xid.shape)
    ##xid = xid[1]
    #plt.loglog(r, xi_hm)
    #plt.loglog(r, xi_mm, ls=':')
    #plt.loglog(rdata, xid)
    #plt.show()
    #exit()
    
    Rmis = tau*Rlam #Mpc/h comoving
    #Sigmas are in Msun h/pc^2 comoving
    Sigma = ct.deltasigma.Sigma_at_R(Rperp, r, xi_hm, M, c, Om)
    Sigma_mis  = ct.miscentering.Sigma_mis_at_R(Rperp, Rperp, Sigma, M, c, Om, Rmis, kernel="exponential")
    full_Sigma = (1-fmis)*Sigma + fmis*Sigma_mis
    kappa = SCI*full_Sigma*h*(1+z)**2
    #DeltaSigmas are in Msun/pc^2 physical
    DeltaSigma     = ct.deltasigma.DeltaSigma_at_R(Rperp, Rperp, Sigma, M, c, Om) *h*(1+z)**2
    DeltaSigma_mis = ct.miscentering.DeltaSigma_mis_at_R(Rperp, Rperp, Sigma_mis) *h*(1+z)**2
    full_DS = (1-fmis)*DeltaSigma + fmis*DeltaSigma_mis

    #Apply corrections
    B = args['boost']
    full_DS *= Am/(B*(1-kappa))
    ave_fDS = ct.averaging.average_profile_in_bins(Redges, Rperp/(h*(1+z)), full_DS)
    
    return ave_fDS

def lnlike(pars, args):
    Mtrue = args['Mass']
    Cal = pars
    M = Mtrue/Cal
    DSmodel = get_model(M, args)[args['inds']]
    #Get the data
    DSd = args['DSd']
    icov = args['icov']
    X = DSd - DSmodel
    chi2 = -0.5*np.dot(X,np.dot(icov,X))
    return chi2

if __name__ == "__main__":
    #Load in the halo catalog
    sigs = np.arange(0.05, 0.45, step=0.05)
    inds = [6,7,8,9]
    bins = np.array([20,30,45,60,999])
    zs = [1.0, 0.5, 0.25, 0.0]
    zmap = [2,1,0,0] #Map from fox zi to data zi, for SAC matrices

    covpath = "/Users/tom/Data/DESY1/RMWL/SACs/SAC_z%_l%d.txt"
    datapath = "ds_testdata/DSave_z%03d_%.2fsigintr.npy"
    halopath = "/Users/tom/Data/DESY1/RMWL/fox_files/halo_catalogs/reduced_halos_lamobs_%.2fsigintr_%03d.npy"

    #Output path
    outpath = "calibration_fits/result_%.2fsigintr.npy"

    for sig in sigs:
        outarray = np.zeros((6, 16)) #6 columns, 16 rows for each z-Lambda bin in the sim
        #zindex, lindex, Mtrue, lambda, cal, calunc

        for i,ind in enumerate(inds):
            print(i,ind)
            outarray[0, i*4:(i+1)*4]  = ind #Z index
            outarray[1, i*4:(i+1)*4] = np.arange(4)+3 #l index
            
            zid = zmap[i] #Z index for data
            z = zs[i]
            deltap1s = np.loadtxt("Y1_deltap1.txt") #pz biases
            SCIs = np.loadtxt("sigma_crit_inv.txt")
            boost_params = np.load("boost_params.npy")
            #print("Sigma crit inv shape: ",SCIs.shape)
            #print("boost params shape:   ",boost_params.shape)

            #Load in some data
            DS_all = np.load(datapath%(ind, sig))
            Redges = np.loadtxt("ds_testdata/Redges.txt")
            R = (Redges[1:] + Redges[:-1])/2 #Mpc phys; midpoint of
            rinds = (R < 999.)*(R > 0.2) #apply cuts
            R = R[rinds]

            #Load in the halo catalog
            halos = np.load(halopath%(sig, ind))
            cat = halo_catalog(halos, bins)
            masses = cat.mean_masses[1:]
            lams = cat.mean_observable[1:]
            Rlams = (lams/100.)**0.2 #Mpc/h comoving
            outarray[2, i*4:(i+1)*4] = masses
            outarray[3, i*4:(i+1)*4] = lams

            #Fox cosmology
            Om = 0.318
            h  = 0.6704
            Ob = 0.049
            ns = 0.962

            #Precompute some things
            k = np.loadtxt("testdata/k.txt")
            Plin = np.loadtxt("testdata/plin_z%d.txt"%(ind-6))
            Pnl = np.loadtxt("testdata/pnl_z%d.txt"%(ind-6))
            #Distances for the modeling
            r = np.logspace(-3, 3, 1000) #Mpc/h comoving
            Rperp = np.logspace(-3, 2.4, 1000) #Mpc/h comoving
            xi_mm = ct.xi.xi_mm_at_R(r, k, Pnl, exact=True)
            
            #Default parameters for the lensing model
            tau = 0.17
            fmis = 0.25
            boostpars = np.load("boost_params.npy")
            
            #Loop over the mass bins
            for lj in range(len(masses)):
                Mtrue = masses[lj]
                lamtrue = lams[lj]
                Rlam = Rlams[lj]
                cov = np.loadtxt("/Users/tom/Data/DESY1/RMWL/SACs/SAC_z%d_l%d.txt"%(zid, lj+3))
                DSd = DS_all[lj]
                DSd = DSd[rinds]
                cov = cov[rinds]
                cov = cov[:,rinds]
                B0, Rs = boost_params[zid,lj]
                boost = ct.boostfactors.boost_nfw_at_R(Rperp/(h*(1+z)), B0, Rs)
                SCI = SCIs[zid,lj+3]
                deltap1 = deltap1s[zid,lj+3]
                m = 0.012 #shear bias
                Am = deltap1 + m
                c = ct.concentration.concentration_at_M(Mtrue, k, Plin, ns, Ob, Om, h, Mass_type='mean')
                params = [c,tau,fmis,Am,B0,Rs] #lensing model parameters
                args={'R':R,'DSd':DSd,'icov':np.linalg.inv(cov),'cov':cov,'err':np.sqrt(cov.diagonal()),
                      'xi_mm':xi_mm,'Redges':Redges, 'SCI':SCI,'Rlam':Rlam, 'lam':lamtrue,'z':z,
                      'boost':boost,'params':params, "Mass":Mtrue, 'r3d':r, 'Rperp':Rperp,
                      'k':k,'Plin':Plin,'Om':Om,'h':h, 'Rmid':R, 'inds':rinds}

                #Do the optimization
                print("Z%d L%d sig=%.2f"%(ind, lj+3, sig))
                print("\tMtrue = %.3e\tz = %.2f"%(Mtrue,z))
                guess = 1.00
                nll = lambda *args: -lnlike(*args)
                result = op.minimize(nll,guess,args=(args,),tol=1e-3,method='BFGS')
                unc = np.sqrt(result['hess_inv'][0])
                print("\tresult: %.3f +- %.3f"%(result.x, unc))
                print(i*4+lj)
                outarray[4, i*4 + lj] = result.x
                outarray[5, i*4 + lj] = unc

                #Mbest = Mtrue#/result['x']
                #DSmodel = get_model(Mbest, args)[args['inds']]
                #print(ind, sig)
                #Rpdata = np.loadtxt("ds_testdata/Rp.txt") #Mpc/h comoving
                #dsc = np.load("ds_testdata/DSs_z%03d_%.2fsigintr.npy"%(ind,sig))
                #print(dsc.shape, lj)
                #dsc = dsc[lj]
                #plt.errorbar(args['R'], args['DSd'], args['err']) #Msun/pc^2 phys
                #plt.loglog(args['R'], ave_fDS[args['inds']]) #Msun/pc^2 phys
                #plt.loglog(Rpdata/(h*(1+z)), dsc*h*(1+z)**2, ls=':') #Msun/pc^2 phys
                #plt.show()
                #exit()
                continue
            continue
        print("Saving results for sig=%.2f"%sig)
        np.save(outpath%sig, outarray)
