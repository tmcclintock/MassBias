"""Fit the averaged delta sigma profiles.
"""
from catalog import *
import numpy as np
import cluster_toolkit as ct


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
    #data = np.load("testdata/reduced_halos_lamobs_0.20sigintr_009.npy")
    sigs = np.arange(0.05, 0.45, step=0.05)
    inds = [6,7,8,9]
    bins = np.array([20,30,45,60,999])
    zs = [1.0, 0.5, 0.25, 0.0]    

    covpath = "/Users/tmcclintock/Data/DATA_FILES/y1_data_files/FINAL_FILES/SACs/SAC_z%d_l%d.txt"
    datapath = ""

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

            #Distances
            r = np.logspace(-3, 3, 1000) #Mpc/h comoving
            Rperp = np.logspace(-3, 2.4, 1000) #Mpc/h comoving

            
