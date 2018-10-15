import numpy as np
import matplotlib.pyplot as plt

Rp = np.loadtxt("Rp.txt")
sigs = np.arange(0.05, 0.45, step=0.05)
inds = [6,7,8,9]
bins = np.array([20,30,45,60,999])

DSs = []
for sig in sigs:
    for i in [9]:
        DS = np.load("./DSs_z%03d_%.2fsigintr.npy"%(i,sig))
        for j in [3]:
            dsj = DS[j]
            plt.loglog(Rp, dsj, label=r'$\sigma = %.2f$'%sig)

plt.xlim(.1,30)
plt.legend(loc=0)
plt.show()
