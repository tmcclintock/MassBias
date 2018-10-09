import numpy as np
import Corrfunc
import math


def partition(data, boxsize, gridsize):

    Nperdim = int(math.ceil(boxsize / gridsize))                            # Number of grid cells per side.
    grid_id = data[:] / gridsize                                            # Grid ID for each data point.
    grid_id = grid_id.astype(int)
    grid_id[np.where(grid_id == Nperdim)] = Nperdim - 1                     # Correction for points on the edges.

    data_id = [[] for i in range(Nperdim**3)]                               # List of size Nperdim*Nperdim*Nperdim.
                                                                            #  This list stores all of the particles
                                                                            # original IDs in a convenient 3D list.
                                                                            # It is kind of a pointer.
    for i in range(np.size(data, 0)):
        s = Nperdim**2 * grid_id[i, 0] + Nperdim * grid_id[i, 1] + grid_id[i, 2]
        data_id[s].append(i)

    return data_id


def DD_jk(d1, d2, d1_id, d2_id, boxsize, gridsize, minsep, maxsep, nbins, nthreads):

    Nperdim = int(math.ceil(boxsize / gridsize))

    # Set up bins
    bins = 10. ** np.linspace(np.log10(minsep), np.log10(maxsep), nbins + 1)

    # Create pairs list
    ddpairs = [[[] for t in range(Nperdim**3)]
                   for s in range(Nperdim**3)]
    zeros = np.zeros(nbins)

    # Pair counting
    # Loop for each minibox
    print('Pair count.')
    for i1, j1, k1 in [(i1, j1, k1) for i1 in range(Nperdim) for j1 in range(Nperdim) for k1 in range(Nperdim)]:

        # Get box index
        #print(i1, j1, k1)
        s1 = Nperdim ** 2 * i1 + Nperdim * j1 + k1

        # Get data1 box
        xd1 = d1[d1_id[s1], 0]
        yd1 = d1[d1_id[s1], 1]
        zd1 = d1[d1_id[s1], 2]

        # Get in each box: self and 8 neighbors
        for a in [-1, 0, 1]:
            i2 = i1 + a                         # i2 is the index for data2 box.
            periodi = 0                         # periodi is to apply periodic boundary conditions
            if i2 == -1:
                i2 = Nperdim - 1
                periodi = -1
            elif i2 == Nperdim:
                i2 = 0
                periodi = 1

            for b in [-1, 0, 1]:
                j2 = j1 + b
                periodj = 0
                if j2 == -1:
                    j2 = Nperdim - 1
                    periodj = -1
                elif j2 == Nperdim:
                    j2 = 0
                    periodj = 1

                for c in [-1, 0, 1]:
                    k2 = k1 + c
                    periodk = 0
                    if k2 == -1:
                        k2 = Nperdim - 1
                        periodk = -1
                    elif k2 == Nperdim:
                        k2 = 0
                        periodk = 1

                    # Get data2 box
                    s2 = Nperdim ** 2 * i2 + Nperdim * j2 + k2

                    # Get data2 box
                    xd2 = d2[d2_id[s2], 0] + periodi * boxsize
                    yd2 = d2[d2_id[s2], 1] + periodj * boxsize
                    zd2 = d2[d2_id[s2], 2] + periodk * boxsize

                    # Data pair counting
                    if np.size(xd1) != 0 and np.size(xd2) != 0:
                        autocorr = 0
                        DD_counts = Corrfunc.theory.DD(autocorr, nthreads, bins, xd1, yd1, zd1, X2=xd2, Y2=yd2, Z2=zd2, periodic=False, verbose=False)
                        DD_counts = [DD_counts[m][3] for m in range(np.size(DD_counts, 0))]
                        DD_counts = np.array(DD_counts)
                        ddpairs[s1][s2].append(DD_counts)
                        del DD_counts
                    else:
                        ddpairs[s1][s2].append(zeros)

                    #gc.collect()

    return ddpairs


def DD_to_tpcf_jk(d1, d2, d1_id, d2_id, boxsize, gridsize, minsep, maxsep, nbins, ddpairs):

    # Set up bins
    bins = 10. ** np.linspace(np.log10(minsep), np.log10(maxsep), nbins + 1)

    # Some quantities
    Nperdim = int(math.ceil(boxsize / gridsize))                            # Number of cells per dimension
    N = Nperdim**3                                                          # Number of jackknife samples
    d1tot = np.size(d1, 0)                                                  # Number of objects in d1
    d2tot = np.size(d2, 0)                                                  # Number of objects in d2
    Vbox = boxsize ** 3                                                     # Volume of box
    Vshell = np.zeros(nbins)                                                # Volume of spherical shell
    for m in range(nbins):
        Vshell[m] = 4. / 3. * np.pi * (bins[m + 1] ** 3 - bins[m] ** 3)
    n2 = float(d2tot) / Vbox                                                # Number density of d2

    # Some arrays
    dd_pairs = np.zeros(nbins)
    dd_pairs_i = np.zeros((N, nbins))
    xi = np.zeros(nbins)
    xi_i = np.zeros((N, nbins))
    meanxi_i = np.zeros(nbins)
    cov = np.zeros((nbins, nbins))

    # Loop for every d1 box
    for i1, j1, k1 in [(i1, j1, k1) for i1 in range(Nperdim)
                                    for j1 in range(Nperdim)
                                    for k1 in range(Nperdim)]:

        # Get data1 box
        s1 = Nperdim ** 2 * i1 + Nperdim * j1 + k1

        # Get into each minibox: self and 8 neighbors
        for a in [-1, 0, 1]:
            i2 = i1 + a  # i2 is the index for data2 box.
            if i2 == -1:
                i2 = Nperdim - 1
            elif i2 == Nperdim:
                i2 = 0

            for b in [-1, 0, 1]:
                j2 = j1 + b
                if j2 == -1:
                    j2 = Nperdim - 1
                elif j2 == Nperdim:
                    j2 = 0

                for c in [-1, 0, 1]:
                    k2 = k1 + c
                    if k2 == -1:
                        k2 = Nperdim - 1
                    elif k2 == Nperdim:
                        k2 = 0

                    # Get data2 box
                    s2 = Nperdim ** 2 * i2 + Nperdim * j2 + k2

                    # Substract pairs from s1 box
                    dd_pairs_i[s1] = dd_pairs_i[s1] - ddpairs[s1][s2]

                    # Sum pairs for xi over entire sample
                    dd_pairs = dd_pairs + ddpairs[s1][s2]

    # Compute xi_i
    for s1 in range(N):
        # Sum pairs
        dd_pairs_i[s1] = dd_pairs_i[s1] + dd_pairs
        d1tot_s1 = np.size(d1, 0) - np.size(d1[d1_id[s1]], 0)
        n1 = float(d1tot_s1) / Vbox
        xi_i[s1] = dd_pairs_i[s1] / (n1 * n2 * Vbox * Vshell) - 1

    # Compute meanxi_i
    for i in range(nbins):
        meanxi_i[i] = np.mean(xi_i[:, i])

    # Compute covariance matrix
    cov = (float(N) - 1.) * np.cov(xi_i.T, bias=True)

    # Compute xi
    n1 = float(d1tot) / Vbox
    xi = dd_pairs / (n1 * n2 * Vbox * Vshell) - 1

    return xi, xi_i, meanxi_i, cov


def cross_tpcf_jk(d1, d2, boxsize, gridsize, minsep, maxsep, nbins, nthreads, jk_estimates = False):

    # Partition boxes
    print('Partition.')
    d1_id = partition(d1, boxsize, gridsize)
    d2_id = partition(d2, boxsize, gridsize)

    # Pair counting
    print('DD pair counting.')
    ddpairs = DD_jk(d1, d2, d1_id, d2_id, boxsize, gridsize, minsep, maxsep, nbins, nthreads)

    # Compute xi
    print('Compute xi_hm.')
    xi, xi_i, meanxi_i, cov = DD_to_tpcf_jk(d1, d2, d1_id, d2_id, boxsize, gridsize, minsep, maxsep, nbins, ddpairs)

    # Return estimators
    if jk_estimates is True:
        return meanxi_i, cov, xi_i, xi
    else:
        return meanxi_i, cov


def cross_tpcf_jk_fixed_d1(d1, d1_id, d2, boxsize, gridsize, minsep, maxsep, nbins, nthreads, jk_estimates = False):

    # Partition boxes
    print('Partition.')
    d2_id = partition(d2, boxsize, gridsize)

    # Pair counting
    print('DD pair counting.')
    ddpairs = DD_jk(d1, d2, d1_id, d2_id, boxsize, gridsize, minsep, maxsep, nbins, nthreads)

    # Compute xi
    print('Compute xi_hm.')
    xi, xi_i, meanxi_i, cov = DD_to_tpcf_jk(d1, d2, d1_id, d2_id, boxsize, gridsize, minsep, maxsep, nbins, ddpairs)

    # Return estimators
    if jk_estimates is True:
        return meanxi_i, cov, xi_i, xi
    else:
        return meanxi_i, cov