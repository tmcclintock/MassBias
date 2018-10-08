"""Take in a halo catalog and perform splits on it.
"""
import numpy as np

class halo_catalog(object):
    def __init__(self, halo_array, binning, binning_index=-1):
        """Constructor for the catalog.

        Args:
            halo_array (array-like): Contains x,y,z,M,...
            binning (array-like): Bin edges for whatever property is being used to split the halos.
            binning_index (int): Column index for halo_array that will be used for binning.

        """
        assert np.ndim(halo_array) == 2, "halo_array must be 2D: N_halos x N_properties"
        assert len(halo_array[0]) > 3, "halo_array must have more attributes than just positions."
        self.halos   = halo_array
        self.binning = binning
        self.index   = binning_index
        self.get_number_per_bin
        self.get_mean_observable
        self.get_mean_masses

    def get_halo_bin_indices(self):
        halos = self.halos
        edges  = self.binning
        index = self.index
        #Pick out the observable we are splitting on
        obs   = halos[:,index]
        indices = np.digitize(obs, edges, False)
        self.indicies = indices
        return indices

    @property
    def get_number_per_bin(self):
        edges  = self.binning
        indices = self.get_halo_bin_indices()
        N = np.bincount(indices, minlength = len(edges)-1)
        self.number_per_bin = N
        return N

    @property
    def get_mean_observable(self):
        halos = self.halos
        edges  = self.binning
        index = self.index
        obs   = halos[:,index] #Obseravble
        indices = np.digitize(obs, edges, False)
        Om = np.array([np.mean(obs[(indices==i)]) for i in range(len(edges)-1)])
        self.mean_observable = Om
        return Om

    @property
    def get_mean_masses(self):
        halos = self.halos
        edges  = self.binning
        index = self.index
        M     = halos[:,3] #Mass
        obs   = halos[:,index] #Obseravble
        indices = np.digitize(obs, edges, False)
        Mm = np.array([np.mean(M[(indices==i)]) for i in range(len(edges)-1)])
        self.mean_masses = Mm
        return Mm

    def get_dm_particles(self, dm_array=None, dm_path=None):
        if (dm_array is not None) and (dm_path is not None):
            raise Exception("Specify either a dm_array or the dm_path.")
        if dm_array:
            self.dm_array = dm_array
            return dm_array
        else:
            dmdf = pd.read_csv(inpath, dtype='float64', delim_whitespace=True)
            dm = dmdf.as_matrix()
            dm_array = np.copy(dm, order='C')
            self.dm_array = dm_array
            return dm_array

    def calculate_hmcfs(self, dm_array=None, dm_path=None):
        """Calculate the halo-matter correlation function.
        
        Args:
            dm_array (array like): DM particle positions
            dm_path (string): Path to a file with DM particle positions.

        """
        dm_pos = self.get_dm_particles(dm_array, dm_path)
        halos = self.halos
        halo_pos = halos[:, :3]
        halo_pos = np.copy(halo_pos, order='C')
        import auto_correlation, cross_correlation
        return
        
if __name__ == "__main__":
    print("Testing")
    data = np.load("testdata/reduced_halos_lamobs_009.npy")
    bins = np.array([20,30,45,60,999])
    cat = halo_catalog(data, bins)
    print(cat.number_per_bin)
    print(cat.mean_masses)
    print(cat.mean_observable)
    cat.calculate_hmcfs([0])
