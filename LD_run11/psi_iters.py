#finds the psi dihedrals
import numpy as np
from westpa.analysis import Run
import pyemma
import pickle

west = Run.open('west.h5')
topology = 'common_files/lld_ini.pdb'
featurizer = pyemma.coordinates.featurizer(topology)
dihedral_indices = np.genfromtxt('../dihedral_indices.txt', skip_header=2).astype(int)
psi = 6
featurizer.add_dihedrals(dihedral_indices)

psi_iters = []
for iters in west:
    traj_list = ['traj_segs/'+str(iters.number).zfill(6) +'/' + str(w).zfill(6)+'/seg.xtc' for w in range(iters.num_walkers)]
    coord_data = pyemma.coordinates.load(traj_list,features=featurizer)
    coord_data = np.array([i[1,:] for i in coord_data])
    coord_data = coord_data[:,psi].reshape(-1,1)
    psi_iters.append(coord_data)
with open('psi_iters.pkl','wb') as f:
    pickle.dump(psi_iters,f)
