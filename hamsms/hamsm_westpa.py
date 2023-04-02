#import sys
#sys.path.append("/home/chemical/phd/chz198152/apps/msm_we")
from msm_we import msm_we
import mdtraj as md
import tqdm
import numpy as np
import pyemma
import matplotlib.pyplot as plt
import pickle
import ray

file_paths = ["./west.h5"]
ref_structure = "./random_centered_seg.pdb"
def processCoordinates(self, coords):
    if self.dimReduceMethod == "none":
        nC = np.shape(coords)
        nC = nC[0]
        data = coords.reshape(nC, 3*self.nAtoms)
        return data
    elif self.dimReduceMethod == "pca":
        xt = md.Trajectory(xyz=coords, topology=None)
        indCA = self.reference_structure.topology.select('index 0 to 38')
        pair1, pair2 = np.meshgrid(indCA, indCA, indexing="xy")
        indUT = np.where(np.triu(pair1, k=1) > 0)
        pairs = np.transpose(np.array([pair1[indUT], pair2[indUT]])).astype(int)
        dist = md.compute_distances(xt, pairs, periodic=True, opt=True)        
        return dist

msm_we.modelWE.processCoordinates = processCoordinates
ray.init(num_cpus=4, ignore_reinit_error=True)

model = msm_we.modelWE()
model.initialize(
    file_paths,
    ref_structure,
    modelName='bispidine',
    basis_pcoord_bounds = [[-130, -50],[50, 130]],
    target_pcoord_bounds = [[-130, -50],[-130, -50]],
    dim_reduce_method = 'pca',
    tau = 1,
    pcoord_ndim=2,
)

model.get_iterations()

#model.get_coordSet(last_iter=model.maxIter, streaming=True)
model.get_coordSet(last_iter=70, streaming=True)

model.dimReduce()

model.cluster_coordinates(n_clusters=10, first_cluster_iter=50, streaming=True, stratified=True, use_ray=True)
with open('model.pkl', 'wb') as file:
        pickle.dump(model,file)
print("Clustering is complete")

model.get_fluxMatrix(0, first_iter=50, last_iter=model.maxIter, use_ray=True)
with open('model.pkl', 'wb') as file:
        pickle.dump(model,file)
print("Obtained flux matrices")

model.organize_fluxMatrix(use_ray=True)
with open('model.pkl', 'wb') as file:
        pickle.dump(model,file)
print("Organized flux matrices")

model.get_Tmatrix()

with open('model.pkl', 'wb') as file:
        pickle.dump(model,file)

model.get_steady_state()
model.get_steady_state_target_flux()

print(f"Steady-state target flux is {model.JtargetSS:.2e}")

model.plot_flux(suppress_validation=True)

model.get_committor()
model.plot_committor()

model.plot_flux_committor(suppress_validation=True)
plt.gca().set_xscale('linear')

with open('model.pkl', 'wb') as file:
	pickle.dump(model,file)
