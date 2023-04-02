# # Flux calculation, see .ipynb in run5(mac)

# * This script calculates the frame-wise flux entering at a given molecular time instead of the conventional 'after-iteration' calculation.
# * Flux per molecular time (in terms of the frames) is the total flux entering from all walkers

# ## We will use westpa api for creating this custom analysis

# 
# See https://github.com/westpa/westpa/wiki/man%3Awestpa.analysis \
# See https://github.com/westpa/westpa2_tutorials/blob/main/tutorial-3.2/analysis.ipynb \
# See https://github.com/westpa/westpa2_tutorials/blob/main/tutorial-3.1/NaCl_Tut.ipynb \
# See https://github.com/westpa/westpa2_tutorials/blob/main/paper/AdvancedTutorialManuscript.pdf Section 3.4.2 \
# See https://westpa.readthedocs.io/en/latest/documentation/analysis/index.html#how-to

import h5py
from westpa.analysis import Run
import matplotlib.pyplot as plt
import numpy as np
from numpy import inf


# ### Open westpa run file

run = Run.open('../west.h5')


# ### Define and assign bins and bin indices

bins = [-inf, -130, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 130, inf ] # bin definitions in the x and y direction
di_bin = [((np.digitize(run.iteration(iteration.number).pcoords[:,:,0],bins,right=True)-1)*(len(bins)-1)+np.digitize(run.iteration(iteration.number).pcoords[:,:,1],bins,right=True)-1) for iteration in run ]


# ### Define basis and target state bins and indices

state1 = [-90, 90]
state2 = [90, 90]
state_A = (np.digitize(state1[0],bins,right=True)-1)*(len(bins)-1)+np.digitize(state1[1],bins,right=True)-1
state_B = (np.digitize(state2[0],bins,right=True)-1)*(len(bins)-1)+np.digitize(state2[1],bins,right=True)-1
print('The cis state bin index is', state_A )
print('The trans state bin index is', state_B )


# ### Calculcate the $\alpha$ and $\beta$ ensemble populations
# 






# ### Calculate fluxes for each frame and iteration


flux_A = []
flux_B = []
for iter in range(100,run.num_iterations+1): # Starts from 1
    iteration = run.iteration(iter)
    for ind in range(0,run.iteration(1).walker(0).num_snapshots):
        total_flux_to_A = 0
        total_flux_to_B = 0
        for walker in iteration: # Starts from 0
            flux_to_A = 0
            flux_to_B = 0
            trace = walker.trace()
            xs = [ walker.iteration.number for walker in trace ] # Starts from 1
            ys = [ walker.index for walker in trace] # Starts form 0
            history_walker = np.concatenate([di_bin[i-1][j] for i,j in zip(xs[:-1],ys[:-1])]) # -1 for iter not walker
            current_walker = di_bin[xs[-1]-1][ys[-1]] # -1 for iter not walker
            history_walker = np.hstack((history_walker,current_walker[:ind]))
            current_bin = current_walker[ind]
            
            try:
                recent_A = np.where(history_walker==state_A)[0].max()
            except ValueError:
                recent_A = -1
            try:
                recent_B = np.where(history_walker==state_B)[0].max()
            except ValueError:
                recent_B = -1
            
            if (ind != 0) and (current_bin == state_A ) and (history_walker[-1] != state_A) and (recent_B>recent_A):
                flux_to_A = run.iteration(xs[-1]).walker(ys[-1]).weight
            elif (ind !=0) and (current_bin == state_B ) and (history_walker[-1] != state_B) and (recent_A>recent_B):
                flux_to_B = run.iteration(xs[-1]).walker(ys[-1]).weight    
            
            total_flux_to_A += flux_to_A
            total_flux_to_B += flux_to_B
        flux_A.append(total_flux_to_A)
        flux_B.append(total_flux_to_B)


# ## Save flux files

np.save('flux_A.npy', flux_A)
np.save('flux_B.npy', flux_B)
