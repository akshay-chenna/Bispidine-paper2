# This script finds those states once entering C1 never visit anywhere.
import numpy as np
from westpa.analysis import Run
import h5py
from joblib import Parallel, delayed

west = Run.open('west.h5')
f = h5py.File('ANALYSIS/C1_T1_100itfine/assign.h5', 'r')
statelabels = np.array(f['statelabels'])[:, :, 1]  # Current states

iters = np.where(statelabels == 0)[0] + 1
walkers = np.where(statelabels == 0)[1]

# iter_num from 1, walker_num from 0
def if_nohist(iter_num, walker_num, start_iter):
    if iter_num >= start_iter:
        print(iter_num)
        trace = west.iteration(iter_num).walker(walker_num).trace()
        trace_walker_num = [seg.index for seg in trace]
        trace_states = statelabels[np.arange(iter_num), trace_walker_num]
        # if it never leaves C1 once entering from int
        hist_current_state = np.where(trace_states == 0)[0]
        continuity = np.all(np.diff(hist_current_state) == 1)
        if continuity and (1 not in trace_states):
            return 'traj_segs/'+str(iter_num).zfill(6) + '/' + str(walker_num).zfill(6)+'/seg.xtc'

vals = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(if_nohist)(i, j, 101) for i, j in zip(iters, walkers))
vals = np.array(list(filter(None, vals)))
np.save('C1_never_transit.npy', vals)
