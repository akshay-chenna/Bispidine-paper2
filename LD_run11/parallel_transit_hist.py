# This script finds those states which have just entered T1 and have visited C1 in the past.
import numpy as np
from westpa.analysis import Run
import h5py
from joblib import Parallel, delayed

west = Run.open('west.h5')
f = h5py.File('ANALYSIS/C1_T1_100itfine/assign.h5', 'r')
statelabels = np.array(f['statelabels'])[:, :, 1]  # Current states

iters = np.where(statelabels == 1)[0] + 1
walkers = np.where(statelabels == 1)[1]

# iter_num from 1, walker_num from 0
def if_just_hist(iter_num, walker_num, start_iter, hist_label):
    if iter_num >= start_iter:
        print(iter_num)
        trace = west.iteration(iter_num).walker(walker_num).trace()
        trace_walker_num = [seg.index for seg in trace]
        trace_states = statelabels[np.arange(iter_num), trace_walker_num]
        # if just entered and visited C1
        if (trace_states[-2] != trace_states[-1]) and (hist_label in trace_states[20:]):
            return 'traj_segs/'+str(iter_num).zfill(6) + '/' + str(walker_num).zfill(6)+'/seg.xtc'

vals = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(if_just_hist)(i, j, 101, 0) for i, j in zip(iters, walkers))
vals = np.array(list(filter(None, vals)))
np.save('T1_immediate_histC1.npy', vals)
