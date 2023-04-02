# # Traditional MSM instead of vampnets
# ## The input trajectories are generated from a 10 $\mu$s  run on westpa

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import deeptime as dt
import pyemma
import pickle
from tqdm.notebook import tqdm

data = np.load('../data.npy')
di1 = 10
di2 = 18
di_slow_data = np.stack([np.concatenate(data)[:,di1],np.concatenate(data)[:,di2]], axis =1)
data_list = [data[i] for i in range(data.shape[0])]
concat_data = np.concatenate(data)

'''
pyemma.plots.plot_free_energy(*di_slow_data.T)
plt.xlabel(r'Dihedral$_1$')
plt.ylabel(r'Dihedral$_2$')
plt.title('Free energy from all WE runs')

from deeptime.util.validation import implied_timescales, ck_test
from deeptime.plots import plot_implied_timescales, plot_ck_test
from deeptime.decomposition import VAMP, TICA

lagtimes = np.arange(2,6,1)

vamp_models = [VAMP(lagtime=lag).fit_fetch(data_list) for lag in tqdm(lagtimes)]

ax = plot_implied_timescales(implied_timescales(vamp_models))
ax.set_yscale('log')
ax.set_xlabel('lagtime')
ax.set_ylabel('timescale')

import pickle
with open('vamp.pkl', 'wb') as file:
    pickle.dump(vamp_models, file)

'''

import pickle
infile = open('vamp.pkl','rb')
vamp_models =  pickle.load(infile)
infile.close()

#dt.plots.plot_ck_test(vamp_models[0].ck_test(vamp_models));
#plt.savefig('ck_test.jpg')

from deeptime.clustering import KMeans

vamp = vamp_models[0]

projection = vamp.transform(data_list)

kmeans_estimator = KMeans(n_clusters=200, progress=tqdm)
clustering = kmeans_estimator.fit_fetch(np.concatenate(projection))
dtrajs = clustering.transform(np.concatenate(projection))
np.save('clusters.npy',dtrajs)
