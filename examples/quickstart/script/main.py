import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.use('Agg')

bond = np.load('./data/K_Ca-CB_bondlen.npy')

fig = plt.figure()
plt.clf()
plt.hist(bond, 30, density = True)
plt.savefig('./output/bond_hist.png')
