import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kochen_ma_2019_apoptosis_model import model

chain0_100000 = np.load('ECRP_ICRP_calibration_2019116_0_100000_parameters.npy')
chain0_120000 = np.load('ECRP_ICRP_calibration_2019116_0_120000_parameters.npy')
chain0_140000 = np.load('ECRP_ICRP_calibration_2019116_0_140000_parameters.npy')
chain0_160000 = np.load('ECRP_ICRP_calibration_2019116_0_160000_parameters.npy')
chain0 = np.concatenate((chain0_100000, chain0_120000, chain0_140000, chain0_160000))

chain1_100000 = np.load('ECRP_ICRP_calibration_2019116_1_100000_parameters.npy')
chain1_120000 = np.load('ECRP_ICRP_calibration_2019116_1_120000_parameters.npy')
chain1_140000 = np.load('ECRP_ICRP_calibration_2019116_1_140000_parameters.npy')
chain1_160000 = np.load('ECRP_ICRP_calibration_2019116_1_160000_parameters.npy')
chain1 = np.concatenate((chain1_100000, chain1_120000, chain1_140000, chain1_160000))

chain2_100000 = np.load('ECRP_ICRP_calibration_2019116_2_100000_parameters.npy')
chain2_120000 = np.load('ECRP_ICRP_calibration_2019116_2_120000_parameters.npy')
chain2_140000 = np.load('ECRP_ICRP_calibration_2019116_2_140000_parameters.npy')
chain2_160000 = np.load('ECRP_ICRP_calibration_2019116_2_160000_parameters.npy')
chain2 = np.concatenate((chain2_100000, chain2_120000, chain2_140000, chain2_160000))

chain3_100000 = np.load('ECRP_ICRP_calibration_2019116_3_100000_parameters.npy')
chain3_120000 = np.load('ECRP_ICRP_calibration_2019116_3_120000_parameters.npy')
chain3_140000 = np.load('ECRP_ICRP_calibration_2019116_3_140000_parameters.npy')
chain3_160000 = np.load('ECRP_ICRP_calibration_2019116_3_160000_parameters.npy')
chain3 = np.concatenate((chain3_100000, chain3_120000, chain3_140000, chain3_160000))

samples = np.concatenate((chain0, chain1, chain2, chain3))

idx_pars_calibrate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                      34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                      51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

ndims = len(idx_pars_calibrate)
colors = sns.color_palette(n_colors=ndims)
rows = 7
columns = 10
counter = 0

f, axes = plt.subplots(rows, columns, figsize=(10, 10))
for r in range(rows):
    for c in range(columns):

        weights = np.ones_like(samples[:, counter]) / float(len(samples[:, counter]))
        axes[r, c].hist(samples[:, counter], bins=25, color=colors[counter], weights=weights)
        # axes[r, c].set_title(idx_pars_calibrate[counter], fontdict={'fontsize': 8})
        # axes[r, c].set_xlim(-6, 6)
        # We change the fontsize of minor ticks label
        axes[r, c].tick_params(axis='both', which='major', labelsize=4)
        axes[r, c].tick_params(axis='both', which='minor', labelsize=4)
        counter += 1

        if counter >= len(idx_pars_calibrate):
            break
f.add_subplot(111, frameon=False)
f.subplots_adjust(wspace=0.9)
f.subplots_adjust(hspace=0.9)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.xlabel("Log(Parameter value)", fontsize=14)
plt.ylabel("Probability", fontsize=14, labelpad=15)

# plt.show()
plt.savefig('pars_dist_plot.pdf', format='pdf', bbox_inches="tight")
