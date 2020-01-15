from pysb.simulator import ScipyOdeSimulator
from kochen_ma_2019_apoptosis_model import model
import numpy as np

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

sampled_log_pars = np.concatenate((chain0, chain1, chain2, chain3))
unlog_pars = 10**sampled_log_pars

initials_values = [j.value for i,j in enumerate(model.parameters) if j in model.parameters_initial_conditions()]
initials_values = np.array(initials_values)

n_samples = unlog_pars.shape[0]
initials_to_add = np.tile(initials_values, (n_samples, 1))

all_pars = np.append(unlog_pars, initials_to_add, 1)

tspan = np.linspace(0, 20000, 1111)

sim = ScipyOdeSimulator(model, tspan).run(param_values=all_pars, num_processors=100)
sim.save('sims_irvin_converged_kochen_original_ic.h5')
