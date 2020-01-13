import os
import numpy as np
import pandas as pd
from pydream.core import run_dream
from pysb.simulator import ScipyOdeSimulator
from pydream.parameters import SampledParam
from scipy.stats import norm
from scipy import interpolate
from pydream.convergence import Gelman_Rubin
from models.earm_kochen import model

# load experimental data
two_up = os.path.abspath(os.path.join(__file__, "../.."))
data_path = os.path.join(two_up, 'calibration_data/EC-RP_IMS-RP_IC-RP_data_for_models.csv')

exp_data = pd.read_csv(data_path, index_col=False)

# Build time points for the integrator, using the same time scale as the
# experimental data but with greater resolution to help the integrator converge.
ntimes = len(exp_data['# Time'])
# Factor by which to increase time resolution
tmul = 10
# Do the sampling such that the original experimental timepoints can be
# extracted with a slice expression instead of requiring interpolation.
tspan = np.linspace(exp_data['# Time'].values[0], exp_data['# Time'].values[-1],
                    (ntimes-1) * tmul + 1)

rate_params = model.parameters_rules()
param_values = np.array([p.value for p in model.parameters])
rate_mask = np.array([p in rate_params for p in model.parameters])

solver = ScipyOdeSimulator(model, tspan)

# Mean and variance of Td (delay time) and Ts (switching time) of MOMP, and
# yfinal (the last value of the IMS-RP trajectory)
momp_data = np.array([9810.0, 180.0, model.parameters['SmacM_0'].value])
momp_var = np.array([7245000.0, 3600.0, 1e4])

like_mbid = norm(loc=exp_data['norm_IC-RP'], scale=exp_data['nrm_var_IC-RP'])
like_cparp = norm(loc=exp_data['norm_EC-RP'], scale=exp_data['nrm_var_EC-RP'])
like_momp = norm(loc=momp_data, scale=momp_var)

sampled_parameter_names = [SampledParam(norm, loc=np.log10(pa), scale=1)
                           for pa in param_values[rate_mask]]


def likelihood(position):
    Y = np.copy(position)
    param_values[rate_mask] = 10 ** Y

    sim = solver.run(param_values=param_values)

    logp_mbid = np.sum(like_mbid.logpdf(sim.observables['BidM_obs'][::tmul] / model.parameters['BidU_0'].value))
    logp_cparp = np.sum(like_cparp.logpdf(sim.observables['ParpC_obs'][::tmul] / model.parameters['ParpU_0'].value))

    momp_traj = sim.observables['SmacC_obs']

    # Here we fit a spline to find where we get 50% release of MOMP reporter
    if np.nanmax(momp_traj) == 0:
        print('No aSmac!')
        t10 = 0
        t90 = 0
    else:
        ysim_momp_norm = momp_traj / np.nanmax(momp_traj)
        st, sc, sk = interpolate.splrep(tspan, ysim_momp_norm)
        try:
            t10 = interpolate.sproot((st, sc - 0.10, sk))[0]
            t90 = interpolate.sproot((st, sc - 0.90, sk))[0]
        except IndexError:
            t10 = 0
            t90 = 0

    # time of death  = halfway point between 10 and 90%
    td = (t10 + t90) / 2

    # time of switch is time between 90 and 10 %
    ts = t90 - t10

    # final fraction of aSMAC (last value)
    yfinal = momp_traj[-1]
    momp_sim = [td, ts, yfinal]

    logp_csmac = np.sum(like_momp.logpdf(momp_sim))

    # If model simulation failed due to integrator errors, return a log probability of -inf.
    logp_total = logp_mbid + logp_csmac + logp_cparp
    if np.isnan(logp_total):
        logp_total = -np.inf

    return logp_total


nchains = 5
niterations = 50000

if __name__ == '__main__':

    # Run DREAM sampling.  Documentation of DREAM options is in Dream.py.
    converged = False
    total_iterations = niterations
    sampled_params, log_ps = run_dream(parameters=sampled_parameter_names, likelihood=likelihood, nCR=6,
                                       niterations=niterations, nchains=nchains, multitry=False,
                                       gamma_levels=4, adapt_gamma=True, history_thin=1,
                                       model_name='earm_kochen_dreamzs_5chain2', verbose=True)

    # Save sampling output (sampled parameter values and their corresponding logps).
    for chain in range(len(sampled_params)):
        np.save('earm_kochen_dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
        np.save('earm_kochen_dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])

    #Check convergence and continue sampling if not converged

    GR = Gelman_Rubin(sampled_params)
    print('At iteration: ',total_iterations,' GR = ',GR)
    np.savetxt('earm_kochen_dreamzs_5chain_GelmanRubin_iteration_'+str(total_iterations)+'.txt', GR)

    old_samples = sampled_params
    if np.any(GR>1.2):
        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while not converged:
            total_iterations += niterations
            sampled_params, log_ps = run_dream(parameters=sampled_parameter_names, likelihood=likelihood,
                                               niterations=niterations, nchains=nchains, start=starts, multitry=False, gamma_levels=4,
                                               adapt_gamma=True, history_thin=1, model_name='earm_kochen_dreamzs_5chain2',
                                               nverbose=100, verbose=True, restart=True)


            # Save sampling output (sampled parameter values and their corresponding logps).
            for chain in range(len(sampled_params)):
                np.save('earm_kochen_dreamzs_5chain_sampled_params_chain_' + str(chain)+'_'+str(total_iterations), sampled_params[chain])
                np.save('earm_kochen_dreamzs_5chain_logps_chain_' + str(chain)+'_'+str(total_iterations), log_ps[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            GR = Gelman_Rubin(old_samples)
            print('At iteration: ',total_iterations,' GR = ',GR)
            np.savetxt('earm_kochen_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations)+'.txt', GR)

            if np.all(GR<1.2):
                converged = True

    try:
        #Plot output
        import seaborn as sns
        from matplotlib import pyplot as plt
        total_iterations = len(old_samples[0])
        burnin = total_iterations/2
        samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :], old_samples[2][burnin:, :],
                                  old_samples[3][burnin:, :], old_samples[4][burnin:, :]))

        ndims = len(sampled_parameter_names)
        colors = sns.color_palette(n_colors=ndims)
        for dim in range(ndims):
            fig = plt.figure()
            sns.distplot(samples[:, dim], color=colors[dim], norm_hist=True)
            fig.savefig('PyDREAM_earm_kochen_dimension_'+str(dim))

    except ImportError:
        pass

else:

    run_kwargs = {'parameters':sampled_parameter_names, 'likelihood':likelihood, 'niterations':niterations, 'nchains':nchains, \
                  'multitry':False, 'gamma_levels':4, 'adapt_gamma':True, 'history_thin':1, 'model_name':'earm_kochen_dreamzs_5chain2', 'verbose':True, 'nverbose':100}
