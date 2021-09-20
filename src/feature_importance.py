
import numpy as np
import pickle as pkl
import pymc3 as pm
import theano.tensor as tt
from sklearn.metrics import r2_score
from utils import *
import xarray
import arviz 



def deterministic_naive(w, bias, model=None, **kwargs):
    model = pm.modelcontext(model)
    return tt.dot(model.features, w) + bias

def deterministic_hierarchical_full(w_shared, w, bias_shared, bias, model=None, **kwargs):
    model = pm.modelcontext(model)
    return (
        model.features.eval()[...,np.newaxis] * (w_shared.values + w[model.group_index.eval(),:].values)
    ).sum(axis=1) + bias_shared.values + bias[model.group_index.eval()].values

def deterministic_hierarchical_only_bias(w, bias_shared, bias, model=None, **kwargs):
    model = pm.modelcontext(model)
    return tt.dot(model.features.eval(), w.values) + bias_shared.values + bias[model.group_index.eval()].values


def compute_logp(deterministic, trace, model=None):
    """
    Computes the log-likelihood of the observed variable `Y` of the `model` (taken from context if not specified)
    for the parameter samples from the given `trace`.
    Uses the `deterministic` function to compute the deterministic linear part of the GLM (= µ of the Lognormal dist.)
    """
    # get current model
    model = pm.modelcontext(model)

    # get all samples of all parameters into the right shape
    args = dict()
    for (key,val) in trace.posterior.data_vars.items():
        args[key] = val.stack(sample=["chain", "draw"])

    
    # the SD argument sets the (over-)dispersion of the model
    sd = args.pop("sd")
    
    # compute the deterministic linear part of the GLM
    mu = deterministic(**args, model=model)

    # compute the log-likelihood of the observed data for the modified deterministic part
    dist = pm.distributions.Lognormal.dist(mu=mu, sigma=sd)
    logp = dist.logp(model.observed.reshape((-1,1)))

    # create dataset for output
    ds = xarray.Dataset(
        data_vars = {
            "Y": (
                ["chain", "draw", "Y_dim_0"], 
                logp.T.reshape((trace.sample_stats.dims["chain"],trace.sample_stats.dims["draw"],trace.observed_data.dims["Y_dim_0"])).eval()
                )
        },
        coords = {
            "chain": trace.sample_stats.coords["chain"],
            "draw": trace.sample_stats.coords["draw"],
            "Y_dim_0": trace.observed_data.coords["Y_dim_0"],
        }
    )
    return ds



# load data
full_data = load_data()
full_LAI = full_data["LAI"]

# we average the results of permuting each feature over `num_trials` runs
num_trials = 10

for label,deterministic in [("naive_pooled", deterministic_naive), ("hierarchical_full", deterministic_hierarchical_full), ("hierarchical_only_bias", deterministic_hierarchical_only_bias) ]:
    # load model
    print("Calculating feature importance for model '{}'".format(label))
    with open(os.path.join(os.path.dirname(__file__), "..", "data", label+".pkl"), "rb") as file:
        model_data = pkl.load(file)

    features = model_data["glm_model"]["feature_matrix_all"]
    model = model_data["glm_model"]["model_all"]
    trace = model_data["glm_model"]["trace"]
    

    # Compute importance for each feature
    feature_importance = []
    with model:
        # Compute unpermuted log-likelihood
        _logp_unpermuted = compute_logp(deterministic, trace)
        assert np.abs(_logp_unpermuted-trace.log_likelihood).max() < 1e-4, "Too large deviation from previously computed logp!"
        logp_unpermuted = float(_logp_unpermuted.mean(dim=["chain","draw"]).sum(dim="Y_dim_0")["Y"])

        for feature in range(features.shape[1]):
            print("\t Sampling from posterior predictive distribution on permuted data of feature {}".format(feature))
            
            fi_trial = np.zeros(num_trials)
            for trial in range(num_trials):
                # permute the values for the chosen feature
                feature_matrix_permuted = features.copy()
                np.random.shuffle(feature_matrix_permuted[:,feature])

                # update the feature matrix used in the model
                pm.set_data({"features": feature_matrix_permuted})

                # compute log-likelihood for the permuted feature    
                _logp_permuted = compute_logp(deterministic, trace)            
                logp_permuted = float(_logp_permuted.mean(dim=["chain","draw"]).sum(dim="Y_dim_0")["Y"])

                # compute the feature importance as the ratio of permuted/unpermuted
                fi_trial[trial] = logp_unpermuted - logp_permuted

                print("\t\t Trial {} - Feature importance: {}".format(trial, fi_trial[trial]))
            
            print("\t Got feature importance {} with std {}".format(fi_trial.mean(), fi_trial.std()))

            feature_importance.append(fi_trial.mean())

    print("Saving data")
    with open(os.path.join(os.path.dirname(__file__), "..", "data", label+"_feature_importance.pkl"), "wb") as file:
        pkl.dump(feature_importance, file)