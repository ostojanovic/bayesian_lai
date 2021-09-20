
import numpy as np
import pickle as pkl
import pymc3 as pm
from utils import *
import splines_utils
from datetime import time
import os

# Set meta parameters
wavelength_start = 400
wavelength_stop = 1350
smoothing_kernel_width = 10
num_knots = 10
spline_order = 3
num_features = (num_knots+spline_order-2)

weight_std = 1.0
bias_std = weight_std*num_features

deviation_scale = 0.1

def model_hierarchical_only_bias(LAI, feature_matrix, group_index):
    """
    Constructs and returns a hierarchical `model` for a given dataset of `LAI` labels and a `feature_matrix`. 
    In this "only-bias" hierarchical model, each group has its own bias (but not feature weights!), 
    which are represented by their deviation from a shared bias.
    `group_index` identifies for each data point which group it belongs to.
    """
    with pm.Model() as model:
        sd = pm.Lognormal("sd", mu=0, sigma=1.0)
        bias_shared = pm.Normal("bias_shared", mu=0, sigma=bias_std)
        w = pm.Normal('w', mu=0, sigma=weight_std, shape=feature_matrix.shape[1])
        
        bias_deviations = pm.Normal("bias", mu=0, sigma=bias_std*deviation_scale, shape=len(unique_idxs))
        
        features = pm.Data("features", feature_matrix)
        observed = pm.Data("observed", LAI.values)
        group_index = pm.Data("group_index", group_index.values)
        
        mu = tt.dot(model.features, w) + bias_shared + bias_deviations[model.group_index]

        Y = pm.Lognormal("Y", mu = mu, sigma = sd, observed = observed)
    return model

def model_hierarchical_full(LAI, feature_matrix, group_index):
    """
    Constructs and returns a hierarchical `model` for a given dataset of `LAI` labels and a `feature_matrix`. 
    In this full hierarchical model, each group has its own set of feature weights and biases, which are represented by
    their deviation from a shared set of feature weights and biases.
    `group_index` identifies for each data point which group it belongs to.
    """
    with pm.Model() as model:
        sd = pm.Lognormal("sd", mu=0, sigma=1.0)
        bias_shared = pm.Normal("bias_shared", mu=0, sigma=bias_std)
        w_shared = pm.Normal('w_shared', mu=0, sigma=weight_std, shape=feature_matrix.shape[1])
        
        bias_deviations = pm.Normal("bias", mu=0, sigma=bias_std*deviation_scale, shape=len(unique_idxs))
        w_deviations = pm.Normal('w', mu=0, sigma=weight_std*deviation_scale, shape=(len(unique_idxs),feature_matrix.shape[1]))
        
        features = pm.Data("features", feature_matrix)
        observed = pm.Data("observed", LAI.values)
        group_index = pm.Data("group_index", group_index.values)
        
        mu = (model.features * (w_shared + w_deviations[model.group_index,:])).sum(axis=1) + bias_shared + bias_deviations[model.group_index]

        Y = pm.Lognormal("Y", mu = mu, sigma = sd, observed = observed)
    return model

def model_naive(LAI, feature_matrix, group_index=None):  
    """
    Constructs and returns a naive (i.e. non-hierarchical) `model` for a given dataset of `LAI` labels and a 
    `feature_matrix`. For the naive model, `group_index` is only needed for compatibility with the hierarchical models!
    """
    with pm.Model() as model:
        sd = pm.Lognormal("sd", mu=0, sigma=1.0)
        bias = pm.Normal("bias", mu=0, sigma=bias_std)
        w = pm.Normal('w', mu=0, sigma=weight_std, shape=feature_matrix.shape[1])   # equivalent to l1 regularization
        
        features = pm.Data("features", feature_matrix)
        observed = pm.Data("observed", LAI.values)
        group_index = pm.Data("group_index", group_index.values)
        
        mu = tt.dot(model.features,w) + bias
        
        Y = pm.Lognormal("Y", mu = mu, sigma = sd, observed = observed)
        
    return model

models = [
    ("naive_field_A", "Naive model, Field A", "A", model_naive), # uses data form field 1, year A
    ("naive_field_B", "Naive model, Field B", "B", model_naive), # uses data form field 1, year B
    ("naive_field_C", "Naive model, Field C", "C", model_naive), # uses data form field 2, year A
    ("naive_field_D", "Naive model, Field D", "D", model_naive), # uses data form field 2, year B
    ("naive_pooled", "Naive model, pooled", slice(None), model_naive), # uses all data
    ("hierarchical_only_bias", "Hierarchical bias model", slice(None), model_hierarchical_only_bias), # uses hiearchical model with different bias terms, only
    ("hierarchical_full", "Full hierarchical model", slice(None), model_hierarchical_full), # uses hiearchical model
]


# load data
full_data = load_data()
full_LAI = full_data["LAI"]
full_spectrum = full_data.loc[:,wavelength_start:wavelength_stop]

# add numeric group labels
# compute numeric group-index of each row in the data
full_idx = full_data.index.droplevel([0,2])
# map unique group indices to numeric indices
unique_idxs = full_idx.unique()
mapping = pd.DataFrame(range(len(unique_idxs)), index=unique_idxs, columns=["num_idx"])
full_idx_numeric = mapping.loc[full_idx, "num_idx"]
full_idx_numeric.index = full_data.index


# run each model
for label, name, index, model_fun in models:
    print("=== Running for {}===".format(name))
    
    # extract relevant data slices
    LAI_all = full_LAI.loc[(slice(None),index,slice(None))]
    LAI_train = full_LAI.loc[(False,index,slice(None))]
    LAI_test = full_LAI.loc[(True,index,slice(None))]

    group_index_all = full_idx_numeric.loc[(slice(None),index, slice(None))]
    group_index_train = full_idx_numeric.loc[(False,index, slice(None))]
    group_index_test = full_idx_numeric.loc[(True,index, slice(None))]

    spectrum_all = full_spectrum.loc[(slice(None),index,slice(None))].values  
    spectrum_train = full_spectrum.loc[(False,index,slice(None))].values
    spectrum_test = full_spectrum.loc[(True,index,slice(None))].values


    # Spline model with an adaptive knot vector (heuristic placement of knots based on absolute value of the second derivative)
    cum_abs_curvature = splines_utils.calc_cum_abs_deriv(spectrum_all, sigma=smoothing_kernel_width, order=2).mean(axis=0)
    adaptive_rate_knots, percentiles = splines_utils.find_percentiles(y=cum_abs_curvature, T=spectrum_all.shape[1], num_percentiles=num_knots, return_thresholds=True)
    knots = splines_utils.augknt(adaptive_rate_knots, spline_order-1) # uses spline_order-1 to fix left and right basis function at zero
    spline_bases = splines_utils.spcol(range(spectrum_all.shape[1]),knots,spline_order)
    spline_bases = spline_bases / np.linalg.norm(spline_bases, axis=0)

    # calculates vector of fourier-coefficients (dot-product between each basis function and spectrogram) --> use this as features
    feature_matrix_train = np.dot(spectrum_train, spline_bases)    
    feature_matrix_test = np.dot(spectrum_test, spline_bases)    
    feature_matrix_all = np.dot(spectrum_all, spline_bases)    
    
    # subtract mean & scale by std
    feature_mean = feature_matrix_all.mean(axis=0)
    feature_std = feature_matrix_all.std(axis=0)

    feature_matrix_train = (feature_matrix_train - feature_mean)/feature_std
    feature_matrix_test = (feature_matrix_test - feature_mean)/feature_std
    feature_matrix_all = (feature_matrix_all - feature_mean)/feature_std

    # create model for ALL data
    model_all = model_fun(LAI_all, feature_matrix_all, group_index_all)
    # sample from the model
    with model_all:
        print("\t Sampling from prior predictive distribution")
        # "predict" parameters as well as a labels from priors alone
        trace_prior = pm.sample_prior_predictive()

        print("\t Sampling from conditional distribution for all data")
        # sample the model parameters given the observed LAI values of the data
        trace = pm.sample(2000, init="jitter+adapt_full", cores=4, return_inferencedata=True)

        print("\t Sampling from posterior predictive distribution")
        # predict the LAI values on the test data from the previously sampled trace of parameter values
        pred = pm.fast_sample_posterior_predictive(trace)

        print("\t Calculating PSIS-LOO")
        # leave-one-out prediction accuracy using pareto-smoothed importance sampling (PSIS-LOO)
        loo = pm.loo(trace, pointwise=True)
    
    # create model for TRAINING data only
    # re-create model for new shape, because as of current PyMC3 version 3.10 
    # changing the shape of observed data as follows does not work:
    # pm.set_data({"features": feature_matrix_train, "observed": LAI_train.values, "group_index": group_index_train.values})
    # (see issue: https://github.com/pymc-devs/pymc3/issues/4114)
    model_train = model_fun(LAI_train, feature_matrix_train, group_index_train)
    with model_train:
        print("\t Sampling from conditional distribution for training data only")
        # sample the model parameters given the observed LAI values of the training data
        trace_train = pm.sample(2000, init="jitter+adapt_full", cores=4, return_inferencedata=True)
        
        print("\t Sampling from posterior predictive distribution on training data")
        # predict the LAI values on the training data from the previously sampled trace of parameter values
        pred_train = pm.fast_sample_posterior_predictive(trace_train)

        # swap-in test set features
        pm.set_data({"features": feature_matrix_test, "group_index": group_index_test.values})

        print("\t Sampling from posterior predictive distribution on test data")
        # predict the LAI values on the test data from the previously sampled trace of parameter values
        pred_test = pm.fast_sample_posterior_predictive(trace_train)


    # collect relevant information
    model_dict = {
        "label": label,
        "name": name,
        "spline_model": {
            "wavelength_start": wavelength_start,
            "wavelength_stop": wavelength_stop,
            "smoothing_kernel_width": smoothing_kernel_width,
            "num_knots": num_knots,
            "spline_order": spline_order,
            "cum_abs_curvature": cum_abs_curvature,
            "adaptive_rate_knots": adaptive_rate_knots,
            "percentiles": percentiles,
            "knots": knots,
            "spline_bases": spline_bases
        },
        "glm_model": {
            "feature_matrix_all": feature_matrix_all,
            "feature_matrix_train": feature_matrix_train,
            "feature_matrix_test": feature_matrix_test,
            "model_all": model_all,
            "model_train": model_train,
            "LAI_all": LAI_all,
            "LAI_train": LAI_train,
            "LAI_test": LAI_test,
            "trace": trace,
            "trace_prior": trace_prior,
            "trace_train": trace_train,
            "psis-loo": loo, 
            "pred": pred["Y"],
            "pred_train": pred_train["Y"],
            "pred_test": pred_test["Y"]
        }
    }

    # save results
    print("saving data")
    with open(os.path.join(os.path.dirname(__file__), "..", "data", label+".pkl"), "wb") as file:
        pkl.dump(model_dict, file)
