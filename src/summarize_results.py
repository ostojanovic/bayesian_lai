from pandas.core.indexes.range import RangeIndex
from utils import *
import pandas
import pickle as pkl

## Summarize  model selection results
print("Loading model-selection results")

# Collect results for all models
series = []
colnames = ["ELPD", "P_loo", "(-Inf, 0.5]", "(0.5, 0.7]", "(0.7, 1]", "(1, Inf)"]
for (label,name) in [("naive_pooled","Naive"), ("hierarchical_full","Hier. Full"), ("hierarchical_only_bias","Hier. Bias")]:
    with open(os.path.join(os.path.dirname(__file__), "..", "data", label+".pkl"), "rb") as file:
        model = pkl.load(file)

    psis_loo = model["glm_model"]["psis-loo"]
    bins = np.asarray([-np.Inf, 0.5, 0.7, 1, np.Inf])
    counts, _ = np.histogram(psis_loo.pareto_k.values, bins)

    series.append(pandas.Series(name=name, data={
        "ELPD": "{:0.1f}±{:0.1f}".format(psis_loo["loo"],psis_loo["loo_se"]),
        "P_loo": psis_loo["p_loo"],
        "(-Inf, 0.5]": counts[0], 
        "(0.5, 0.7]": counts[1], 
        "(0.7, 1]": counts[2], 
        "(1, Inf)": counts[3]
    }))

# combine results into data-frame
model_selection = pandas.DataFrame(data=series)
model_selection.columns = pandas.MultiIndex.from_tuples([("","ELPD"),("","P_loo"),("Pareto k","(-Inf, 0.5]"),("Pareto k","(0.5, 0.7]"),("Pareto k","(0.7, 1]"),("Pareto k","(1, Inf)")])
model_selection.index.set_names("model", inplace=True)


# display and store results to disk
print("Model selection results: ")
print(model_selection)
model_selection.to_latex(os.path.join(os.path.dirname(__file__), "..", "tables", "model_selection.tex"), float_format="{:0.1f}".format)




## Summarize feature importance results
print("Loading feature-importance results")

# Collect results for all models
series = []
for (label,name) in [("naive_pooled","Pooled"), ("hierarchical_full","Hier. Full"), ("hierarchical_only_bias","Hier. Bias")]:
    with open(os.path.join(os.path.dirname(__file__), "..", "data", label+"_feature_importance.pkl"), "rb") as file:
        model_feature_importance = pkl.load(file)
    series.append(pandas.Series(name=name, index=pandas.RangeIndex(1,12), data=model_feature_importance))

# combine results into data-frame
feature_importance = pandas.DataFrame(data=series)
feature_importance.index.set_names("model", inplace=True)

# display and store results to disk
print("Feature-importance results: ")
print(feature_importance)
feature_importance.to_latex(os.path.join(os.path.dirname(__file__), "..", "tables", "feature_importance.tex"), float_format="{:0.1f}".format)

##