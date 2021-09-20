# Bayesian hierarchical predictions of leaf area index

This is a code repository for the paper *"Bayesian hierarchical models can infer interpretable predictions of leaf area index from heterogeneous datasets"*.
The developed model is a Bayesian hierarchical model that predicts the leaf area index (of white winter wheat) from measurements of reflectance spectra, collected at different locations and growth stages.

The model is implemented in Python and relies on the [PyMC3][1] and [Theano][2] packages for computationally efficient sampling.

Key features of the model:
* it uses Bayesian hierarchical model and Monte Carlo sampling techniques in order to incorporate prior domain knowledge and capture the association between leaf area index and the spectral reflectance at various wavelengths by spline-based kernel functions
* we compare models with three different levels of hierarchy
* it is extensible and requires only readily available data, it can be easily adapted to measurements from different locations.

## Usage
The results shown in our publication can be produced by running the following scripts in this order:
1. the `preprocessing.py` script parses and re-structures the raw data for further use.
2. the `modeling.py` script fits the different models to the available data.
3. the `feature_importance.py` script estimates the feature importance of each parameter.
4. the `summarize_results.py` script generates the tables in the [tables subfolder](/tables) summarizing the various models' results.
5. the `plot_<name>.py` scripts each generate the corresponding figure from the [figures subfolder](/figures).

(The files `spline_utils.py` and `utils.py` contain utility functions for spline fitting, theming and plotting.)

The code is licensed under an MIT license.

## Data sources ![CC BY-NC-ND license](cc-by-nc-nd.png)
The dataset was collected by the [Working Group Remote Sensing and Digital Image Analysis, from the Institute of Computer Science at the University of Osnabrück, Germany][3].
It consist of measurements of reflectance spectra and accompanying leaf area index, collected on four different locations in Germany at different growth stages.
The data is provided here under a creative commons [CC BY-NC-ND license][4] and original copyright resides with the [owners][3].
If you are interested in the dataset or wish to use it under conditions not permitted by this license, please contact the [Working Group Remote Sensing and Digital Image Analysis][3]. 

[1]: https://docs.pymc.io/
[2]: https://github.com/Theano/Theano
[3]: https://www.informatik.uni-osnabrueck.de/arbeitsgruppen/fernerkundung_und_digitale_bildverarbeitung.html
[4]: https://creativecommons.org/licenses/by-nc-nd/4.0/
[5]: https://opensource.org/licenses/MIT