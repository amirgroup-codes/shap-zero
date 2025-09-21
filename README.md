# SHAP zero: Explaining Biological Sequence Models <img src="assets/SHAPzero.svg" alt="SHAP zero logo" align="right" height="200px"/>

[![PyPI version](https://badge.fury.io/py/shapzero.svg)](https://badge.fury.io/py/shapzero)
[![PyPI - License](https://img.shields.io/pypi/l/shapzero.svg)](https://opensource.org/licenses/MIT)
[![PyPI Status](https://img.shields.io/pypi/status/shapzero.svg?color=blue)](https://pypi.org/project/shapzero)
[![PyPI Version](https://img.shields.io/pypi/pyversions/shapzero.svg)](https://pypi.org/project/shapzero)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Last Commit](https://img.shields.io/github/last-commit/amirgroup-codes/shap-zero)](https://github.com/amirgroup-codes/shap-zero/commits/main)

SHAP zero is a Python package that enables the amortized computation of Shapley values and interactions. It does this by paying a one-time cost to sketch the model's Fourier transform. After this one-time cost, SHAP zero enables **near-zero marginal cost** for future query sequences by mapping the Fourier transform to Shapley values and interactions.

## Installation 

`shapzero` is designed to work with Python 3.10 and above. Installation can be done via `pip`:
```sh
pip install shapzero
```

## Quickstart

Initialize your model using `shapzero.init` and compute the Fourier transform using `compute_fourier_transform`. From there, you can explain SHAP values and interactions using `explain`.

```python
import shapzero

# Train example model
X, y = shapzero.load_dna_example()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, interaction_only=True)),  
    ('linear_regression', Ridge(alpha=0.5)) 
])
model.fit(X, y)

# Set up SHAP zero explainer
q = 4   # alphabet size (q=4 nucleotides for DNA and RNA, q=20 amino acids for proteins)
n = 10  # sequence length
explainer = shapzero.init(
    q=q,
    n=n,
    model=model,
    exp_dir=output_directory
)
# pay one-time cost to compute the Fourier transform
explainer.compute_fourier_transform(
    budget=30000, verbose=True
)
>> ----------
>> R^2 is 0.96
>> There are 20 1-order interactions.
>> There are 208 2-order interactions.
>> There are 1 0-order interactions.
>> ----------

# Explain sequences using SHAP values
seqs = shapzero.load_dna_sequences_to_explain() # list of strings
print(seqs)
>> ['ACTCTTGAGG', 'TATATCTGTG', 'GATGTATAGG'...
shap = explainer.explain(seqs, explanation='shap_value') # list of SHAP values
print(shap[0])
>> {(0,): 1.3241669688536364,   # SHAP value of the 1st nucleotide
>>  (1,): 0.4545280155565195,     
>>  (2,): -3.6661905864093125, 
>>  ...}
# plot and save SHAP values
explainer.plot()
explainer.save()
```
<p align="center">
  <img width="800px" src="assets/dna_shap_values.png" alt="Plot of SHAP values over DNA sequences">
</p>

```python
# Explain sequences using Shapley interactions
interaction = explainer.explain(sample, explanation='interaction')  # list of interactions
print(interactions[0])
>> {(0, 7): 2.867415008537887,   # interaction between the 1st and 8th nucleotides
>>  (6, 7): -1.2684576082389891,
>>  (4, 5): 0.4493051300654991: 
>>  ...}
# plot and save interactions
explainer.plot()
explainer.save()
```
<p align="center">
  <img width="800px" src="assets/dna_interactions.png" alt="Plot of Shapley interactions over DNA sequences">
</p>

## Load a previously computed Fourier transform

If you previously ran `explainer.compute_fourier_transform()`, SHAP zero will automatically save the Fourier transform to `output_directory/fourier_transform.pickle`. To resume explaining from that previous checkpoint, you can load in the Fourier transform path into `shapzero.init`.
```python
explainer = shapzero.init(
    q=q,
    n=n,
    fourier_transform=f"{output_directory}/fourier_transform.pickle"
    exp_dir=output_directory
)
# explain using the pre-computed Fourier transform! 
shap = explainer.explain(seqs, explanation='shap_value')
explainer.plot()
explainer.save()
interaction = explainer.explain(sample, explanation='interaction') 
explainer.plot()
explainer.save()
```

## What types of models is SHAP zero compatible with?

SHAP zero aims to be compatible with most biological sequence models out of the box! SHAP zero will automatically detect what type of model you have (e.g. PyTorch, sklearn, XGBoost, etc.) and attempt to query from said model. To streamline the process, we ask that your model takes in either *one-hot* (with input dimension $q \times n$) or $q$-ary inputs. 

By default, SHAP zero uses the following $q$-ary encoding scheme:
```python
DNA_ENCODING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
RNA_ENCODING = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
PROTEIN_ENCODING = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
    'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'V': 17, 'W': 18, 'Y': 19
}
```
For example, if our one-hot DNA model takes in as an input `[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]`, SHAP zero will attempt to query the sequence `'AGCT'`. If our $q$-ary protein model takes as an input `[18, 2, 5, 15]`, SHAP zero will attempt to query the sequence `'WDGS'`.

## What if my model uses a different input scheme/uses a unique architecture?

In an effort to be compatible with every possible biological sequence model, SHAP zero is also fully capable of taking in user-written *functions*. We request that the input of the function is capable of taking in a 2D $q$-ary numpy array of shape `(num_samples, n)` and outputs a 1D numpy of shape `(num_samples,)`. Alternatively, your function can also take in as an input a list of sequences, where each sequence is a string of length `n`, and the list is of length `num_samples`. 

Examples of possible functions:
```python
def model(X):
    """
    Computes y = 5 * X[:, 0] - 2 * X[:, 3] + X[:, 1] * X[:, 4]
    """
    return y = 5 * X[:, 0] - 2 * X[:, 3] + X[:, 1] * X[:, 4]
explainer = shapzero.init(
    q=q,
    n=n,
    exp_dir=output_dir,
    model=model
)
```

```python
# Assume 'load_model' and 'compute_model_scores' are functions from an external library
def load_model(model_path):
    ...
def compute_model_scores(model, samples, context_data):
    # This is a placeholder for the function that gets predictions.
    # It might take the model, the new sequences (samples), and other contextual data.
    ...

# Define a wrapper that will interface with SHAP zero
class ModelScorer:
    def __init__(self, model_path, context_data=None):
        """
        Initializes the scorer by loading the model and storing any
        contextual data needed for predictions.

        Args:
            model_path (str): Path to the pre-trained model.
            context_data (dict, optional): A dictionary of any other data the
                model needs outside of just the length-n sequence.
        """
        self.model = load_model(model_path)
        self.context_data = context_data if context_data is not None else {}

    def predict(self, samples_numpy_array):
        """
        This is the method that will be passed to SHAP zero.
        It takes a 2D q-ary numpy array and returns a 1D numpy array of scores.
        """
        # This function calls your underlying model's prediction logic,
        # passing along the model object, the new samples, and any other
        # contextual data that was stored during initialization.
        scores = compute_model_scores(
            model=self.model,
            samples=samples_numpy_array,
            context_data=self.context_data
        )
        return np.array(scores)

model_path = "model"
context_data = {
    "target_sequence": "ACGTACGT",
    "positions_of_interest": [2, 3, 6]
}
scorer = ModelScorer(model_path=model_path, context_data=context_data)
# pass scorer.predict into SHAP zero! 
explainer = shapzero.init(
    q=q,
    n=n,
    exp_dir=output_dir,
    model=scorer.predict  # pass the wrapper here
)
```

## Citation

If you use `shapzero` and enjoy it, please consider citing our [paper](https://arxiv.org/pdf/2410.19236)! SHAP zero was recently accepted into **NeurIPS 2025**, and we look forward to the great discussions!

```bibtex
@inproceedings{tsui2025shapzero,
  title={{SHAP} Zero Explains Biological Sequence Models with Near-zero Marginal Cost for Future Queries},
  author={Tsui, Darin and Musharaf, Aryan and Erginbas, Yigit E. and Kang, Justin S. and Aghazadeh, Amirali},
  booktitle={Advances in Neural Information Processing Systems (Accepted)},
  year={2025}
}
```