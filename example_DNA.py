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
    exp_dir="example_DNA/"
)
# pay one-time cost to compute the Fourier transform
explainer.compute_fourier_transform(
    budget=30000, verbose=True
)

# Explain sequences using SHAP values
seqs = shapzero.load_dna_sequences_to_explain() # list of strings
shap = explainer.explain(seqs, explanation='shap_value') # list of SHAP values
# plot and save SHAP values
explainer.plot()
explainer.save()