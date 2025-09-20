import numpy as np
import re
from functools import wraps
DNA_ENCODING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
RNA_ENCODING = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
PROTEIN_ENCODING = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
    'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'V': 17, 'W': 18, 'Y': 19
}
DNA_DECODING = {v: k for k, v in DNA_ENCODING.items()}
RNA_DECODING = {v: k for k, v in RNA_ENCODING.items()}
PROTEIN_DECODING = {v: k for k, v in PROTEIN_ENCODING.items()}

# Handle Optional Dependencies
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
try:
    import tensorflow
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def get_predict_function(model, n, q):
    """
    Takes a model and returns a standardized prediction function that accepts
    a single argument: a 2D q-ary NumPy array.
    """
    raw_predict_function, input_type, sequence_type = _get_model_properties(model, n, q)

    # Build sampling_function
    def final_sampling_function(qary_data):
        """
        This is the function that will be called by your explainer.
        It always expects a q-ary NumPy array and handles all conversions internally.
        """
        if input_type == 'one-hot':
            processed_data = _one_hot_encode(qary_data, q, n)
            return raw_predict_function(model, processed_data)
        elif input_type == 'string':
            string_data = _convert_qary_to_strings(qary_data, sequence_type)
            return raw_predict_function(model, string_data)
        else: # 'qary'
            return raw_predict_function(model, qary_data)

    return final_sampling_function


def _get_model_properties(model, n, q):
    """
    Determines the raw prediction function and the expected input type for a model.
    Returns: (raw_predict_function, input_type, sequence_type)
    'input_type' can be 'qary', 'one-hot', or 'string'.
    'sequence_type' is 'dna', 'rna', or 'protein' if input_type is 'string'.
    """
    if SKLEARN_AVAILABLE and isinstance(model, sklearn.base.BaseEstimator):
        qary = _check_input_type(model, n, q)
        return _predict_sklearn, ('qary' if qary else 'one-hot'), None
    elif XGBOOST_AVAILABLE and isinstance(model, (xgboost.Booster, xgboost.XGBModel)):
        qary = _check_input_type(model, n, q)
        return _predict_xgboost, ('qary' if qary else 'one-hot'), None
    elif TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        qary = _check_input_type(model, n, q)
        return _predict_torch, ('qary' if qary else 'one-hot'), None
    elif TENSORFLOW_AVAILABLE and isinstance(model, tensorflow.keras.Model):
        qary = _check_input_type(model, n, q)
        return _predict_tensorflow, ('qary' if qary else 'one-hot'), None
    elif callable(model):
        return _probe_callable(model, n, q)
    raise TypeError(f"Model type not supported: {str(type(model))}")


def _convert_qary_to_strings(data, sequence_type):
    """Converts a q-ary NumPy array into a list of strings."""
    if sequence_type == 'dna': decoding_map = DNA_DECODING
    elif sequence_type == 'rna': decoding_map = RNA_DECODING
    elif sequence_type == 'protein': decoding_map = PROTEIN_DECODING
    else: raise ValueError("Invalid sequence type for decoding.")
    return ["".join([decoding_map[val] for val in row]) for row in data]


def _probe_callable(model, n, q):
    """
    Probes a callable model to determine its input type and returns its
    wrapper, input_type, and sequence_type. Assumes q-ary inputs are vectorized.
    """
    num_test_samples = 10

    # Function to test whether model accepts DNA, RNA, or protein sequences
    def get_test_strings(num_samples, sequence_type):
        if sequence_type == 'dna':
            alphabet, first_char = ['A', 'C', 'G', 'T'], 'T'
        elif sequence_type == 'rna':
            alphabet, first_char = ['A', 'C', 'G', 'U'], 'U'
        elif sequence_type == 'protein':
            alphabet, first_char = ['L', 'M', 'P', 'Q', 'R', 'S', 'V', 'W', 'Y'], 'L'
        else: return []
        sequences = [first_char + "".join(np.random.choice(alphabet, n - 1)) for _ in range(num_samples)]
        return sequences

    # Check if model takes in numpy arrays
    try: 
        if np.array(model(np.random.randint(0, q, size=(num_test_samples, n)))).shape == (num_test_samples,):
            return (lambda m, d: np.array(m(d))), 'qary', None
    except Exception: pass

    # Check if model takes in strings
    def _probe_string_model(sequence_type):
        try: # List of strings
            if np.array(model(get_test_strings(num_test_samples, sequence_type))).shape == (num_test_samples,):
                return (lambda m, s: np.array(m(s))), 'string', sequence_type
        except Exception: pass
        try: # Single string
            np.array(model(get_test_strings(1, sequence_type)[0])).item()
            return (lambda m, s_list: np.array([m(s) for s in s_list])), 'string', sequence_type
        except Exception: pass
        return None, None, None

    if q == 4:
        fn, i_type, s_type = _probe_string_model('dna')
        if fn: return fn, i_type, s_type
        fn, i_type, s_type = _probe_string_model('rna')
        if fn: return fn, i_type, s_type
    elif q == 20:
        fn, i_type, s_type = _probe_string_model('protein')
        if fn: return fn, i_type, s_type

    raise TypeError("Provided callable model failed all probing tests.")


def _check_input_type(model, n, q):
    """
    Checks if a model likely expects q-ary (n) or one-hot (q*n) input.
    """
    try:
        input_dim = None
        if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            for layer in model.modules():
                if isinstance(layer, torch.nn.Linear):
                    input_dim = layer.in_features
                    break
        elif hasattr(model, 'n_features_in_'):
            input_dim = model.n_features_in_
        elif hasattr(model, 'input_shape') and isinstance(model.input_shape, tuple):
            input_dim = model.input_shape[-1]

        if input_dim is None: return True
        if input_dim == n: return True
        if input_dim == q * n: return False
        return True
    except Exception:
        return True


def _one_hot_encode(data, q, n):
    num_samples = data.shape[0]
    one_hot = np.zeros((num_samples, n, q))
    one_hot[np.arange(num_samples)[:, None], np.arange(n), data] = 1
    return one_hot.reshape(num_samples, n * q)


def _handle_encoding(qary, q, n):
    """
    Decorator that one-hot encodes input data if the model expects it - only for library models.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(model, data):
            if qary:
                processed_data = data
            else: 
                processed_data = _one_hot_encode(data, q, n)
            return func(model, processed_data)
        return wrapper
    return decorator


# Set up prediction functions for library models
def _batch_predict(func):
    @wraps(func)
    def wrapper(model, data):
        batch_size = 64
        predictions = [func(model, data[i:i + batch_size]) for i in range(0, len(data), batch_size)]
        flat_preds = [p[:, 1] if p.ndim > 1 and p.shape[1] > 1 else p.flatten() for p in predictions]
        return np.concatenate(flat_preds)
    return wrapper


@_batch_predict
def _predict_sklearn(model, data):
    return model.predict_proba(data) if hasattr(model, 'predict_proba') else model.predict(data)


@_batch_predict
def _predict_xgboost(model, data):
    if isinstance(model, xgboost.Booster):
        from xgboost import DMatrix
        return model.predict(DMatrix(data))
    return model.predict(data)


@_batch_predict
def _predict_tensorflow(model, data):
    return model.predict(data, verbose=0)


@_batch_predict
def _predict_torch(model, data):
    with torch.no_grad():
        model.eval()
        return model(torch.from_numpy(data).float().to(device)).cpu().detach().numpy()