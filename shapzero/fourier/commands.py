import numpy as np
import argparse
from .helper import Helper
import os
from pathlib import Path
from .utils import save_data, load_data, summarize_results, gwht
import pickle
import pandas as pd
from .input_signal_subsampled import ScriptExit
import time
from .models import get_predict_function
import random
random.seed(42)

def get_qary_indices(params):
    """
    Get q-ary indices to query to compute the Fourier transform

    params: dictionary with the following parameters:
        q (int): The alphabet size.
        n (int): Length of sequence.
        b (int): Subsampling dimension.
        num_subsample (int): The number of subsamples.
        num_repeat (int): The number of repeats.
        exp_dir (str): The experiment directory.
        delays_method_channel (str): The delays method channel.
        query_method (str): The query method (complex default).
        n_samples (int): The number of samples to use for testing.
    """

    # Extract parameters from dictionary
    q = params["q"]
    n = params["n"]
    b = params["b"]
    num_subsample = params["num_subsample"]
    num_repeat = params["num_repeat"]
    exp_dir = Path(params["exp_dir"])
    delays_method_channel = params["delays_method_channel"]
    if "query_method" not in params:
        params["query_method"] = "complex"
    if "n_samples" not in params:
        params["n_samples"] = 10000
    query_method = params["query_method"]
    seed = params.get("seed", None)


    if delays_method_channel:
        delays_method_channel = params["delays_method_channel"]
    else:
        delays_method_channel = "nso"

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    query_args = {
        "query_method": 'generate_samples',
        "train_samples": query_method,
        "method": "generate_samples",
        "num_subsample": num_subsample,
        "num_repeat": num_repeat,
        "b": b,
        "folder": exp_dir,
        "delays_method_channel": delays_method_channel,
        "seed": seed
    }
    signal_args = {
        "n":n,
        "q":q,
        "query_args":query_args,
        "len_seq":n,
        "delays_method_channel": delays_method_channel
        }
    test_args = {
            "n_samples": params["n_samples"],
            "method": "generate_samples"
        }
    file_path = os.path.join(exp_dir, 'train', 'samples')
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(exp_dir, 'test')
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)

    # If M{i}_D{i}_queryindices.pickle is already generated, check to make sure the parameters of b, num_subsample, and num_repeat are correct.
    # Else, re-generate them
    random_i = random.randint(0, num_subsample - 1)
    random_j = random.randint(0, num_repeat - 1)
    query_indices_file = exp_dir / "train" / "samples" / f"M{random_i}_D{random_j}_queryindices.pickle"
    if not query_indices_file.exists():
        pass
    try:
        query_indices = load_data(query_indices_file)        
        # Check if the shape matches the expected dimensions
        if not (query_indices.shape[0] == q**b and query_indices.shape[2] == n):
            # Incorrect shape. Deleting files so they can be regenerated.
            for i in range(num_subsample):
                for j in range(num_repeat):
                    file_types = ["_queryindices", "_qaryindices", ""]
                    for file_type in file_types:
                        file_path = exp_dir / "train" / "samples" / f"M{i}_D{j}{file_type}.pickle"
                        if file_path.exists():
                            os.remove(file_path)
    except Exception as e:
        # If file cannot be loaded, delete so it can be regenerated too
        for i in range(num_subsample):
            for j in range(num_repeat):
                file_types = ["_queryindices", "_qaryindices", ""]
                for file_type in file_types:
                    file_path = exp_dir / "train" / "samples" / f"M{i}_D{j}{file_type}.pickle"
                    if file_path.exists():
                        os.remove(file_path)

    try:
        helper = Helper(signal_args=signal_args, methods=["qsft"], subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir)
    except ScriptExit:
        pass


def compute_fourier_samples(params, complex_queries=False, verbose=True):
    """
    Computes samples for a given set of parameters

    params: dictionary with the following parameters:
        q (int): The alphabet size.
        n (int): Length of sequence.
        b (int): Subsampling dimension.
        num_subsample (int): The number of subsamples.
        num_repeat (int): The number of repeats.
        exp_dir (str): The experiment directory.
        delays_method_channel (str): The delays method channel.
        query_method (str): The query method (simple for fourier).

        sampling_function (function): The sampling function to use. Takes in a numpy array of length (num_samples x n) and returns a 1D numpy array of values (num_samples,).
    """
    # Extract parameters from dictionary
    M = params["num_subsample"]
    D = params["num_repeat"]
    exp_dir = params["exp_dir"]
    sampling_function = params["sampling_function"]

    # Wrap sampling function as a callable, depending on the model passed
    sampling_function = get_predict_function(sampling_function, params["n"], params["q"]) 



    """
    Initialize files
    """
    folder_path = os.path.join(exp_dir)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, "train"))
        os.makedirs(os.path.join(folder_path, "train", "samples"))
        os.makedirs(os.path.join(folder_path, "test"))
    folder_path = os.path.join(folder_path, "train", "samples")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = os.path.join(exp_dir, "train", "samples_mean")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folder_path = os.path.join(exp_dir, "test")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



    """
    Compute samples needed
    """
    for i in range(M):
        for j in range(D):
            query_indices_file = os.path.join(exp_dir, "train", "samples", "M{}_D{}_queryindices.pickle".format(i, j))
            query_indices = load_data(query_indices_file)
            flag = True

            sample_file = os.path.join(exp_dir, "train", "samples", "M{}_D{}.pickle".format(i, j))
            sample_file_mean = os.path.join(exp_dir, "train", "samples_mean", "M{}_D{}.pickle".format(i, j))
            if os.path.isfile(sample_file):
                flag = False

            if flag:
                if sampling_function is None:
                    raise ValueError("Sampling function not defined!")
                all_query_indices = np.concatenate(query_indices)

                all_samples = np.zeros((np.shape(all_query_indices)[0], 1), dtype=complex)
                block_length = len(query_indices[0])
                samples = [np.zeros((len(query_indices), block_length), dtype=complex) for _ in range(1)]
                if verbose:
                    print('- Computing samples for M{}_D{}:'.format(i, j))
                all_samples[:,0] = sampling_function(all_query_indices)
                
                for sample, arr in zip(samples, all_samples.T):
                    for k in range(len(query_indices)):
                        sample[k] = arr[k * block_length: (k+1) * block_length]
                    sample = sample.T
                    save_data(sample, sample_file_mean)



    # Save the empirical mean separately
    folder_path = os.path.join(exp_dir)
    mean_file = os.path.join(exp_dir, "train", "samples", "train_mean.npy") 
    if not os.path.isfile(mean_file):
        all_samples = []
        for i in range(M):
            for j in range(D):
                sample_file = os.path.join(exp_dir, "train", "samples_mean", "M{}_D{}.pickle".format(i, j))
                samples = np.array(load_data(sample_file), dtype=complex)
                samples = np.concatenate(samples)
                all_samples = np.concatenate([all_samples, samples])
        all_samples_mean = np.mean(all_samples).astype(complex)
        np.save(mean_file, all_samples_mean)
    else:
        all_samples_mean = np.load(mean_file).astype(complex)
    
    for i in range(M):
        for j in range(D):
            sample_file_zeromean = os.path.join(exp_dir, "train", "samples", "M{}_D{}.pickle".format(i, j))
            sample_file_mean = os.path.join(exp_dir, "train", "samples_mean", "M{}_D{}.pickle".format(i, j))
            if not os.path.isfile(sample_file_zeromean):
                samples = np.array(load_data(sample_file_mean), dtype=complex)
                if not complex_queries:
                    samples_zeromean = samples - all_samples_mean
                else:
                    samples_zeromean = samples
                save_data(samples_zeromean, sample_file_zeromean)



    """
    Testing samples to compute NMSE and R^2
    """
    query_indices_file = os.path.join(exp_dir, "test", "signal_t_queryindices.pickle")
    query_indices = load_data(query_indices_file)

    query_qaryindices_file = os.path.join(exp_dir, "test", "signal_t_query_qaryindices.pickle")
    query_qaryindices = load_data(query_qaryindices_file)

    # Loop through all files and check if they exist
    sample_file = os.path.join(exp_dir, "test", "signal_t.pickle")
    sample_file_mean = os.path.join(exp_dir, "test", "signal_t_mean.pickle")
    flag = True

    if os.path.isfile(sample_file):
        flag = False

    if flag:
        if sampling_function is None:
            raise ValueError("Sampling function not defined!")
        all_query_indices = query_indices

        all_samples = np.zeros((np.shape(all_query_indices)[0], 1), dtype=complex)
        block_length = len(query_indices[0])
        samples = [np.zeros((len(query_indices), block_length), dtype=complex) for _ in range(1)]
        if verbose:
            print('- Computing test samples:')
        all_samples[:,0] = sampling_function(all_query_indices)

        for arr in all_samples.T:
            sample_file = os.path.join(exp_dir, "test", "signal_t.pickle")
            sample_file_mean = os.path.join(exp_dir, "test", "signal_t_mean.pickle")
            samples_dict = dict(zip(query_qaryindices, arr))
            save_data(samples_dict, sample_file)
            save_data(samples_dict, sample_file_mean)

        # Remove empirical mean
        mean_file = os.path.join(exp_dir, "train", "samples", "train_mean.npy")
        all_samples_mean = np.load(mean_file).astype(complex)

        sample_file_mean = os.path.join(exp_dir, "test", "signal_t_mean.pickle")
        sample_file = os.path.join(exp_dir, "test", "signal_t.pickle")
        samples_dict = load_data(sample_file_mean)

        all_values = list(samples_dict.values())
        if not complex_queries:
            all_values = np.array(all_values, dtype=complex) - all_samples_mean
        else:
            all_values = np.array(all_values, dtype=complex)
        samples_dict = {key: value for key, value in zip(samples_dict.keys(), all_values)}
        save_data(samples_dict, sample_file)


def run_fourier(params, verbose=True):
    """
    Estimate the Fourier transform of a given set of parameters

    params: dictionary with the following parameters:
        q (int): The alphabet size.
        n (int): Length of sequence.
        b (int): Subsampling dimension.
        num_subsample (int): The number of subsamples.
        num_repeat (int): The number of repeats.
        exp_dir (str): The experiment directory.
        delays_method_channel (str): The delays method channel.
        query_method (str): The query method (complex default).
        n_samples (int): The number of samples to use for testing.
        verbose (bool): Whether to print the progress of the computation.

        noise_sd (float): noise_sd to use if hyperparam is False.
        hyperparam (bool): Whether to use a hyperparameter.
        hyperparam_range (list): The range of hyperparameters search over to find optimal noise_sd is hyperparam is True: [min, max, step].
    """
    # Extract parameters from dictionary
    q = params["q"]
    n = params["n"]
    b = params["b"]
    num_subsample = params["num_subsample"]
    num_repeat = params["num_repeat"]
    exp_dir = params["exp_dir"]
    delays_method_source = "identity"
    delays_method_channel = params["delays_method_channel"]
    if "noise_sd" not in params:
        noise_sd = 0
    else:
        noise_sd = params["noise_sd"]

    if "hyperparam" not in params:
        hyperparam = False
        hyperparam_range = None
    else:
        hyperparam = params["hyperparam"]
        if "hyperparam_range" not in params and hyperparam:
            raise ValueError("hyperparam_range must be specified if hyperparam is True.")
        elif not hyperparam:
            hyperparam_range = None
        else:
            hyperparam_range = params["hyperparam_range"]

    if "query_method" not in params:
        query_method = "complex"
    else: 
        query_method = params["query_method"]
    if delays_method_channel:
        delays_method_channel = params["delays_method_channel"]
    else:
        delays_method_channel = "nso"



    """
    Initialization
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    query_args = {
        "query_method": query_method,
        "num_subsample": num_subsample,
        "delays_method_source": delays_method_source,
        "subsampling_method": "qsft",
        "delays_method_channel": delays_method_channel,
        "num_repeat": num_repeat,
        "b": b,
        "folder": exp_dir 
    }
    signal_args = {
                    "n":n,
                    "q":q,
                    "noise_sd":noise_sd,
                    "query_args":query_args,
                    }
    test_args = {
            "n_samples": params["n_samples"]
        }

    model_kwargs = {}
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_repeat"] = num_repeat
    model_kwargs["b"] = b



    """
    Recover Fourier coefficients and get summary statistics
    """
    helper = Helper(signal_args=signal_args, methods=["qsft"], subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir)

    model_kwargs = {}
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_repeat"] = num_repeat
    model_kwargs["b"] = b
    test_kwargs = {}
    model_kwargs["n_samples"] = num_subsample * (helper.q ** b) * num_repeat * (helper.n + 1)

    if hyperparam:
        if verbose:
            print('Hyperparameter tuning noise_sd:')
        start_time_hyperparam = time.time()
        range_values = [float(x) for x in hyperparam_range]
        noise_sd = np.arange(range_values[0], range_values[1], range_values[2]).round(3)
        nmse_entries = []
        r2_entries = []

        for noise in noise_sd:
            signal_args.update({
                "noise_sd": noise
            })
            model_kwargs["noise_sd"] = noise
            model_result = helper.compute_model(method="qsft", model_kwargs=model_kwargs, report=True, verbosity=0)
            test_kwargs["beta"] = model_result.get("gwht")
            nmse, r2 = helper.test_model("qsft", **test_kwargs)
            gwht = model_result.get("gwht")
            locations = model_result.get("locations")
            n_used = model_result.get("n_samples")
            avg_hamming_weight = model_result.get("avg_hamming_weight")
            max_hamming_weight = model_result.get("max_hamming_weight")
            nmse_entries.append(nmse)
            r2_entries.append(r2)
            if verbose:
                print(f"noise_sd: {noise} - NMSE: {nmse}, R2: {r2}")

        end_time_hyperparam= time.time()
        elapsed_time_hyperparam = end_time_hyperparam - start_time_hyperparam
        min_nmse_ind = nmse_entries.index(min(nmse_entries))
        min_nmse = nmse_entries[min_nmse_ind]
        if verbose:
            print('----------')
            print(f"Hyperparameter tuning time: {elapsed_time_hyperparam} seconds")
            print(f"noise_sd: {noise_sd[min_nmse_ind]} - Min NMSE: {min_nmse}")

        # Recompute qsft with the best noise_sd
        signal_args.update({
            "noise_sd": noise_sd[min_nmse_ind]
        })
        model_kwargs["noise_sd"] = noise_sd[min_nmse_ind]
        model_result = helper.compute_model(method="qsft", model_kwargs=model_kwargs, report=True, verbosity=0)
        test_kwargs["beta"] = model_result.get("gwht")
        nmse, r2_value = helper.test_model("qsft", **test_kwargs)
        gwht = model_result.get("gwht")
        locations = model_result.get("locations")
        n_used = model_result.get("n_samples")
        avg_hamming_weight = model_result.get("avg_hamming_weight")
        max_hamming_weight = model_result.get("max_hamming_weight")

        plt.figure()
        plt.title(f'q{q}_n{n}_b{b}')
        plt.plot(noise_sd, nmse_entries[:], marker='o', linestyle='-', color='b')
        plt.scatter(noise_sd[min_nmse_ind], nmse_entries[min_nmse_ind], color='red', marker='x', label='Min NMSE')
        plt.text(noise_sd[min_nmse_ind], nmse_entries[min_nmse_ind], f'noise_sd: {noise_sd[min_nmse_ind]} - Min NMSE: {min_nmse:.2f}', ha='right', va='top')
        plt.xlabel('noise_sd')
        plt.ylabel('NMSE')
        plt.savefig(str(exp_dir) + '/nmse.png')  
        df = pd.DataFrame({'noise_sd': noise_sd, 'nmse': nmse_entries})
        df.to_csv(str(exp_dir) + '/nmse.csv', index=False)
        noise_sd = noise_sd[min_nmse_ind]

    else:
        model_kwargs["noise_sd"] = noise_sd
        model_result = helper.compute_model(method="qsft", model_kwargs=model_kwargs, report=True, verbosity=0)
        test_kwargs["beta"] = model_result.get("gwht")
        nmse, r2_value = helper.test_model("qsft", **test_kwargs)
        gwht = model_result.get("gwht")
        locations = model_result.get("locations")
        n_used = model_result.get("n_samples")
        avg_hamming_weight = model_result.get("avg_hamming_weight")
        max_hamming_weight = model_result.get("max_hamming_weight")
        print('----------')
        print(f"R^2 is {r2_value:.2f}")

    with open(str(exp_dir) + "/" + "fourier_transform.pickle", "wb") as pickle_file:
        pickle.dump(gwht, pickle_file)

    summarize_results(locations, gwht, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight, exp_dir, params)
    print('----------')


def evaluate_nmse(params, search_range, verbose=True):
    """
    Evaluates the NMSE for a given range of noise_sd values.
    This is a reusable helper for hyperparameter tuning.
    """
    # --- 1. Perform initial setup (extracted from run_fourier) ---
    q = params["q"]
    n = params["n"]
    b = params["b"]
    num_subsample = params["num_subsample"]
    num_repeat = params["num_repeat"]
    exp_dir = Path(params["exp_dir"])
    delays_method_source = "identity"
    delays_method_channel = params.get("delays_method_channel")
    if "query_method" not in params:
        query_method = "complex"
    else: 
        query_method = params["query_method"]
    if delays_method_channel:
        delays_method_channel = params["delays_method_channel"]
    else:
        delays_method_channel = "nso"
    if "n_samples" not in params:
        params["n_samples"] = 10000


    
    """
    Initialization
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    query_args = {
        "query_method": query_method,
        "num_subsample": num_subsample,
        "delays_method_source": delays_method_source,
        "subsampling_method": "qsft",
        "delays_method_channel": delays_method_channel,
        "num_repeat": num_repeat,
        "b": b,
        "folder": exp_dir 
    }
    signal_args = {
                    "n":n,
                    "q":q,
                    "noise_sd":search_range[0],
                    "query_args":query_args,
                    }
    test_args = {
            "n_samples": params["n_samples"]
            }
    
    helper = Helper(signal_args=signal_args, methods=["qsft"], subsampling_args=query_args, test_args=test_args, exp_dir=exp_dir)
    model_kwargs = {}
    model_kwargs["num_subsample"] = num_subsample
    model_kwargs["num_repeat"] = num_repeat
    model_kwargs["b"] = b
    test_kwargs = {}
    model_kwargs["n_samples"] = num_subsample * (helper.q ** b) * num_repeat * (helper.n + 1)
    
    nmse_results = []
    if verbose:
        print(f"- - Evaluating {len(search_range)} noise_sd values...")

    # --- 2. Loop through the search range and get NMSE for each ---
    for noise in search_range:
        signal_args.update({
                "noise_sd": noise
            })
        model_kwargs["noise_sd"] = noise
        model_result = helper.compute_model(method="qsft", model_kwargs=model_kwargs, report=True, verbosity=0)
        test_kwargs["beta"] = model_result.get("gwht")
        nmse, _ = helper.test_model("qsft", **test_kwargs)
        nmse_results.append(nmse)
        if verbose:
            print(f"- - - noise_sd: {noise:.3e}, NMSE: {nmse:.4f}")
        
    return np.array(nmse_results)


def compute_subsampled_fourier_transforms(params, exp_dir):
    """
    Computes subsampled Fourier transforms (transforms/U{i}_{j}) for all samples (samples/M{i}_D{j}.pickle) in params['exp_dir']

    params: dictionary with the following parameters:
        q (int): The alphabet size.
        b (int): Subsampling dimension.
        num_subsample (int): The number of subsamples.
        num_repeat (int): The number of repeats.
    exp_dir (str): The experiment directory.

    Repurposed from fourier/input_signal_subsampled.py
    """

    # Extract parameters from dictionary
    q = params["q"]
    all_bs = [params["b"]]
    Ms = params["num_subsample"]
    Ds = params["num_repeat"]

    def compute_subtransform(samples, q, b):
        transform = [gwht(row[::(q ** (b - b))], q, b) for row in samples]
        return transform

    # Compute subsampled Fourier transforms
    Us = [[{} for j in range(Ds)] for i in range(Ms)]
    transformTimes = [[{} for j in range(Ds)] for i in range(Ms)]
    for i in range(Ms):
        for j in range(Ds):
            transform_file = Path(exp_dir) / "train" / "transforms" / f"U{i}_D{j}.pickle"
            if transform_file.is_file():
                continue
            else:
                sample_file = Path(exp_dir) / "train" / "samples" / f"M{i}_D{j}.pickle"
                samples = load_data(sample_file)
                for b in all_bs:
                    Us[i][j][b] = compute_subtransform(samples, q, b)
                if exp_dir:
                    save_data((Us[i][j], transformTimes[i][j]), transform_file)