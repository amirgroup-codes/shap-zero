"""
Main Explainer class to compute the Fourier transform and find SHAP values and interactions. 

Computing the Fourier transform is done using the commands.py file in the fourier/ directory.
Converting to the Mobius transform and computing SHAP values and interactions is done using the shap_explainer.py file in the shap/ directory.
"""
from pathlib import Path
import pickle
import numpy as np
from .fourier import commands
from .fourier.utils import find_number_of_samples_used, load_data
from .shap.shap_explainer import shap_explainer
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import json
import os
import pandas as pd

DNA_ENCODING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
RNA_ENCODING = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
PROTEIN_ENCODING = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
    'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
    'V': 17, 'W': 18, 'Y': 19
}

class Explainer:
    """
    An explainer object that 1) computes the q-ary Fourier transform of a given sampling function, 2) converts the Fourier transform to SHAP zero values and interactions, and 3) plots the results.
    """
    def __init__(self, q, n, exp_dir, sampling_function, fourier_transform, fourier_params):
        self.q = q
        self.n = n
        self.sampling_function = sampling_function
        self.exp_dir = Path(exp_dir)
        self.fourier_transform = fourier_transform
        if self.fourier_transform is not None:
            if isinstance(self.fourier_transform, (str, Path)):
                fourier_path = str(self.fourier_transform)
                self.fourier_transform = np.load(fourier_path, allow_pickle=True)
            elif isinstance(self.fourier_transform, dict):
                pass
            else:
                raise TypeError("fourier_transform must be a Path, a string, or a dictionary.")
        self.run_params = fourier_params
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output files will be saved in: {self.exp_dir.resolve()}")


    def compute_fourier_transform(self, budget=1024, verbose=True, max_tuning_iterations=5, num_points_per_iteration=10):
        """
        Computes the Fourier transform using the initialized parameters.

        Args:
            budget (int): The maximum number of samples the sampling_function can be called.
            verbose (bool): Whether to print detailed logs.
            max_tuning_iterations (int): The maximum number of tuning iterations for noise_sd.
            num_points_per_iteration (int): The number of noise_sd points to evaluate per iteration.

        Returns:
            A dictionary representing the sparse Fourier transform.
        """
        # Find the best parameters to use within the budget
        # Load in parameters, otherwise generate
        if self.run_params:
            self._check_run_params()    
        else:
            self.run_params = self._find_best_params(budget)
            output_path = os.path.join(self.exp_dir, "params.json")
            with open(output_path, 'w') as f:
                json.dump(self.run_params, f, indent=4)
            print(f"Fourier parameters saved to {output_path}")
        if verbose:
            print(f"Optimized parameters found: b={self.run_params['b']}, num_subsample (C)={self.run_params['num_subsample']}, num_repeat (P1)={self.run_params['num_repeat']}. Expected samples to compute: {self.run_params['total_samples_required']}")

        # Compute samples and subsampled Fourier transforms
        parameters = {
            "q": self.q,
            "n": self.n,
            "b": self.run_params['b'],
            "num_subsample": self.run_params['num_subsample'],
            "num_repeat": self.run_params['num_repeat'],
            "sampling_function": self.sampling_function,
            "exp_dir": self.exp_dir,
            "delays_method_channel": "nso",
            "n_samples": 1000,
            "hyperparam": False,
        }
        if verbose:
            print("\n####################")
            print(f"Estimating Fourier transform...")
            print("####################\n")
            print(f"Generating query indices...")
        commands.get_qary_indices(parameters)
        if verbose:
            print(f"Computing samples and their subsampled Fourier transforms...")
        commands.compute_fourier_samples(parameters)
        commands.compute_subsampled_fourier_transforms(parameters, exp_dir=self.exp_dir)



        """
        Estimate noise_sd hyperparameter that minimizes NMSE
        *Updated search procedure that's different than Appendix E of SHAP zero. Optimized so the user does not have to specify the range of noise_sd values to search over.*

        Adaptive search procedure:
        - Obtain an initial estimate of noise_sd
        - Define a range of noise_sd values to search over given by max_tuning_iterations and num_points_per_iteration
        - For each noise_sd value, compute NMSE and R^2
        - Choose the noise_sd value that minimizes NMSE
        """
        initial_noise_sd = self._obtain_initial_estimate_noise_sd()
        if verbose: 
            print(f"Now tuning noise_sd hyperparameter...")
            print(f"- Initial heuristic for noise_sd: {initial_noise_sd:.4e}")
        search_range = np.logspace(
            np.log10(initial_noise_sd / 10),
            np.log10(initial_noise_sd * 10),
            num=num_points_per_iteration
        )
        search_range = np.concatenate(([0.1], search_range)) # Alaways include 0.1 in the search range 
        best_overall_noise_sd = None
        best_overall_nmse = float('inf')
        for i in range(max_tuning_iterations):
            if verbose:
                print(f"- - - Tuning Iteration {i+1}/{max_tuning_iterations}")
            params = {**self.run_params, "q": self.q, "n": self.n, "exp_dir": self.exp_dir}
            nmse_values = commands.evaluate_nmse(params, search_range, verbose=verbose)
            results = dict(zip(search_range, nmse_values))
            current_best_noise, current_best_nmse = min(results.items(), key=lambda item: item[1])
            if current_best_nmse < best_overall_nmse:
                best_overall_nmse = current_best_nmse
                best_overall_noise_sd = current_best_noise
            else:
                if verbose:
                    print("- - - NMSE did not improve in this iteration. Halting search.")
                break
            
            if best_overall_nmse < 0.1:
                if verbose:
                    print(f"- - - NMSE is {best_overall_nmse:.4f} < 0.1. Halting search.")
                break
                best_overall_noise_sd = current_best_noise

            # Define the next search range based on the best performing noise_sds
            sorted_results = sorted(results.items(), key=lambda item: item[1])
            best_noise_sd_1 = sorted_results[0][0]
            best_noise_sd_2 = sorted_results[1][0]
            search_range = np.sort(search_range)
            idx1 = np.where(search_range == best_noise_sd_1)[0][0]
            idx2 = np.where(search_range == best_noise_sd_2)[0][0]
            # Define lower and upper bounds for the search range
            lower_idx = max(0, min(idx1, idx2) - 1)
            upper_idx = min(len(search_range) - 1, max(idx1, idx2) + 1)
            if lower_idx < upper_idx:
                new_lower_bound = search_range[lower_idx]
                new_upper_bound = search_range[upper_idx]
                if verbose:
                    print(f"- - - Minimum found near {best_noise_sd_1:.2e} and {best_noise_sd_2:.2e}. Zooming in on range [{new_lower_bound:.2e}, {new_upper_bound:.2e}].")
                search_range = np.linspace(new_lower_bound, new_upper_bound, 10)
            else:
                if verbose:
                    print("- - - Optimal region found. Halting search.")
                break
    
        # Compute final Fourier transform with the best noise_sd
        if verbose:
            print(f"Recomputing final transform with optimal noise_sd...")
        parameters.update({"noise_sd": best_overall_noise_sd})
        commands.run_fourier(parameters, verbose=verbose)
        
        transform_file = self.exp_dir / "fourier_transform.pickle"
        if transform_file.exists():
            with open(transform_file, "rb") as f:
                self.fourier_transform = pickle.load(f)
            print(f"Successfully computed and loaded the Fourier transform at {transform_file}")
        else:
            warnings.warn("Warning: Fourier transform file was not generated.")
            self.fourier_transform = None
        return self.fourier_transform
        
    
    def explain(self, sample, explanation='shap_value'):
        """
        Explain a single sequence or a batch of sequences.

        Args:
            sample (str or list): A single sequence can be a string (e.g. 'ACGTACGTAC') or a list of strings (e.g. ['ACGTACGTAC', 'ACGTACGTAC', 'ACGTACGTAC']).
            explanation (str): The type of explanation to generate. Can be 'shap_value' or 'interaction'.

        Returns:
            A list of dictionaries containing the SHAP zero values and interactions for each sequence.
        """
        self.explantion = explanation
        is_batch = isinstance(sample, list)
        if is_batch:
            self.sequences = sample
            self.shapzero_results = [self._explain_sequence(s, explanation) for s in sample]
            return self.shapzero_results
        elif isinstance(sample, str):
            shapzero_results = self._explain_sequence(sample, explanation)
            self.sequences = [sample]
            self.shapzero_results = [shapzero_results]
            return [shapzero_results]
        else:
            raise TypeError(f"Input must be a string (e.g. 'ACGTACGTAC') or a list of strings (e.g. ['ACGTACGTAC', 'ACGTACGTAC', 'ACGTACGTAC']), but got {type(sample)}")


    def plot(self, output_path=None, title=None, y_label=None, min_order=2, legend=True):
        """
        Plots the SHAP zero values and interactions for the input sequences.

        Args:
            output_path (str, optional): The path where the output plot image will be saved.
            title (str, optional): The title for the plot.
            y_label (str, optional): The label for the y-axis.
            min_order (int, optional): The minimum order of interactions to include in the plot (only for interaction plots).
            legend (bool, optional): Whether to include a legend in the plot.
        """
        if output_path is None:
            if self.explantion == 'shap_value':
                output_path = self.exp_dir / 'shap_values.png'
            elif self.explantion == 'interaction':
                output_path = self.exp_dir / 'interaction.png'
            else: 
                raise ValueError(f"Explanation method {self.explantion} not supported.")

        if self.q != 4 and self.q != 20:
            raise ValueError(f"Plotting is only supported for q=4 and q=20. q={self.q} is not supported.")
    
        if self.explantion == 'shap_value' and self.q == 4:
            if y_label is None:
                y_label = 'SHAP zero value'
            self._plot_shap_values_dna_rna(output_path=output_path, title=title, y_label=y_label, legend=True)
        elif self.explantion == 'interaction' and self.q == 4:
            if y_label is None:
                y_label = 'SHAP zero interaction'
            self._plot_interactions_dna_rna(output_path=output_path, min_order=min_order, title=title, y_label=y_label, legend=True)
        elif self.explantion == 'shap_value' and self.q == 20:
            if y_label is None:
                y_label = 'SHAP zero value'
            self._plot_shap_values_proteins(output_path=output_path, title=title, y_label=y_label, legend=True)
        elif self.explantion == 'interaction' and self.q == 20:
            if y_label is None:
                y_label = 'SHAP zero interaction'
            self._plot_interactions_proteins(output_path=output_path, title=title, y_label=y_label, legend=True)
        else:
            raise ValueError(f"Explanation method {self.explantion} not supported.")


    def save(self, output_folder=None, top_values=10, min_order=2):
        """
        Save all SHAP zero values and interactions, as well as the top values and interactions, for the input sequences.

        Args:
            output_folder (str, optional): Folder where the output csv will be saved.
            top_values (int, optional): How many values to save per positive and negative grouping.
            min_order (int, optional): The minimum order of interactions to include in the plot (only for interaction plots).
        """
        if output_folder is None:
            output_folder = self.exp_dir
        else:
            output_folder = Path(output_folder)

        if self.explantion == 'shap_value':
            self._save_top_shap_values(output_folder, top_values=top_values)
            self._save_all_shap_values(output_folder)
        elif self.explantion == 'interaction':
            self._save_top_interactions(output_folder, top_values=top_values, min_order=min_order)
            self._save_all_interactions(output_folder)
        else:
            raise ValueError(f"Explanation method {self.explantion} not supported.")


    def _check_run_params(self):
        """
        Checks to make sure self.run_params is formatted properly as a json, a valid path as a string, or a dictionary.
        """
        if isinstance(self.run_params, (str, Path)):
            output_path = Path(self.exp_dir) / "params.json"
            try:
                with open(self.run_params, 'r') as f:
                    loaded_params = json.load(f)
                # Check for required parameters and update total_samples_required
                required_keys = ["b", "num_subsample", "num_repeat"]
                for key in required_keys:
                    if key not in loaded_params:
                        raise ValueError(f"Missing required parameter: '{key}' in the loaded configuration.")
                b = loaded_params["b"]
                num_subsample = loaded_params["num_subsample"]
                num_repeat = loaded_params["num_repeat"]
                loaded_params["total_samples_required"] = find_number_of_samples_used(self.q, self.n, b, num_subsample, num_repeat)
                with open(output_path, 'w') as f:
                    json.dump(loaded_params, f, indent=4)
                self.run_params = loaded_params
                print(f"Parameters loaded from {self.run_params} and saved to {output_path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found at: {self.run_params}")
            except json.JSONDecodeError:
                raise ValueError(f"File at {self.run_params} is not a valid JSON file.")    
        elif isinstance(self.run_params, dict):
            # Check for required parameters and update total_samples_required
            required_keys = ["b", "num_subsample", "num_repeat"]
            for key in required_keys:
                if key not in self.run_params:
                    raise ValueError(f"Missing required parameter: '{key}' in the loaded configuration.")
            b = self.run_params["b"]
            num_subsample = self.run_params["num_subsample"]
            num_repeat = self.run_params["num_repeat"]
            self.run_params["total_samples_required"] = find_number_of_samples_used(self.q, self.n, b, num_subsample, num_repeat)
            output_path = os.path.join(self.exp_dir, "params.json")
            with open(output_path, 'w') as f:
                json.dump(self.run_params, f, indent=4)
            print(f"Parameters saved to {output_path}")
        else:
            raise TypeError(f"self.run_params must be a string, Path, or dictionary, not {type(self.run_params).__name__}.") 


    def _find_best_params(self, budget):
        """
        Finds the optimal b, num_subsample, and num_repeat to maximize samples within the budget.
        *Roughly follows Appendix E.1 of SHAP zero. Updated to maximize a user budget with b, C and P1, instead of just increasing b as originally reported.*
        """
        # Start with num_subsample (C) and num_repeat (P1) set to 3, as the recommended defaults in SHAP zero
        num_subsample = 3
        num_repeat = 3
        best_b = 0

        # First, maximize b as large as possible (same as Appendix E.1)
        while num_subsample >= 1 and num_repeat >= 1:
            temp_b = 0
            for b_candidate in range(1, self.n + 1):
                cost = find_number_of_samples_used(self.q, self.n, b_candidate, num_subsample, num_repeat)
                if cost <= budget:
                    temp_b = b_candidate
                else:
                    break # This b is too expensive, stop searching higher b's
            if temp_b > 0:
                best_b = temp_b
                break 
            # If it happens that no b works, we'll lower num_subsample and num_repeat and try again (unrecommended to be this low with budget but we set this for low-resource users)
            if num_subsample > num_repeat:
                num_subsample -= 1
            elif num_repeat > num_subsample:
                num_repeat -= 1
            else: # If they are equal, decrease num_subsample first
                num_subsample -= 1
        if best_b == 0:
            raise ValueError(f"Budget of {budget} is too small. Even with b=1, C=1, and P1=1, it is exceeded.")
        b = best_b

        # Check how many samples it would cost to go to the next b. Print to user to increase b if the cost is insignificant.
        if b + 1 <= self.n: 
            cost_next_b = find_number_of_samples_used(self.q, self.n, b + 1, num_subsample, num_repeat)
            cost_diff = cost_next_b - budget
            if cost_diff > 0 and (cost_diff / budget <= 0.10 or cost_diff < 1000):
                warnings.warn(f"To significantly improve results, consider increasing your budget to {cost_next_b} samples (an increase of {cost_diff} samples).")

        # Now, increase num_subsample and num_repeat until we find one that maximizes the budget
        while True:
            # Increase num_subsample
            cost_next_subsample = find_number_of_samples_used(self.q, self.n, b, num_subsample + 1, num_repeat)
            can_increase_subsample = cost_next_subsample <= budget
            if can_increase_subsample:
                num_subsample += 1
            
            # Increase num_repeat
            cost_next_repeat = find_number_of_samples_used(self.q, self.n, b, num_subsample, num_repeat + 1)
            can_increase_repeat = cost_next_repeat <= budget
            if can_increase_repeat:
                num_repeat += 1
            
            # If we couldn't increase either, we're done
            if not can_increase_subsample and not can_increase_repeat:
                break
                
        # Projected number of samples to use, within the constraint of the user budget
        final_samples = find_number_of_samples_used(self.q, self.n, b, num_subsample, num_repeat)
        return {
            "b": b,
            "num_subsample": num_subsample,
            "num_repeat": num_repeat,
            "total_samples_required": final_samples
        }


    def _obtain_initial_estimate_noise_sd(self):
        """
        Obtains an initial estimate of the noise_sd hyperparameter that minimizes NMSE.
        We use the heuristic of 0.05 * median magnitude of the Fourier coefficients for the cutoff threshold and then tune the noise_sd hyperparameter.
        """
        exp_dir = self.exp_dir
        Ms = self.run_params['num_subsample']
        Ds = self.run_params['num_repeat']

        # Concatenate all subsampled Fourier transforms into one long array
        all_transforms = []
        for i in range(Ms):
            for j in range(Ds):
                transform_file = Path(exp_dir) / "train" / "transforms" / f"U{i}_D{j}.pickle"
                if transform_file.is_file():
                    all_transforms.append(np.array(list(load_data(transform_file)[0].values())).flatten())
        Us = np.concatenate(all_transforms).flatten()

        # Estimate noise_sd hyperparameter that minimizes NMSE based on the magnitudes of the Fourier coefficients as 0.05 times the median magnitude
        magnitudes = np.abs(Us)
        print('MAG', magnitudes[0], len(magnitudes))
        median = np.median(magnitudes)
        estimated_cutoff = 0.05 * median
        estimated_noise_sd = self._compute_noise_sd_from_cutoff(estimated_cutoff, self.q, self.run_params['b'])
        return estimated_noise_sd


    def _compute_noise_sd_from_cutoff(self, cutoff, q, b, gamma=0.5):
        """
        Computes the implied noise_sd from a given cutoff value.
        Formula taken from q-SFT GitHub line 125 (https://github.com/basics-lab/qsft/blob/master/qsft/qsft.py)
        
        Args:
            cutoff (float): The cutoff value to use for the noise_sd hyperparameter.
            q (int): The alphabet size.
            b (int): The number of bases.
            gamma (float, optional): The constant used in the noise_sd calculation. Defaults to 0.5.

        Returns:
            float: The implied noise_sd value.
        """
        value = (cutoff - 1e-9) * (q ** b) / (1 + gamma)
        if value < 0:
            return 0
        return np.sqrt(value)


    def _explain_sequence(self, sample, explanation):
        """
        Function that runs SHAP zero for a single sequence.
        
        Args:
            sample (str): The sequence to explain.
            explanation (str): The type of explanation to generate. Can be 'shap_value' or 'interaction'.

        Returns:
            A dictionary containing the SHAP zero values and interactions for the sequence.
        """
        # Preprocess string to to q-ary NumPy array ---
        processed_sample = sample.upper()
        unique_chars = set(processed_sample)
        
        # Check whether sequences are DNA, RNA, or proteins
        self.encoding_map = None
        if self.q == 20 and unique_chars.issubset(PROTEIN_ENCODING.keys()):
            self.encoding_map = PROTEIN_ENCODING
        elif self.q == 4:
            if 'U' in unique_chars and unique_chars.issubset(RNA_ENCODING.keys()):
                self.encoding_map = RNA_ENCODING
            elif unique_chars.issubset(DNA_ENCODING.keys()):
                self.encoding_map = DNA_ENCODING
        if self.encoding_map is None:
            raise ValueError(f"Input string '{sample}' contains invalid characters for q={self.q}.")
            
        try:
            encoded_sample = np.array([self.encoding_map[char] for char in processed_sample])
        except KeyError as e:
            raise ValueError(f"Character {e} not found in the selected encoding map.")

        # Run SHAP zero
        shap_zero = shap_explainer(self.fourier_transform, q=self.q, n=self.n)
        if explanation in ('shap_value', 'interaction'):
            return shap_zero.explain(encoded_sample, explanation=explanation)
        else:
            raise ValueError(f"Explanation method '{explanation}' not supported.")

    
    def _convert_shap_to_array(self):
        """
        Converts a list of SHAP dictionaries into a 2D NumPy array.

        Args:
            self.shapzero_results (list): A list of dictionaries, where each dict contains SHAP values with position tuples as keys.
            self.n (int): The total length of the sequence (the number of features).

        Returns:
            np.ndarray: A 2D array of shape (num_samples, n).
        """
        num_samples = len(self.shapzero_results)
        shap_array = np.zeros((num_samples, self.n))
        for i, shap_dict in enumerate(self.shapzero_results):
            for position_tuple, value in shap_dict.items():
                position_index = position_tuple[0]
                if 0 <= position_index < self.n:
                    shap_array[i, position_index] = value
                else:
                    print(f"Warning: Position index {position_index} is out of bounds for n={self.n}. Skipping.")
        return shap_array


    def _plot_shap_values_dna_rna(self, output_path, title=None, y_label='SHAP zero value', legend=True):
        """
        Creates and saves a plot of SHAP values for one or more sequences for DNA and RNA.

        Args:
            self.sequences (list): A list of sequence strings (e.g., ['ACGT']).
            self.shapzero_results (list): A list of dictionaries, where each dict contains SHAP values with position tuples as keys.
            self.encoding_map (dict): A dictionary mapping nucleotides to their corresponding encodings.
            output_path (str or Path): The path where the output plot image will be saved.
            title (str, optional): The title for the plot.
            y_label (str, optional): The label for the y-axis.
            legend (bool, optional): Whether to include a legend in the plot.
        """
        shap_values = self._convert_shap_to_array()
        if len(self.sequences) < 30:
            alpha = 1
        else:
            alpha = 0.6

        if self.encoding_map == DNA_ENCODING:
            colors = {'A': '#008000', 'C': '#0000ff', 'G': '#ffa600', 'T': '#ff0000'}
            markers = {'A': 'o',  'C': 's',  'G': 'd',  'T': '^'}
        elif self.encoding_map == RNA_ENCODING:
            colors = {'A': '#008000', 'C': '#0000ff', 'G': '#ffa600', 'U': '#ff0000'}
            markers = {'A': 'o',  'C': 's',  'G': 'd',  'U': '^'}
        else:
            raise ValueError(f"Encoding map {self.encoding_map} not supported.")

        num_sequences, seq_length = np.shape(shap_values)
        width_per_position = 0.3
        fig_width = max(8, min(25, seq_length * width_per_position))
        fig, ax = plt.subplots(figsize=(fig_width, 4), dpi=300)
        for seq, sample in zip(self.sequences, shap_values):
            for i, nt in enumerate(seq):
                ax.scatter(
                    i + 1,
                    sample[i],
                    color=colors.get(nt.upper(), 'gray'),
                    marker=markers.get(nt.upper(), 'x'),
                    alpha=alpha,
                )
        ax.set_xlabel('Sequence position')
        ax.set_ylabel(y_label)
        ax.tick_params(axis='both', width=0.5)
        ax.set_xlim(0.5, seq_length + 0.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        tick_interval = 1 if seq_length <= 20 else 5
        tick_positions = [i for i in range(1, seq_length + 1) if i % tick_interval == 0]
        ax.set_xticks(tick_positions)
        if title:
            ax.set_title(title, fontweight='bold')
        if legend:
            legend_handles = [
                plt.Line2D([0], [0], marker=markers.get(nt), color='w',
                        markerfacecolor=colors.get(nt), markersize=8, label=nt)
                for nt in colors.keys()
            ]
            ax.legend(handles=legend_handles, loc='best', ncol=len(colors))
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig) 


    def _plot_interactions_dna_rna(self, output_path, min_order=2, title=None, y_label='SHAP zero interaction', legend=True):
        """
        Creates and saves a stacked bar plot of SHAP interaction values for DNA and RNA.

        Args:
            self.sequences (list): A list of sequence strings (e.g., ['ACGT']).
            self.shapzero_results (list): A list of dictionaries, where each dict contains SHAP interactions with position tuples as keys.
            self.encoding_map (dict): A dictionary mapping nucleotides to their corresponding encodings.
            output_path (str or Path): The path where the output plot image will be saved.
            min_order (int, optional): The minimum order of interactions to include in the plot. Defaults to 2.
            title (str, optional): The title for the plot.
            y_label (str, optional): The label for the y-axis.
            legend (bool, optional): Whether to include a legend in the plot.
        """ 
        seq_length = len(self.sequences[0])
        if self.encoding_map == DNA_ENCODING:
            colors = {'A': '#008000', 'C': '#0000ff', 'G': '#ffa600', 'T': '#ff0000'}
        elif self.encoding_map == RNA_ENCODING:
            colors = {'A': '#008000', 'C': '#0000ff', 'G': '#ffa600', 'U': '#ff0000'}
        else:
            raise ValueError(f"Encoding map {self.encoding_map} not supported.")

        interaction_values_positive = {}
        interaction_values_negative = {}
        # Condense higher order interactions per position
        def group_by_position(interaction_dict, length, keys):
            grouped = {pos: {nuc: 0 for nuc in keys} for pos in range(length)}
            for (pos, nuc), value in interaction_dict.items():
                if pos < length:
                    grouped[pos][nuc] += value
            return grouped
        for shap_interaction, sequence in zip(self.shapzero_results, self.sequences):
            shap_interactions_min_order = {k: v for k, v in shap_interaction.items() if len(k) >= min_order}
            for key, value in shap_interactions_min_order.items():
                for pos in key:
                    nt = sequence[pos]
                    if value > 0:
                        interaction_values_positive[(pos, nt)] = interaction_values_positive.get((pos, nt), 0) + (value / len(key))
                    elif value < 0:
                        interaction_values_negative[(pos, nt)] = interaction_values_negative.get((pos, nt), 0) + (value / len(key))
        plotting_keys = list(colors.keys())
        grouped_positive = group_by_position(interaction_values_positive, seq_length, plotting_keys)
        grouped_negative = group_by_position(interaction_values_negative, seq_length, plotting_keys)

        width_per_position = 0.3
        fig_width = max(8, min(25, seq_length * width_per_position))
        fig, ax = plt.subplots(figsize=(fig_width, 4), dpi=300)
        bottom_positive = np.zeros(seq_length)
        bottom_negative = np.zeros(seq_length)
        x_positions = np.arange(1, seq_length + 1)
        bar_width = 0.4
        for nucleotide in plotting_keys:
            positive_values = np.array([grouped_positive[pos].get(nucleotide, 0) for pos in range(seq_length)])
            negative_values = np.array([grouped_negative[pos].get(nucleotide, 0) for pos in range(seq_length)])
            ax.bar(x_positions, positive_values, bottom=bottom_positive, color=colors[nucleotide], width=bar_width, label=nucleotide)
            ax.bar(x_positions, negative_values, bottom=bottom_negative, color=colors[nucleotide], width=bar_width)
            bottom_positive += positive_values
            bottom_negative += negative_values

        ax.set_xlabel('Sequence Position')
        ax.set_ylabel(y_label)
        ax.set_xlim(0.5, seq_length + 0.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', width=0.5)
        tick_interval = 1 if seq_length <= 20 else 5
        tick_positions = [i for i in range(1, seq_length + 1) if i % tick_interval == 0]
        ax.set_xticks(tick_positions)

        if title:
            ax.set_title(title, fontweight='bold')
        max_y_val = max(bottom_positive.max(), abs(bottom_negative.min()))
        ax.set_ylim(-max_y_val * 1.1, max_y_val * 1.1)
        if legend:
            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=colors.get(nt), markersize=8, label=nt)
                for nt in colors.keys()
            ]
            ax.legend(handles=legend_handles, loc='best', ncol=len(colors))
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    
    def _plot_shap_values_proteins(self, output_path, title=None, y_label='SHAP zero value', legend=True):
        """
        Creates and saves a heatmap of the top SHAP values for protein sequences.

        Args:
            self.sequences (list): A list of sequence strings (e.g., ['ACGT']).
            self.shapzero_results (list): A list of dictionaries, where each dict contains SHAP values with position tuples as keys.
            self.encoding_map (dict): A dictionary mapping amino acids to their corresponding encodings.
            output_path (str or Path): The path where the output plot image will be saved.
            title (str, optional): The title for the plot.
            y_label (str, optional): The label for the y-axis.
            legend (bool, optional): Whether to include a legend in the plot.
        """
        shap_values = self._convert_shap_to_array()
        aa_dict = PROTEIN_ENCODING
        index_to_aa = {v: k for k, v in aa_dict.items()}
        aa_order = [aa for aa, _ in sorted(aa_dict.items(), key=lambda x: x[1])]
        num_aas = len(aa_order)
        num_sequences, seq_len = shap_values.shape

        # Compute average SHAP value for each amino acid at each position
        heatmap = np.full((num_aas, seq_len), np.nan)
        for aa_idx in range(num_aas):
            for pos in range(seq_len):
                mask = np.array([seq[pos] == index_to_aa[aa_idx] for seq in self.sequences])
                if np.any(mask):
                    heatmap[aa_idx, pos] = np.mean(shap_values[mask, pos])

        # Threshold to find top interactions and determine the number of top values to keep
        flat_vals = heatmap[~np.isnan(heatmap)]
        pos_vals = flat_vals[flat_vals > 0]
        neg_vals = flat_vals[flat_vals < 0]
        if len(self.sequences) < 30:
            percent = 100
        else:
            percent = 20
        num_pos_to_keep = max(1, int((percent / 100) * len(pos_vals)))
        num_neg_to_keep = max(1, int((percent / 100) * len(neg_vals)))
        pos_thresh = np.sort(pos_vals)[-num_pos_to_keep] if len(pos_vals) > 0 else np.inf
        neg_thresh = np.sort(neg_vals)[:num_neg_to_keep][-1] if len(neg_vals) > 0 else -np.inf
        masked_heatmap = np.full_like(heatmap, np.nan)
        keep_mask = (heatmap >= pos_thresh) | (heatmap <= neg_thresh)
        masked_heatmap[keep_mask] = heatmap[keep_mask]
        plot_heatmap = masked_heatmap.T

        fig_height = max(4, seq_len * 0.25)
        fig_width = max(8, num_aas * 0.4)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
        vmax = np.nanmax(np.abs(plot_heatmap))
        norm = Normalize(vmin=-vmax, vmax=vmax)
        im = ax.imshow(plot_heatmap, cmap='bwr', norm=norm, aspect='equal', interpolation='nearest')
        ax.set_xlabel('Amino acid')
        ax.set_ylabel('Sequence position')
        ax.set_xticks(np.arange(num_aas))
        ax.set_xticklabels(aa_order)
        ax.set_yticks(np.arange(seq_len))
        ax.set_yticklabels([str(i + 1) for i in range(seq_len)])
        ax.tick_params(axis='both', which='both', width=0.25)

        if title:
            ax.set_title(title, fontweight='bold')
        if legend:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(y_label, rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)


    def _plot_interactions_proteins(self, output_path, percent=20, title=None, y_label='SHAP zero interaction', legend=True):
        """
        Creates and saves a 20x20 heatmap of the top average pairwise SHAP interactions between amino acids.

        Args:
            self.sequences (list): A list of sequence strings (e.g., ['ACGT']).
            self.shapzero_results (list): A list of dictionaries, where each dict contains SHAP interactions with position tuples as keys.
            self.encoding_map (dict): A dictionary mapping amino acids to their corresponding encodings.
            output_path (str or Path): The path where the output plot image will be saved.
            percent (float, optional): The percent of top interactions to include in the plot. Defaults to 20.
            title (str, optional): The title for the plot.
            y_label (str, optional): The label for the y-axis.
            legend (bool, optional): Whether to include a legend in the plot.
        """
        aa_order = [aa for aa, _ in sorted(PROTEIN_ENCODING.items(), key=lambda x: x[1])]
        aa_index = {aa: i for i, aa in enumerate(aa_order)}
        n_aa = len(aa_order)
        interaction_matrix = np.zeros((n_aa, n_aa))
        count_matrix = np.zeros((n_aa, n_aa))
        for seq, interaction_dict in zip(self.sequences, self.shapzero_results):
            for positions, value in interaction_dict.items():
                if len(positions) == 2: # Only consider pairwise interactions
                    i, j = positions
                    if i < len(seq) and j < len(seq): # Bounds check
                        aa1, aa2 = seq[i].upper(), seq[j].upper()
                        if aa1 in aa_index and aa2 in aa_index:
                            idx1, idx2 = aa_index[aa1], aa_index[aa2]
                            interaction_matrix[idx1, idx2] += value
                            interaction_matrix[idx2, idx1] += value
                            count_matrix[idx1, idx2] += 1
                            count_matrix[idx2, idx1] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_matrix = np.where(count_matrix > 0, interaction_matrix / count_matrix, np.nan)
        np.fill_diagonal(avg_matrix, np.nan)

        # Determine the number of top values to keep
        if len(self.sequences) < 30:
            percent = 100
        else:
            percent = 20
        flat_vals = avg_matrix[~np.isnan(avg_matrix)]
        pos_vals = flat_vals[flat_vals > 0]
        neg_vals = flat_vals[flat_vals < 0]
        num_pos = max(1, int((percent / 100) * len(pos_vals)))
        num_neg = max(1, int((percent / 100) * len(neg_vals)))
        pos_thresh = np.sort(pos_vals)[-num_pos] if len(pos_vals) > 0 else np.inf
        neg_thresh = np.sort(neg_vals)[:num_neg][-1] if len(neg_vals) > 0 else -np.inf
        masked_matrix = np.full_like(avg_matrix, np.nan)
        keep_mask = (avg_matrix >= pos_thresh) | (avg_matrix <= neg_thresh)
        masked_matrix[keep_mask] = avg_matrix[keep_mask]

        fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
        vmax = np.nanmax(np.abs(masked_matrix))
        norm = Normalize(vmin=-vmax, vmax=vmax)
        im = ax.imshow(masked_matrix, cmap='bwr', norm=norm, aspect='equal', interpolation='nearest')
        ax.set_xticks(np.arange(n_aa))
        ax.set_xticklabels(aa_order)
        ax.set_yticks(np.arange(n_aa))
        ax.set_yticklabels(aa_order)
        ax.set_xlabel("Amino acid")
        ax.set_ylabel("Amino acid")
        ax.tick_params(axis='both', which='both', width=0.25)

        if title:
            ax.set_title(title, fontweight='bold')
        if legend:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label(y_label, rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)


    def _save_top_shap_values(self, output_folder, top_values=10):
        """
        Calculates and saves the top average SHAP values for each feature
        (e.g., 'A' at position 5) across all sequences. See Appendix E.4.

        Args:
            output_folder (str): Folder where the output csv will be saved.
            top_values (int, optional): How many values to save per positive and negative grouping.
        """
        shap_array = self._convert_shap_to_array()

        # Tally the sum and count for each (position, feature) pair
        sum_positive = {}
        count_positive = {}
        sum_negative = {}
        count_negative = {}

        for i, seq in enumerate(self.sequences):
            for pos, nt in enumerate(seq):
                value = shap_array[i, pos]
                key = (pos, nt.upper()) 
                if value > 0:
                    sum_positive[key] = sum_positive.get(key, 0) + value
                    count_positive[key] = count_positive.get(key, 0) + 1
                elif value < 0:
                    sum_negative[key] = sum_negative.get(key, 0) + value
                    count_negative[key] = count_negative.get(key, 0) + 1
        
        # Calculate the average SHAP value for each feature
        avg_positive = {k: sum_positive[k] / count_positive[k] for k in sum_positive}
        avg_negative = {k: sum_negative[k] / count_negative[k] for k in sum_negative}

        # Sort to find the top values
        top_positive = dict(sorted(avg_positive.items(), key=lambda item: item[1], reverse=True)[:top_values])
        top_negative = dict(sorted(avg_negative.items(), key=lambda item: item[1])[:top_values])
        
        # Save to CSV
        output_path = output_folder / 'top_shapzero_values.csv'
        data_for_df = []
        for (pos, feat), val in top_positive.items():
            data_for_df.append(['Positive', pos, feat, val])
        for (pos, feat), val in top_negative.items():
            data_for_df.append(['Negative', pos, feat, val])
        df = pd.DataFrame(data_for_df, columns=['Type', 'Position', 'Feature', 'Average SHAP Value'])
        df.to_csv(output_path, index=False)


    def _save_all_shap_values(self, output_folder):
        """
        Saves the raw SHAP values for every position in every sequence to a CSV file.

        Args:
            output_folder (str): Folder where the output csv will be saved.
        """
        output_path = output_folder / 'all_shapzero_values.csv'
        shap_array = self._convert_shap_to_array()
        seq_df = pd.DataFrame(self.sequences, columns=['Sequence'])
        shap_columns = [f'Position_{i+1}' for i in range(shap_array.shape[1])]
        shap_df = pd.DataFrame(shap_array, columns=shap_columns)
        final_df = pd.concat([seq_df, shap_df], axis=1)
        final_df.to_csv(output_path, index=False)


    def _save_top_interactions(self, output_folder, top_values=10, min_order=2):
        """
        Calculates and saves the top average SHAP interactions across all sequences. See Appendix E.4.

        Args:
            output_folder (str): Folder where the output csv will be saved.
            top_values (int, optional): How many values to save per positive and negative grouping.
            min_order (int, optional): The minimum order of interactions to include in the plot (only for interaction plots).
        """
        output_path = output_folder / 'top_shapzero_interactions.csv'
        pos_interactions = {}
        pos_counts = {}
        neg_interactions = {}
        neg_counts = {}

        for seq, interaction_dict in zip(self.sequences, self.shapzero_results):
            # Filter for interactions of the minimum order
            filtered_interactions = {k: v for k, v in interaction_dict.items() if len(k) >= min_order}

            for positions, value in filtered_interactions.items():
                # Create a canonical key: tuple of features, tuple of positions
                nucleotides = tuple(seq[pos].upper() for pos in positions)
                key = (nucleotides, tuple(p for p in positions))

                if value > 0:
                    pos_interactions[key] = pos_interactions.get(key, 0) + value
                    pos_counts[key] = pos_counts.get(key, 0) + 1
                elif value < 0:
                    neg_interactions[key] = neg_interactions.get(key, 0) + value
                    neg_counts[key] = neg_counts.get(key, 0) + 1

        # Calculate averages
        avg_pos = {k: v / pos_counts[k] for k, v in pos_interactions.items()}
        avg_neg = {k: v / neg_counts[k] for k, v in neg_interactions.items()}

        # Get top interactions sorted by absolute value
        top_pos = dict(sorted(avg_pos.items(), key=lambda x: abs(x[1]), reverse=True)[:top_values])
        top_neg = dict(sorted(avg_neg.items(), key=lambda x: abs(x[1]), reverse=True)[:top_values])

        # Save to CSV
        data_for_df = []
        for sign, interactions in [("Positive", top_pos), ("Negative", top_neg)]:
            for (features, positions), value in interactions.items():
                data_for_df.append([
                    sign,
                    ', '.join(map(str, positions)),
                    ', '.join(features),
                    value
                ])
        df = pd.DataFrame(data_for_df, columns=['Type', 'Positions', 'Features', 'Average SHAP Interaction'])
        df.to_csv(output_path, index=False)


    def _save_all_interactions(self, output_folder):
        """
        Saves all raw interactions for every sequence to a CSV file.
        The interactions for each sequence are stored as a JSON string in a single cell.

        Args:
            output_folder (str): Folder where the output csv will be saved.
        """
        output_path = output_folder / 'all_shapzero_interactions.csv'
        def convert_keys_to_str(obj):
            if isinstance(obj, dict):
                return {str(k): v for k, v in obj.items()}
            return obj

        data_for_df = []
        for seq, interactions in zip(self.sequences, self.shapzero_results):
            interactions_json = json.dumps(convert_keys_to_str(interactions))
            data_for_df.append([seq, interactions_json])
        df = pd.DataFrame(data_for_df, columns=['Sequence', 'Interactions'])
        df.to_csv(output_path, index=False)


def init(q, n, exp_dir, model=None, fourier_transform=None, fourier_params=None):
    """
    Initializes the SHAP zero explainer.
    """
    return Explainer(q, n, exp_dir, model, fourier_transform, fourier_params)


def load_dna_example():
    """
    Example comes from inDelphi training data (1872 sequences).
    To minimize computations, we make the training sequences be composed of 10 nucleotides around the cut site.
    Data is one-hot encoded, so X_DNA.pkl is a 2D numpy array of shape (1872, 40) and y_DNA.pkl is a 1d numpy array of shape (1872,).
    """
    file_dir = os.path.dirname(__file__)
    file_path = os.path.join(file_dir, "example_data")
    X_DNA_train = np.load(os.path.join(file_path, "X_DNA_train.pkl"), allow_pickle=True)
    y_DNA_train = np.load(os.path.join(file_path, "y_DNA_train.pkl"), allow_pickle=True)
    return X_DNA_train, y_DNA_train, 


def load_dna_sequences_to_explain():
    """
    Heldout inDelphi sequences (84 sequences).
    To minimize computations, we make the training sequences be composed of 10 nucleotides around the cut site.
    Data is a list of strings of length 84.
    """
    file_dir = os.path.dirname(__file__)
    file_path = os.path.join(file_dir, "example_data")
    DNA_seqs = np.load(os.path.join(file_path, "DNA_sequences.pkl"), allow_pickle=True)
    return DNA_seqs