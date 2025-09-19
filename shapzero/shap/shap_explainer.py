"""
shap_explainer class to convert the Fourier transform into the Mobius transform, and then the Mobius transform into SHAP values and interactions.
"""
import pickle
import cmath
from itertools import combinations, product
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class shap_explainer:
    def __init__(self, transform, q, n):
        self.transform = transform
        self.q = q
        self.n = n

    def degree(self, index):
        """
        Counts the number of non-zero indices in a given index vector.
        
        Args:
            index (np.ndarray): A 1D numpy array of integers.

        Returns:
            int: The number of non-zero indices in the input index vector.
        """
        count = 0
        coordinates = []
        coordinate_values = []
        for i in range(len(index)):
            if index[i] != 0:
                count += 1
                coordinates.append(i)
                coordinate_values.append(index[i])
        return count, coordinates, coordinate_values


    def get_permutations(self, coordinates):
        """
        Get all possible combinations of coordinates up to length of input coordinates.

        Args:
            coordinates (list): List of coordinate indices to generate combinations from.

        Returns:
            list: List of tuples containing all possible combinations of coordinates.
        """
        result = []
        for r in range(1, len(coordinates) + 1):
            perms = combinations(coordinates, r)
            result.extend(set(perms))
        return result

    def get_permutations_of_length_n(self, tuple_elements, q=None):
        """
        Get all possible q-ary permutations of a given tuple of elements.

        Args:
            tuple_elements (tuple): Tuple of coordinate indices to generate permutations for.
            q (int, optional): Base of q-ary system. If not provided, uses self.q.

        Returns:
            list: List of tuples containing all possible q-ary permutations of the input coordinates.
                 Each element in the permutations is in range [1, q-1].
        """
        if q is not None:
            perms = list(product(range(1, q), repeat=len(tuple_elements)))
        else:
            perms = list(product(range(1, self.q), repeat=len(tuple_elements)))
        return perms

    def get_permutations_of_lower_orders(self, coordinates, order):
        """
        Get all possible combinations of coordinates up to a specified order.

        Args:
            coordinates (list): List of coordinate indices to generate combinations from.
            order (int): Maximum order of combinations to generate.

        Returns:
            list: List of tuples containing all possible combinations of coordinates up to the specified order.
                 Each tuple represents a subset of the input coordinates.
        """
        result = []
        for r in range(1, order):
            perms = combinations(coordinates, r)
            result.extend(set(perms))
        return result


    def get_permutations_of_order(self, coordinates, order):
        """
        Get all possible combinations of coordinates of a specific order.

        Args:
            coordinates (list): List of coordinate indices to generate combinations from.
            order (int): The exact order of combinations to generate.

        Returns:
            list: List of tuples containing all possible combinations of coordinates of the specified order.
                 Each tuple represents a subset of the input coordinates of length equal to order.
        """
        result = []
        perms = combinations(coordinates, order)
        result.extend(set(perms))
        return result


    def run_mobius_transform(self):
        """
        Run the Mobius transform on the q-ary Fourier transform.
        
        The Mobius transform converts the Fourier coefficients into localized interaction effects.
        It iteratively builds up interaction terms by combining lower-order terms.
        
        The transform is stored in self.mobius_tf, which maps tuples of q-ary indices to complex values.
        The keys are tuples of length n (number of features) where non-zero elements indicate interactions.
        
        Returns:
            None
        
        Side Effects:
            Sets self.mobius_tf with the computed Mobius transform
        """
        omega = cmath.exp(2 * cmath.pi * 1j / self.q)
        data = self.transform

        # Sort transform by degree
        data = dict(sorted(data.items(), key=lambda x: sum(1 for i in x[0] if i != 0)))

        zeros = [0 for _ in range(self.n)]
        all_mobius_tfs = []

        for key, value in zip(data.keys(), data.values()):
            mobius_tf_perkey = dict()
            deg, coordinates, coordinate_vals = self.degree(key)

            # Update the 0th term
            mobius_tf_perkey[tuple(zeros)] = mobius_tf_perkey.get(tuple(zeros), 0) + value

            # Update all other interactions
            if deg > 0:
                all_interactions = self.get_permutations(coordinates)
                for interaction in all_interactions:
                    curr_key = zeros.copy()
                    mobius_nonzeros = self.get_permutations_of_length_n(interaction)
                    for mobius_nonzero in mobius_nonzeros:
                        for coord, iter_val in zip(interaction, mobius_nonzero):
                            curr_key[coord] = iter_val
                        k_i = key[coord]
                        iterative_key = curr_key.copy()
                        y_i = iterative_key[coord]
                        iterative_key[coord] = 0
                        mobius_tf_perkey[tuple(curr_key)] = mobius_tf_perkey.get(tuple(iterative_key), 0) * (omega ** (k_i * y_i) - 1)

            all_mobius_tfs.append(mobius_tf_perkey)

        # Aggregate all terms
        self.mobius_tf = {}
        for d in all_mobius_tfs:
            for key, value in d.items():
                self.mobius_tf[key] = self.mobius_tf.get(key, 0) + value


    def get_mobius_tf(self):
        return self.mobius_tf
    

    def localize_sample(self, sample):
        """
        Given a q-ary Fourier transform, run the Mobius transform such that the transform is localized to the sample (aliasing property from Equation (21) in SHAP zero).
        
        Args:
            sample: numpy array
                The sample to localize the transform around. Each element should be in [0, q-1].
        
        Returns:
            localized_mobius_tf: dict
                The localized Mobius transform, where the all-zeros vector represents the sample without mutations. The other entries represent 
                arbitrary numbers that can be converted back to the original encodings.
                Keys are tuples representing positions, values are the transform coefficients.
            
            localized_mobius_tf_encoding: dict 
                The localized Mobius transform with keys expressed in terms of the original q-ary encodings.
                Keys are tuples representing positions, values are the transform coefficients.
        """
        omega = np.exp(2j * np.pi / self.q)
        w_d_k = omega ** (sample @ np.array(list(self.transform.keys())).T)    
        F_k = np.multiply(list(self.transform.values()), w_d_k)
        delayed_transform = dict(zip(list(self.transform.keys()), F_k))
        mobius_transform = shap_explainer(delayed_transform, q=self.q, n=self.n)
        mobius_transform.run_mobius_transform()
        localized_mobius_tf = mobius_transform.get_mobius_tf()

        # Convert the localized mobius_tf to represent the original encodings
        localized_mobius_tf_encoding = {}
        for key, value in localized_mobius_tf.items():
            delayed_key = tuple((key + np.array(sample)) % self.q)
            localized_mobius_tf_encoding[delayed_key] = value

        return localized_mobius_tf, localized_mobius_tf_encoding
    

    def explain(self, sample, explanation='shap_value'):
        """
        Explain a sample using the sparse mobius transform via shap values or shap interactions
        Returns a list of dictionaries containing the contributions per position
        """
        if explanation == 'shap_value':
            interactions = self.explain_shap_value(sample)
        elif explanation == 'interaction':
            interactions = self.explain_faith_shap_interaction(sample)
        else:
            raise ValueError(f"Explanation method {explanation} not supported")

        return interactions


    def compute_shap_value(self, localized_mobius_tf):
        """
        Given a localized mobius transform, compute the weighted average contribution of mobius coefficients for each position
        """
        interactions = {}
        for key, value in localized_mobius_tf.items():
            mobius_nonzeros = np.nonzero(key)[0]
            if mobius_nonzeros.size: # Check that the key is not all zeros
                interaction_order = mobius_nonzeros.size
                for nonzero in mobius_nonzeros:
                    interactions[tuple([nonzero])] = interactions.get(tuple([nonzero]), 0) - (1/ (interaction_order * self.q**(interaction_order)) ) * np.real(value)

        return interactions
    

    def compute_faith_shap_interactions(self, localized_mobius_tf):
        """
        Given a localized mobius transform, compute the shap interactions for each position
        """
        interactions = {}
        for key, value in localized_mobius_tf.items():
            mobius_nonzeros = np.nonzero(key)[0]
            if mobius_nonzeros.size: # Check that the key is not all zeros
                k = mobius_nonzeros.size
                interaction_permutations = self.get_permutations(mobius_nonzeros) # get all permutations of mobius interactions
                for interaction in interaction_permutations:
                    interaction_order = len(interaction)
                    interactions[tuple(interaction)] = interactions.get(tuple(interaction), 0) + ((-1) ** interaction_order) * (1/ ( self.q**k) ) * np.real(value)

        return interactions
    

    def explain_shap_value(self, sample):
        """
        Given a sample, compute SHAP values using the mobius transform
        """
        localized_mobius_tf, _ = self.localize_sample(sample)
        interactions = self.compute_shap_value(localized_mobius_tf)

        return interactions
    

    def explain_faith_shap_interaction(self, sample):
        """
        Given a sample, compute Faith-Shap interactions using the mobius transform
        """
        localized_mobius_tf, _ = self.localize_sample(sample)
        interactions = self.compute_faith_shap_interactions(localized_mobius_tf)

        return interactions
    
    
    def compute_scores(self, x, mean=0):
        """
        Compute the scores for samples using the mobius transform
        Input: Encoded q-ary sample. Mean value of the sample (if not zero)
        Output: Score given the mobius transform
        """
        xM_value = 0
        zeros = [0 for _ in range(self.n)]

        for key in self.mobius_tf.keys():
            mobius_nonzeros = np.nonzero(key)[0]

            # Check if mobius should be added
            if mobius_nonzeros.size:
                matches = True
                for coord in mobius_nonzeros:
                    if x[coord] != key[coord]:
                        matches = False
                        break
                if matches: 
                    xM_value += self.mobius_tf.get(tuple(key))

        xM_value += self.mobius_tf.get(tuple(zeros), 0) 
        xM_value += mean

        return xM_value