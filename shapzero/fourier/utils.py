'''
Utility functions.
'''
import numpy as np
from scipy.fftpack import fft
import numpy.fft as fft
import itertools
import math
import random
import time
from scipy.spatial import ConvexHull
import zlib
import pickle
import json
import matplotlib.pyplot as plt

from itertools import product

def ft_matrix(q, n, banned_indices = {}):
    '''
    Generate matrix for Fourier Transform, optional banned indices arg to ban certain spots.
    q = signal alphabet size
    n = signal length
    banned_indices = indices to ban, formatted as the index to be banned at with the values in a list
    (ex: if we want to ban the numbers 2 and 3 at the first index, we would have {0:[2,3]})
    '''
    all_pairs = list(product(range(q), repeat=n))
    qs = np.repeat(q, n)
    for key, value in banned_indices.items():
        qs[key] -= len(value) 
    filtered_pairs = [
        pair for pair in all_pairs
        if all(pair[key] not in values for key, values in banned_indices.items())
    ]
    omegas = []
    for q in qs:
        omegas.append(np.exp(2j * np.pi/q))
    matrix = np.empty((len(filtered_pairs), len(filtered_pairs)), dtype=complex)
    for row_index, rowpair in enumerate(filtered_pairs):
        for col_index, colpair in enumerate(filtered_pairs):
            hadamard_array = np.multiply(rowpair, colpair)
            entry = np.prod([omegas[i] ** hadamard_array[i] for i in range(n)])
            matrix[row_index, col_index] = entry
    return matrix

def tft(x, q, n, banned_indices={}, normalize=False):
    '''
    Returns truncated fourier transform of input signal with banned indices. Optional 'normalize'
    argument normalizes the fourier transform by the square root of the length of the signal q^n.
    '''
    normalization_factor = np.sqrt(q**n)
    W_inv = np.linalg.inv(ft_matrix(q, n, banned_indices))
    result = np.dot(W_inv, x)
    if normalize:
        result /= normalization_factor
    return result

def get_qs(q_max, n, banned_indices={}):
    '''
    Returns the alphabet size (qs) for each site (n) in the sequence. 
    Returns length n vector of only qs if banned_indices arg is left blank.
    '''
    qs = np.repeat(q_max, n)
    for key, value in banned_indices.items():
        qs[key] -= len(value) 
    return qs

def fwht(x):
    """Recursive implementation of the 1D Cooley-Tukey FFT"""
    # x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N == 1:
        return x
    else:
        X_even = fwht(x[0:(N//2)])
        X_odd = fwht(x[(N//2):])
        return np.concatenate([(X_even + X_odd),
                               (X_even - X_odd)])


def gwht(x,q,n):
    """Computes the GWHT of an input signal with forward scaling"""
    x_tensor = np.reshape(x, [q] * n)
    x_tf = fft.fftn(x_tensor) / (q ** n)
    x_tf = np.reshape(x_tf, [q ** n])
    return x_tf

def gwht_tensored(x,q,n):
    """Computes the GWHT of an input signal with forward scaling"""
    x_tf = fft.fftn(x) / (q ** n)
    return x_tf

def igwht(x,q,n):
    """Computes the IGWHT of an input signal with forward scaling"""
    x_tensor = np.reshape(x, [q] * n)
    x_tf = fft.ifftn(x_tensor) * (q ** n)
    x_tf = np.reshape(x_tf, [q ** n])
    return x_tf

def igwht_tensored(x,q,n):
    """Computes the IGWHT of an input signal with forward scaling"""
    x_tf = fft.ifftn(x) * (q ** n)
    return x_tf

def bin_to_dec(x):
    n = len(x)
    c = 2**(np.arange(n)[::-1])
    return c.dot(x).astype(np.int)


def nth_roots_unity(n):
    return np.exp(-2j * np.pi / n * np.arange(n))


def near_nth_roots(ratios, q, eps):
    in_set = np.zeros(ratios.shape, dtype=bool)
    omega = nth_roots_unity(q)
    for i in range(q):
        in_set = in_set | (np.square(np.abs(ratios - omega[i])) < eps)
    is_singleton = in_set.all()
    return is_singleton


def qary_vec_to_dec(x, q):
    n = x.shape[0]
    return np.array([q ** (n - (i + 1)) for i in range(n)], dtype=object) @ np.array(x,  dtype=object)


def dec_to_qary_vec(x, q, n):
    qary_vec = []
    for i in range(n):
        qary_vec.append(np.array([a // (q ** (n - (i + 1))) for a in x], dtype=object))
        x = x - (q ** (n-(i + 1))) * qary_vec[i]
    return np.array(qary_vec, dtype=int)


def dec_to_bin(x, num_bits):
    assert x < 2**num_bits, "number of bits are not enough"
    u = bin(x)[2:].zfill(num_bits)
    u = list(u)
    u = [int(i) for i in u]
    return np.array(u)


def binary_ints(m):
    '''
    Returns a matrix where row 'i' is dec_to_bin(i, m), for i from 0 to 2 ** m - 1.
    From https://stackoverflow.com/questions/28111051/create-a-matrix-of-binary-representation-of-numbers-in-python.
    '''
    a = np.arange(2 ** m, dtype=int)[np.newaxis,:]
    b = np.arange(m, dtype=int)[::-1,np.newaxis]
    return np.array(a & 2**b > 0, dtype=int)

def angle_q(x,q):
    return (((np.angle(x) % (2*np.pi) // (np.pi/q)) + 1) // 2) % q # Can be made much faster

def qary_ints(m, q, dtype=int):
    return np.array(list(itertools.product(np.arange(q), repeat=m)), dtype=dtype).T

def comb(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

def qary_ints_low_order(m, q, order):
    num_of_ks = np.sum([comb(m, o) * ((q-1) ** o) for o in range(order + 1)])
    K = np.zeros((num_of_ks, m))
    counter = 0
    for o in range(order + 1):
        positions = itertools.combinations(np.arange(m), o)
        for pos in positions:
            K[counter:counter+((q-1) ** o), pos] = np.array(list(itertools.product(1 + np.arange(q-1), repeat=o)))
            counter += ((q-1) ** o)
    return K.T

def base_ints(q, m):
    '''
    Returns a matrix where row 'i' is the base-q representation of i, for i from 0 to q ** m - 1.
    Covers the functionality of binary_ints when n = 2, but binary_ints is faster for that case.
    '''
    get_row = lambda i: np.array([int(j) for j in np.base_repr(i, base=q).zfill(m)])
    return np.vstack((get_row(i) for i in range(q ** m)))

def polymod(p1, p2, q, m):
    '''
    Computes p1 modulo p2, and takes the coefficients modulo q.
    '''
    p1 = np.trim_zeros(p1, trim='f')
    p2 = np.trim_zeros(p2, trim='f')
    while len(p1) >= len(p2) and len(p1) > 0:
        p1 -= p1[0] // p2[0] * np.pad(p2, (0, len(p1) - len(p2)))
        p1 = np.trim_zeros(p1, trim='f')
    return np.pad(np.mod(p1, q), (m + 1 - len(p1), 0))

def rref(A, b, q):
    '''
    Row reduction, to easily solve finite field systems.
    '''
    raise NotImplementedError()

def sign(x):
    '''
    Replacement for np.sign that matches the convention (footnote 2 on page 11).
    '''
    return (1 - np.sign(x)) // 2

def flip(x):
    '''
    Flip all bits in the binary array x.
    '''
    return np.bitwise_xor(x, 1)

def random_signal_strength_model(sparsity, a, b):
    magnitude = np.random.uniform(a, b, sparsity)
    phase = np.random.uniform(0, 2*np.pi, sparsity)
    return magnitude * np.exp(1j*phase)


def best_convex_underestimator(points):
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    first_point_idx = np.argmin(vertices[:, 0])
    last_point_idx = np.argmax(vertices[:, 0])

    if last_point_idx == vertices.shape[0]:
        return vertices[first_point_idx:]
    if first_point_idx < last_point_idx:
        return vertices[first_point_idx:last_point_idx+1]
    else:
        return np.concatenate((vertices[first_point_idx:], vertices[:last_point_idx+1]))

def sort_qary_vecs(qary_vecs):
    qary_vecs = np.array(qary_vecs)
    idx = np.lexsort(qary_vecs.T[::-1, :])
    return qary_vecs[idx]

def calc_hamming_weight(qary_vecs):
    qary_vecs = np.array(qary_vecs)
    return np.sum(qary_vecs != 0, axis = 1)

def save_data(data, filename):
    with open(filename, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(data, pickle.HIGHEST_PROTOCOL), 9))

def load_data(filename):
    start = time.time()
    with open(filename, 'rb') as f:
        data = pickle.loads(zlib.decompress(f.read()))
    return data

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def count_interactions(locations):
    nonzero_counts = {}
    for row in locations:
        nonzero_indices = np.nonzero(row)[0]
        num_nonzero_indices = len(nonzero_indices)
        nonzero_counts[num_nonzero_indices] = nonzero_counts.get(num_nonzero_indices, 0) + 1
    
    for num_nonzero_indices, count in nonzero_counts.items():
        print("There are {} {}-order interactions.".format(count, num_nonzero_indices))
    
    return nonzero_counts

def calculate_fourier_magnitudes(locations, gwht):
    nonzero_counts = count_interactions(locations)
    k_values = sorted(nonzero_counts.keys())

    # Handle the case where no interactions were found
    if not k_values:
        print("Warning: No interactions found to calculate magnitudes.")
        return {} # Return an empty dictionary

    j = 0 if 0 in k_values else 1
    F_k_values = np.zeros(max(np.max(k_values)+1, len(k_values)))

    for row in locations:
        nonzero_indices = np.nonzero(row)[0]
        num_nonzero_indices = len(nonzero_indices)
        F_k_values[num_nonzero_indices-j] += np.abs(gwht[row])
    
    F_k_values = np.square(F_k_values)
    return dict(zip(k_values, F_k_values))

def plot_interaction_magnitudes(sum_squares, q, n, b, output_folder, args):
    index_counts = list(sum_squares.keys())
    values = list(sum_squares.values())
    plt.figure()
    plt.bar(index_counts, values, align='center', color='limegreen')
    plt.xlabel('$r^{th}$ order interactions')
    plt.ylabel('Magnitude of Fourier coefficients')
    plt.xticks(index_counts)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('q{}_n{}_b{}'.format(q, n, b))
    
    if args.get('param'):
        param_path = args.param.replace('/', '_')
        file_path = 'magnitude_of_interactions_{}.png'.format(param_path)
    else:
        file_path = 'magnitude_of_interactions.png'
    plt.savefig(output_folder / file_path)
    plt.close()

def write_results_to_file(results_file, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight):
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as file:
        file.write("q = {}, n = {}, b = {}, noise_sd = {}\n".format(q, n, b, noise_sd))
        file.write("\nTotal samples = {}\n".format(n_used))
        file.write("Total sample ratio = {}\n".format(n_used / q ** n))
        file.write("R^2 = {}\n".format(r2_value))
        file.write("NMSE = {}\n".format(nmse))
        file.write("AVG Hamming Weight of Nonzero Locations = {}\n".format(avg_hamming_weight))
        file.write("Max Hamming Weight of Nonzero Locations = {}\n".format(max_hamming_weight))
        
def summarize_results(locations, gwht, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight, folder, args):

    sum_squares = calculate_fourier_magnitudes(locations, gwht)
    plot_interaction_magnitudes(sum_squares, q, n, b, folder, args)

    if args.get('param'):
        param_path = args.param.replace('/', '_')
        file_path = 'helper_results_{}.txt'.format(param_path)
    else:
        file_path = 'helper_results.txt'

    results_file = folder / file_path
    write_results_to_file(results_file, q, n, b, noise_sd, n_used, r2_value, nmse, avg_hamming_weight, max_hamming_weight)

def find_number_of_samples_used(q, n, b, num_subsample, num_repeat):
    num_samples = (q ** b) * num_subsample * (num_repeat * (n + 1))
    return num_samples