import numpy as np
import torch
from typing import Tuple


def identity_initialization(shape: torch.Tensor) -> np.ndarray:
    """
    Indentity matrix initializer.

    Args:
    shape: tuple, shape of the numpy array

    Returns:
    weights: numpy array, array of random values with the given shape
    """

    # Check if the shape is valid
    if len(shape) < 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Identity matrix
    weights: np.ndarray = np.eye(int(shape[0].item()), int(shape[1].item()))

    return weights


def identity_001_initialization(shape: torch.Tensor) -> np.ndarray:
    """
    Indentity matrix with 0.01 values initializer.

    Args:
    shape: tuple, shape of the numpy array

    Returns:
    weights: numpy array, array of random values with the given shape
    """

    # Check if the shape is valid
    if len(shape) < 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Identity matrix factorized by 0.01
    weights: np.ndarray = np.eye(int(shape[0].item()), int(shape[1].item()))*0.001

    return weights


def zeros_initialization(shape: torch.Tensor) -> np.ndarray:
    """
    This function initializes a numpy array of zeros with the given shape.

    Args:
    shape: tuple, shape of the numpy array

    Returns:
    weights: numpy array, array of zeros with the given shape
    """

    # Check if the shape is valid
    if len(shape) < 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Generate matrix with zeros
    wieghts: np.ndarray = np.zeros(shape)

    return wieghts


def constant_initialization(shape: torch.Tensor, value: float = 0.5) -> np.ndarray:
    """
    This function initializes a numpy array of the given value with the given shape.

    Args:
    shape: tuple, shape of the numpy array
    value: float, value to initialize the array with

    Returns:
    weights: numpy array, array of the given value with the given shape
    """

    # Check if the shape is valid
    if len(shape) < 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Generate matrix with the given value
    weights: np.ndarray = np.full(shape, value)

    return weights


def random_normal_initialization(
        shape: torch.Tensor, mean: float = 0, std: float = 1)\
             -> np.ndarray:
    """
    Random normal initializer, with a normal distribution with the given mean and std.

    Args:
    shape: tuple, shape of the numpy array
    mean: float, mean of the normal distribution
    std: float, standard deviation of the normal distribution

    Returns:
    weights: numpy array, array of random values with the given shape
    """

    # Check if the shape is valid
    if len(shape) < 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Generate random values with a normal distribution
    weights: np.ndarray = np.random.normal(mean, std, shape)

    return weights


def random_uniform_initialization(
        shape: torch.Tensor, min_val: float = -1, max_val: float = 1)\
             -> np.ndarray:
    """
    Random uniform initializer, with a uniform distribution given min_val and max_val.

    Args:
    shape: tuple, shape of the numpy array
    min_val: float, minimum value of the uniform distribution
    max_val: float, maximum value of the uniform distribution

    Returns:
    weights: numpy array, array of random values with the given shape
    """

    # Check if the shape is valid
    if len(shape) < 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Generate random values with a uniform distribution
    weights: np.ndarray = np.random.uniform(min_val, max_val, shape)

    return weights


def truncated_normal_initialization(
        shape: torch.Tensor, mean: float = 0, std: float = 1)\
            -> np.ndarray:
    """
    Truncated normal initializer, with a normal distribution with the given mean and st.
    The values are truncated to the range of two standard deviations.

    Args:
    shape: tuple, shape of the numpy array
    mean: float, mean of the normal distribution
    std: float, standard deviation of the normal distribution

    Returns:
    weights: numpy array, array of random values with the given shape
    """

    # Check if the shape is valid
    if len(shape) < 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Generate random values with a normal distribution
    weights: np.ndarray = np.random.normal(mean, std, shape)

    # Clip the values to the range of two standard deviations
    weights = np.clip(weights, -2*std, 2*std)

    return weights


def xavier_initialization(shape: torch.Tensor) -> np.ndarray:
    """
    Xavier/Glorot initializer, with a uniform distribution scaled by
    the square root of 1 over the number of input units.

    Args:
    shape: tuple, shape of the numpy array

    Returns:
    weights: numpy array, array of random values with the given shape
    """

    # Check if the shape is valid
    if len(shape) < 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Define the limits for the uniform distribution
    min_limit: float = -np.sqrt(1 / shape[0])
    max_limit: float = np.sqrt(1 / shape[0])

    # Generate random values with a uniform distribution
    weights: np.ndarray = np.random.uniform(min_limit, max_limit, shape)

    return weights


def normalized_xavier_initialization(shape: torch.Tensor) -> np.ndarray:
    """
    Xavier/Glorot initializer, with a uniform distribution scaled by
    the square root of 6 over the sum of the number of input and output units.

    Args:
    shape: tuple, shape of the numpy array

    Returns:
    weights: numpy array, array of random values with the given shape
    """

    # Check if the shape is valid
    if len(shape) < 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Define the limits for the uniform distribution
    min_limit: float = -np.sqrt(6 / (shape[0] + shape[1]))
    max_limit:float = np.sqrt(6 / (shape[0] + shape[1]))

    # Generate random values with a uniform distribution
    weights: np.ndarray = np.random.uniform(min_limit, max_limit, shape)

    return weights


def kaiming_initialization(shape: torch.Tensor) -> np.ndarray:
    """
    Kaiming/He initializer, with a normal distribution scaled by
    the square root of 2 over the number of input units.

    Args:
    shape: tuple, shape of the numpy array

    Returns:
    weights: numpy array, array of random values with the given shape
    """

    # Check if the shape is valid
    if len(shape) < 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Define the mean and standard deviation
    mean: int = 0
    std: float = np.sqrt(2 / shape[0])

    # Generate random values with a normal distribution
    weights: np.ndarray = np.random.normal(mean, std, shape)

    return weights


def orthogonal_initialization(shape: torch.Tensor, gain: float = 1.0) -> np.ndarray:
    """
    Initializer that generates an orthogonal matrix.

    If the shape of the tensor to initialize is two-dimensional, it is
    initialized with an orthogonal matrix obtained from the QR decomposition of
    a matrix of random numbers drawn from a normal distribution.

    Args:
    shape: tuple, shape of the numpy array
    gain: float, multiplicative factor to apply to the orthogonal matrix

    Returns:
    weights: numpy array, array of random values with the given shape
    """

    # Orthogonal matrices are defined in 2D spaces or higher
    if len(shape) < 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Calculate the number of rows for the matrix by multiplying the dimensions
    num_rows: int = 1
    for dim in shape[:-1]:
        num_rows *= dim.item()
    num_cols: int  = int(shape[-1].item())

    flat_shape: Tuple[int, int] = (max(num_rows, num_cols), min(num_rows, num_cols))

    # Generate a random matrix with normal distribution
    random_matrix: np.ndarray = np.random.normal(0.0, 1.0, flat_shape)

    # Compute the qr factorization
    q, r = np.linalg.qr(random_matrix)

    # Make the diagonal elements of 'r' non-negative
    d: np.ndarray = np.diag(r, 0)
    q *= np.sign(d)

    # If the number of rows is less than the number of columns, transpose the matrix
    if num_rows < num_cols:
        q = q.T

    # Reshape the matrix to the desired shape
    with torch.no_grad():
        q_torch: torch.Tensor = torch.from_numpy(q)
        q_torch = q_torch.to(torch.float32)
        weights: np.ndarray = np.ndarray(q_torch * gain)

    return weights
