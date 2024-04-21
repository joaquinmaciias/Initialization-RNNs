import numpy as np
import torch


def identity_initialization(weight: torch.Tensor):
    """
    Indentity matrix initializer.

    Args:
    weight: torch.Tensor to initialize
    """

    # Check if the shape is valid
    if weight.ndim != 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Identity matrix
    with torch.no_grad():
        identity_matrix = np.eye(weight.size(0), weight.size(1))
        weight.copy_(torch.from_numpy(identity_matrix).to(weight.device))



def identity_001_initialization(weight: torch.Tensor):
    """
    Indentity matrix with 0.01 values initializer.

    Args:
    weight: torch.Tensor to initialize
    """

    # Check if the shape is valid
    if weight.ndim != 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Identity matrix factorized by 0.01
    with torch.no_grad():
        identity_matrix = np.eye(weight.size(0), weight.size(1)) * 0.01
        weight.copy_(torch.from_numpy(identity_matrix).to(weight.device))


def zeros_initialization(weight: torch.Tensor):
    """
    This function initializes a numpy array of zeros with the given shape.

    Args:
    weight: torch.Tensor to initialize
    """

    # Check if the shape is valid
    if weight.ndim != 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Generate matrix with zeros
    with torch.no_grad():
        weight.zero_()


def constant_initialization(weight: torch.Tensor, value: float = 0.5):
    """
    This function initializes a numpy array of the given value with the given shape.

    Args:
    weight: torch.Tensor to initialize
    value: float, value to initialize the array with
    """

    # Check if the shape is valid
    if weight.ndim != 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Generate matrix with the given value
    with torch.no_grad():
        weight.fill_(value)


def random_normal_initialization(
        weight: torch.Tensor, mean: float = 0, std: float = 1)\
            :
    """
    Random normal initializer, with a normal distribution with the given mean and std.

    Args:
    weight: torch.Tensor to initialize
    mean: float, mean of the normal distribution
    std: float, standard deviation of the normal distribution

    """

    # Check if the shape is valid
    if weight.ndim != 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Generate random values with a normal distribution
    with torch.no_grad():
        normal_values = np.random.normal(mean, std, weight.shape)
        weight.copy_(torch.from_numpy(normal_values).to(weight.device))


def random_uniform_initialization(
        weight: torch.Tensor, min_val: float = -1, max_val: float = 1)\
            :
    """
    Random uniform initializer, with a uniform distribution given min_val and max_val.

    Args:
    weight: torch.Tensor to initialize
    min_val: float, minimum value of the uniform distribution
    max_val: float, maximum value of the uniform distribution
    """

    # Check if the shape is valid
    if weight.ndim != 2:
        raise ValueError("Only shapes of length 2 or more are supported.")

    # Generate random values with a uniform distribution
    with torch.no_grad():
        uniform_values = np.random.uniform(min_val, max_val, weight.shape)
        weight.copy_(torch.from_numpy(uniform_values).to(weight.device))


def truncated_normal_initialization(
        weight: torch.Tensor, mean: float = 0, std: float = 1)\
        :
    """
    Truncated normal initializer, with a normal distribution with the given mean and st.
    The values are truncated to the range of two standard deviations.

    Args:
    weight: torch.Tensor to initialize
    mean: float, mean of the normal distribution
    std: float, standard deviation of the normal distribution
    """

    # Check if the shape is valid
    if weight.ndim != 2:
        raise ValueError("Only shapes of length 2 or more are supported.")


    with torch.no_grad():
        # Generate random values with a normal distribution
        normal_values = np.random.normal(mean, std, weight.shape)

        # Clip the values to the range of two standard deviations
        truncated_values = np.clip(normal_values, mean - 2 * std, mean + 2 * std)

        # Copy the truncated normal values to the tensor
        weight.copy_(torch.from_numpy(truncated_values).to(weight.device))


def xavier_initialization(weight: torch.Tensor):
    """
    Xavier/Glorot initializer, with a uniform distribution scaled by
    the square root of 1 over the number of input units.

    Args:
    weight: torch.Tensor to initialize
    """

    # Check if the shape is valid
    if weight.ndim != 2:
        raise ValueError("Only shapes of length 2 or more are supported.")
    
    # Calculate the n_in (number of input units)
    n_in = weight.size(0) 

    # Define the limits for the Xavier/Glorot uniform distribution
    limit = np.sqrt(1 / n_in)
    min_limit = -limit
    max_limit = limit

    # Generate random values with a uniform distribution
    with torch.no_grad():
        uniform_values = np.random.uniform(min_limit, max_limit, weight.shape)
        weight.copy_(torch.from_numpy(uniform_values).to(weight.device))


def normalized_xavier_initialization(weight: torch.Tensor):
    """
    Xavier/Glorot initializer, with a uniform distribution scaled by
    the square root of 6 over the sum of the number of input and output units.

    Args:
    weight: torch.Tensor to initialize
    """

    # Check if the shape is valid
    if weight.ndim != 2:
        raise ValueError("Only shapes of length 2 or more are supported.")
    
    # Calculate the n_in (number of input units) and n_out (number of output units)
    n_in = weight.size(0)
    n_out = weight.size(1)

    # Define the limits for the Normalized Xavier/Glorot uniform distribution
    limit = np.sqrt(6 / (n_in + n_out))
    min_limit = -limit
    max_limit = limit

    # Generate random values with a uniform distribution
    with torch.no_grad():
        uniform_values = np.random.uniform(min_limit, max_limit, weight.shape)
        weight.copy_(torch.from_numpy(uniform_values).to(weight.device))


def kaiming_initialization(weight: torch.Tensor):
    """
    Kaiming/He initializer, with a normal distribution scaled by
    the square root of 2 over the number of input units.

    Args:
    weight: torch.Tensor to initialize
    """

    # Check if the shape is valid
    if weight.ndim != 2:
        raise ValueError("Only shapes of length 2 or more are supported.")
    
    # Calculate the n_in (number of input units)
    n_in = weight.size(0)

    # Define the mean and standard deviation
    mean = 0
    std = np.sqrt(2 / n_in)

    # Generate random values with a normal distribution
    with torch.no_grad():
        normal_values = np.random.normal(mean, std, weight.shape)
        weight.copy_(torch.from_numpy(normal_values).to(weight.device))


def orthogonal_initialization(weight: torch.Tensor, gain: float = 1.0):
    """
    Initializer that generates an orthogonal matrix.

    If the shape of the tensor to initialize is two-dimensional, it is
    initialized with an orthogonal matrix obtained from the QR decomposition of
    a matrix of random numbers drawn from a normal distribution.

    Args:
    weight: torch.Tensor, the tensor to initialize
    gain: float, multiplicative factor to apply to the orthogonal matrix
    """

    # Orthogonal matrices are defined in 2D spaces or higher
    if weight.ndim != 2:
        raise ValueError("Only shapes of length 2 or more are supported.")
    
    # Generate a random matrix with normal distribution
    random_matrix = np.random.normal(0.0, 1.0, (max(weight.size(0), weight.size(1)), min(weight.size(0), weight.size(1))))

    # Compute the QR factorization
    q, r = np.linalg.qr(random_matrix)

    # Make the diagonal elements of 'r' non-negative
    q *= np.sign(np.diag(r))

    # Ensure the orthogonal matrix is the right shape
    if weight.size(0) < weight.size(1):
        q = q.T[:weight.size(0), :weight.size(1)]
    else:
        q = q[:weight.size(0), :weight.size(1)]

    # Copy the orthogonal matrix to the tensor with the gain
    with torch.no_grad():
        weight.copy_(torch.from_numpy(q * gain).to(weight.device))