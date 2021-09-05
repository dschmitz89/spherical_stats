import numpy as np

def resultant_length(vectors):
    '''
    Calculate resultant length of a sample of vectors

    Parameters
    ----------
    vectors : ndarray [n,3]

    Returns
    -------
    R : float
    '''
    x_sum = vectors[:,0].sum()
    y_sum = vectors[:,1].sum()
    z_sum = vectors[:,2].sum()
    
    R = np.sqrt(x_sum * x_sum + y_sum * y_sum + z_sum * z_sum)
    
    return R

def spherical_mean(vectors):
    '''
    Calculate spherical mean

    Parameters
    ----------
    vectors : ndarray [n,3]

    Returns
    -------
    mean : ndarray [n,3]
    '''
    r = resultant_length(vectors)
        
    mean = 1/r*np.array([vectors[:,0].sum(), vectors[:,1].sum(), \
                         vectors[:,2].sum()])
    
    return mean

def spherical_variance(vectors):
    '''
    Calculate spherical variance

    Parameters
    ----------
    vectors : ndarray [n,3]

    Returns
    -------
    var : float
    '''
    n_samples = vectors.shape[0]
    
    r = resultant_length(vectors)

    var = 1 - r/n_samples

    return var

def orientation_matrix(vectors, eigen = True):
    '''
    Calculates the orientation matrix and its Eigen decomposition
    of a set of vectors

    Parameters
    ----------
    vectors : ndarray [n,3]
        sample of n vectors
    eigen : bool, optional
        indicate that Eigen decomposition shall be performed. 
        The default is True.

    Returns
    -------
    orientation_matrix
    ndarray [3,3,]
    
    or
    
    eigenvals: ndarray[3]
    eigenvecs: ndarray[3,3]

    '''
    x_vals = vectors[:, 0]
    y_vals = vectors[:, 1]
    z_vals = vectors[:, 2]
    
    orientation_matrix = np.zeros((3,3))
    
    orientation_matrix[0, 0] = np.sum(x_vals * x_vals)
    xysum = np.sum(x_vals * y_vals)
    orientation_matrix[0, 1] = xysum
    orientation_matrix[1, 0] = xysum
    xzsum = np.sum(x_vals * z_vals)
    orientation_matrix[0, 2] = xzsum
    orientation_matrix[2, 0] = xzsum
    orientation_matrix[1, 1] = np.sum(y_vals * y_vals)
    yzsum = np.sum(y_vals * z_vals)
    orientation_matrix[1, 2] = yzsum
    orientation_matrix[2, 1] = yzsum
    orientation_matrix[2, 2] = np.sum(z_vals * z_vals)
    
    if eigen == True:
        
        eigenvals, eigenvecs = np.linalg.eigh(orientation_matrix)
        
        n_samples = vectors.shape[0]
        
        eigenvals = eigenvals/n_samples
        
        return eigenvals, eigenvecs
    
    else:
        
        return orientation_matrix    