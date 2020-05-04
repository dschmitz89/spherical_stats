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
    mean : float
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