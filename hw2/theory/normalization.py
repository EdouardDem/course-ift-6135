import numpy as np

def batch_normalization(x):
    """
    Batch Normalization
    Normalise chaque channel à travers le batch et les features
    """
    # Calcul de la moyenne et de la variance sur les dimensions batch et features
    mean = np.mean(x, axis=(0, 2), keepdims=True)
    var = np.var(x, axis=(0, 2), keepdims=True)
    
    # Normalisation
    x_norm = (x - mean) / np.sqrt(var)
    return x_norm, mean, var

def layer_normalization(x):
    """
    Layer Normalization
    Normalise chaque échantillon à travers les channels et les features
    """
    # Calcul de la moyenne et de la variance sur les dimensions channels et features
    mean = np.mean(x, axis=(1, 2), keepdims=True)
    var = np.var(x, axis=(1, 2), keepdims=True)
    
    # Normalisation
    x_norm = (x - mean) / np.sqrt(var)
    return x_norm, mean, var

def instance_normalization(x):
    """
    Instance Normalization
    Normalise chaque feature à travers les channels pour chaque échantillon individuellement
    """
    # Calcul de la moyenne et de la variance sur la dimension channels pour chaque échantillon
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True) 
    
    # Normalisation
    x_norm = (x - mean) / np.sqrt(var)
    return x_norm, mean, var    


if __name__ == "__main__":
    # Création d'un tensor d'exemple avec shape (4, 2, 3)
    x = np.array([
        [[2,5,1], [4,3,4]],
        [[1,2,4], [1,2,2]],
        [[3,1,2], [3,2,4]],
        [[3,4,3], [5,2,3]],
    ])

    # Application des différentes normalisations
    x_bn, mean_bn, var_bn = batch_normalization(x)
    x_ln, mean_ln, var_ln = layer_normalization(x)
    x_in, mean_in, var_in = instance_normalization(x)

    print("Tensor d'entrée shape:", x.shape)
    print("--------------------------------")
    print("Batch Normalization shape:", x_bn.shape)
    print("Batch Normalization mean shape:", mean_bn.shape)
    print("Batch Normalization mean:", mean_bn)
    print("Batch Normalization var shape:", var_bn.shape)
    print("Batch Normalization var:", var_bn)
    print("--------------------------------")
    print("Layer Normalization shape:", x_ln.shape)
    print("Layer Normalization mean shape:", mean_ln.shape)
    print("Layer Normalization mean:", mean_ln)
    print("Layer Normalization var shape:", var_ln.shape)
    print("Layer Normalization var:", var_ln)
    print("--------------------------------")
    print("Instance Normalization shape:", x_in.shape) 
    print("Instance Normalization mean shape:", mean_in.shape)
    print("Instance Normalization mean:", mean_in)
    print("Instance Normalization var shape:", var_in.shape)
    print("Instance Normalization var:", var_in)