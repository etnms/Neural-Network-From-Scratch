import numpy as np

# Helper function to convert numpy array to JSON
def convert_to_python_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def convert_to_numpy_arrays(obj):
    if isinstance(obj, list):
        return np.array(obj)
    return obj