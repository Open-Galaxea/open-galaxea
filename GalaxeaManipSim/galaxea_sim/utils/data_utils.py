import h5py
import json
import numpy as np
import os

def recursive_create_dict_list(d):
    if isinstance(d, dict):
        return {k: recursive_create_dict_list(v) for k, v in d.items()}
    else:
        return []
    
def recursive_append_dict_list(f, d, data):
    if isinstance(data, dict):
        for k, v in data.items():
            recursive_append_dict_list(f, d[k], v)
    else:
        if isinstance(data, str):
            # d.append(data)
            # data to ASCII
            d.append(data.encode('ascii'))
        else:
            d.append(data)
        
def recursive_convert_dict_list_to_numpy(d):
    if isinstance(d, dict):
        return {k: recursive_convert_dict_list_to_numpy(v) for k, v in d.items()}
    else:
        return np.array(d)

def recursive_create_dict_list_to_hdf5(f, d, key=""):
    if isinstance(d, dict):
        for k, v in d.items():
            recursive_create_dict_list_to_hdf5(f, v, key + "/" + k)
    else:
        f.create_dataset(key, data=d, compression="gzip")

def save_dict_list_to_hdf5(dict_list, file_path):
    # overwrite the file
    if os.path.exists(file_path):
        os.remove(file_path)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with h5py.File(file_path, 'w') as f:
        output_dict = recursive_create_dict_list(dict_list[0])
        for i, d in enumerate(dict_list):
            recursive_append_dict_list(f, output_dict, d)
        output_dict_numpy = recursive_convert_dict_list_to_numpy(output_dict)
        recursive_create_dict_list_to_hdf5(f, output_dict_numpy, "")
        
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, o):
        # Check for NumPy integer types
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(o)
            
        # Check for NumPy float types
        elif isinstance(o, (np.float16, np.float32, np.float64)):
            return float(o)
        
        # ADD THIS BLOCK to handle NumPy booleans
        elif isinstance(o, np.bool_):
            return bool(o)
            
        # Check for NumPy arrays
        elif isinstance(o, np.ndarray):
            return o.tolist()
            
        return super(NumpyEncoder, self).default(o)
        
def save_dict_list_to_json(dict_list, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(dict_list, f, cls=NumpyEncoder, indent=4)