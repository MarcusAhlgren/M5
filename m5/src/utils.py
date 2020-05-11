import numpy as np
import pandas as pd

def get_memory_usage(data):
    if isinstance(data, pd.DataFrame):
        memory = data.memory_usage().sum() / 1024 ** 2
    elif isinstance(data, pd.Series):
        memory = data.memory_usage() / 1024 ** 2
    else:
        return None
    return memory 
    
def reduce_memory_usage(data, verbose = True):
    df = data.copy()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = get_memory_usage(df)    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = get_memory_usage(df)
    if verbose: print(f"Memory usage decreased to {np.around(end_mem, 2)} Mb({np.around(100 * (start_mem - end_mem) / start_mem, 2)}% decrease)")    
    return df