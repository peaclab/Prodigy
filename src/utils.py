from datetime import datetime
import time
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from constants import junk_cols, common_cols, excluded_cols
import yaml
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=yaml.YAMLLoadWarning)

def convert_str_time_to_unix(str_time):
    
    curr_format = '%Y-%m-%d %H:%M:%S.%f'
    
    datetime_object = datetime.strptime(str_time, curr_format).replace(microsecond=0)
    
    return int(time.mktime(datetime_object.timetuple()))

def add_job_ids(df, job_ids):
    """
        The example sampler CSV's have only one job_id. This function synthetically adds job_ids to the same dataframe
    """
    temp_list = []
    
    for job_id in job_ids:
        df['job_id'] = job_id
        temp_list.append(df.copy(deep=True))
        
    return pd.concat(temp_list)

def transform_dsos_data(meminfo_df, vmstat_df, procstat_df, silent=True):
        
    if not (set(meminfo_df.job_id.unique()) == set(vmstat_df.job_id.unique()) == set(procstat_df.job_id.unique())):
        print(f"WARNING: Provided samplers do not contain the same unique job_ids. The code will try to select the minimal subset of job_ids")
        
    common_job_ids = list((set(meminfo_df.job_id.unique()) & set(vmstat_df.job_id.unique()) & set(procstat_df.job_id.unique())))
    
    training_data_list = []
    for job_id in common_job_ids:        
        single_job_data = transform_dsos_job_data((meminfo_df[meminfo_df['job_id'] == job_id]), (vmstat_df[vmstat_df['job_id'] == job_id]), (procstat_df[procstat_df['job_id'] == job_id]), silent)
        training_data_list.append(single_job_data)            
        
    return pd.concat(training_data_list)


def process_raw_metrics(data, silent=True):
    """Process data based on YAML"""      
    
    if not silent:
        print(f"Processing metrics based on the YAML data")
        
    with open('eclipse_metric_info.yaml', 'r') as f:
        metric_info = yaml.load(f)    
                
    new_data = {}
    for col in data.columns:
        if col not in metric_info:   
            if not silent:
                print("{} not in YAML".format(col))
        elif metric_info[col] == 'cumulative':
            new_data[col] = np.diff(data[col].interpolate()) ## maybe NAN problem when the metric is 0
            if any(new_data[col] < 0):
                if not silent:
                    print("Column {} decreased".format(col))
        elif metric_info[col] in ['important', 'noncumulative']:
            new_data[col] = data[col].interpolate()[1:]
        elif metric_info[col] == 'unknown':
            new_data[col] = data[col].interpolate()[1:]
            if all(np.diff(data[col].interpolate()) >= 0):
                if not silent:
                    print("{} did not decrease".format(col))
        elif metric_info[col] in ['limit', 'unimportant']:
            pass
        else:
            raise IOError("Condition doesn't exist for {}".format(
                metric_info[col]))
    return pd.DataFrame(new_data, index=data.index[1:])
    

def transform_dsos_job_data(meminfo_df, vmstat_df, procstat_df, silent=True):
    
    assert len(meminfo_df['job_id'].unique()) == 1, "All the samplers must contain only one job_id. You can input multiple job_ids using transform_dsos_data"
    assert len(vmstat_df['job_id'].unique()) == 1, "All the samplers must contain only one job_id. You can input multiple job_ids using transform_dsos_data"
    assert len(procstat_df['job_id'].unique()) == 1, "All the samplers must contain only one job_id. You can input multiple job_ids using transform_dsos_data"
    
    curr_job_id = meminfo_df['job_id'].unique()[0]
        
    meminfo_df.drop(columns=junk_cols,inplace=True)
    vmstat_df.drop(columns=junk_cols,inplace=True)    
    procstat_df.drop(columns=junk_cols,inplace=True)
    
    if isinstance(meminfo_df['timestamp'].values[0], str):
        meminfo_df['unix_timestamp'] = meminfo_df['timestamp'].apply(lambda x: convert_str_time_to_unix(x))    
        vmstat_df['unix_timestamp'] = vmstat_df['timestamp'].apply(lambda x: convert_str_time_to_unix(x))    
        procstat_df['unix_timestamp'] = procstat_df['timestamp'].apply(lambda x: convert_str_time_to_unix(x))   
    else:
        meminfo_df['unix_timestamp'] = meminfo_df['timestamp'].astype(int)
        vmstat_df['unix_timestamp'] = vmstat_df['timestamp'].astype(int)
        procstat_df['unix_timestamp'] = procstat_df['timestamp'].astype(int)
        
    sampler_col_names = [curr_col + '::{}'.format("meminfo")  if curr_col not in excluded_cols else curr_col for curr_col in meminfo_df.columns ]
    meminfo_df.columns = sampler_col_names

    sampler_col_names = [curr_col + '::{}'.format("vmstat")  if curr_col not in excluded_cols else curr_col for curr_col in vmstat_df.columns ]
    vmstat_df.columns = sampler_col_names

    sampler_col_names = [curr_col + '::{}'.format("procstat")  if curr_col not in excluded_cols else curr_col for curr_col in procstat_df.columns ]
    procstat_df.columns = sampler_col_names
    
    non_per_core_cols = [curr_col for curr_col in procstat_df.columns if not ('per_core' in curr_col) and not (curr_col in excluded_cols)]
    common_comp_ids = list(set(meminfo_df.component_id.values) & set(procstat_df.component_id.values) & set(vmstat_df.component_id.values))
    
    cleaned_node_data = []

    for comp_id in common_comp_ids:
                
        node_meminfo_df = meminfo_df[meminfo_df['component_id'] == comp_id]
        node_vmstat_df = vmstat_df[vmstat_df['component_id'] == comp_id]
        node_procstat_df = procstat_df[procstat_df['component_id'] == comp_id]
        
        common_time = list(set(node_meminfo_df.unix_timestamp.values) & set(node_procstat_df.unix_timestamp.values) & set(node_vmstat_df.unix_timestamp.values))
        

        node_meminfo_df = node_meminfo_df[node_meminfo_df['unix_timestamp'].isin(common_time)].drop(columns=common_cols)
        node_meminfo_df.set_index('unix_timestamp', inplace=True)

        node_vmstat_df = node_vmstat_df[node_vmstat_df['unix_timestamp'].isin(common_time)].drop(columns=common_cols)
        node_vmstat_df.set_index('unix_timestamp', inplace=True)

        node_procstat_df = node_procstat_df[node_procstat_df['unix_timestamp'].isin(common_time)].drop(columns=common_cols)
        node_procstat_df.set_index('unix_timestamp', inplace=True)
        node_procstat_df = node_procstat_df[non_per_core_cols]

        node_data_df = pd.concat([node_meminfo_df, node_vmstat_df, node_procstat_df],axis=1)                
        
        node_data_df = process_raw_metrics(node_data_df)
                
        node_data_df.index.name = "timestamp"
        node_data_df.reset_index(inplace=True)    
        node_data_df.insert(1, 'job_id', curr_job_id)                
        node_data_df.insert(2, 'component_id', comp_id)
                
        cleaned_node_data.append(node_data_df)
        
        if not silent:
            print(f"Component ID: {comp_id}")
            print(f"Num. unique timestamps for each sampler: {len(node_meminfo_df.unix_timestamp.unique())}, {len(node_vmstat_df.unix_timestamp.unique())}, {len(node_procstat_df.unix_timestamp.unique())}")
            print(f"Common time length: {len(common_time)}")        
    
    return pd.concat(cleaned_node_data)