import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters
from tsfresh.feature_extraction import settings

 
class DataPipeline():
    
    def __init__(self, **kwargs):
        """Initializes a `DataPipeline` object.
        
        Args:
            **kwargs: Dictionary containing the following optional keyword arguments:
                system_name (str): Name of the system (default is 'eclipse').
                x_train_filename (str): Name of the HDF file containing training data (default is 'prod_train_data.hdf').
                y_train_filename (str): Name of the CSV file containing training labels (default is 'prod_train_label.csv').
                x_test_filename (str): Name of the HDF file containing test data (default is 'prod_test_data.hdf').
                y_test_filename (str): Name of the CSV file containing test labels (default is 'prod_test_label.csv').
        """        
                
        self.window_size = 0
        self.dataset_name = kwargs.get('system_name', 'eclipse')                
        self.x_train_filename = kwargs.get('x_train_filename', 'prod_train_data.hdf') 
        self.y_train_filename = kwargs.get('y_train_filename', 'prod_train_label.csv') 
        self.x_val_filename = kwargs.get('x_val_filename', 'prod_val_data.hdf') 
        self.y_val_filename = kwargs.get('y_val_filename', 'prod_val_label.csv')         
        self.x_test_filename = kwargs.get('x_test_filename', 'prod_test_data.hdf')  
        self.y_test_filename = kwargs.get('y_test_filename', 'prod_test_label.csv')  

        self.raw_features = None        
        self.fe_features = None
        
        self.logger = logging.getLogger(__name__)

        
    def load_HPC_data(self, data_dir):
        """Loads data from the given directory and returns the training and test data and labels.
        
        Args:
            data_dir (str): Path to the directory containing the data files.
            
        Returns:
            tuple: A tuple containing the training and test data and labels (in that order).
        """        
        
        data_dir = Path(data_dir) 
        
        x_train = self._read_data(data_dir / self.x_train_filename)
        y_train = self._read_label(data_dir / self.y_train_filename) if self.y_train_filename is not None else None
        
        if not self.x_train_filename and not self.y_train_filename:
            assert list(x_train.index.get_level_values('component_id').unique()) == list(y_train.index.unique())        
            
        x_val = self._read_data(data_dir / self.x_val_filename) if self.x_val_filename is not None else None            
        y_val = self._read_label(data_dir / self.y_val_filename) if self.y_val_filename is not None else None
        
        if not self.x_val_filename and not self.y_val_filename:
            assert list(x_val.index.get_level_values('component_id').unique()) == list(y_val.index.unique())        
            
        
        x_test = self._read_data(data_dir / self.x_test_filename) if self.x_test_filename is not None else None            
        y_test = self._read_label(data_dir / self.y_test_filename) if self.y_test_filename is not None else None
        
        if not self.x_test_filename and not self.y_test_filename:
            assert list(x_train.index.get_level_values('component_id').unique()) == list(y_train.index.unique())        
            
        self.logger.info('Data read successfully')
        self.logger.info(f'Shape of x_train: {x_train.shape}')
        
        if y_train is not None:
            self.logger.info(f'Shape of y_train: {y_train.shape}')        
        if x_test is not None:
            self.logger.info(f'Shape of x_test: {x_test.shape}')
        if y_test is not None:
            self.logger.info(f'Shape of y_test: {y_test.shape}')        
                
        return x_train, y_train, x_test, y_test, x_val, y_val
        
        
    def generate_windows(self, data, window_size=60, skip_interval=15):
        """
        Generates rolling time windows for the input data, based on a given window size and skip interval.

        Args:
            data (pd.DataFrame): Input data to generate windows for.
            window_size (int): Size of the rolling window, in minutes. Defaults to 60.
            skip_interval (int): Number of minutes to skip between each window. Defaults to 15.

        Returns:
            pd.DataFrame: Dataframe containing the rolling time windows, with each row corresponding to a window.

        Raises:
            AssertionError: If the window size is 0.
        """
                
        assert window_size != 0, "Window size should be different than 0, to generate windows."
        
        data.reset_index(inplace=True)
        
        self.window_size = window_size
        
        data_windows = roll_time_series(
            data,
            column_id="component_id",
            column_sort="timestamp",
            max_timeshift=window_size,
            min_timeshift=window_size,
            rolling_direction=skip_interval,
            #n_jobs=-1,
        )   
            
        return data_windows
    
    def check_parameters(self, params):
        """
        Check if each parameter in params is allowed according to the allowed_values dictionary.

        Args:
            params (dict): Dictionary containing parameter names and their values.

        Raises:
            ValueError: If any parameter value is not in the allowed list.
        """
        
        #Dictionary containing parameter names and a list of allowed values. You can add parameters below.       
        allowed_values = {
                    'fe_config': ['minimal', 'efficient', None]
        }
        
        
        for param_name, param_value in params.items():
            if param_name not in allowed_values:
                continue
            if param_value not in allowed_values[param_name]:
                raise ValueError(f"Invalid value {param_value} for parameter {param_name}. Allowed values: {allowed_values[param_name]}")
    
    
    def tsfresh_generate_features(self, data, fe_config, kind_to_fc_parameters=None, column_id="uid", column_sort="timestamp"):
        """
        Extracts features from data using tsfresh library.

        Args:
            data (pd.DataFrame): Input data to extract features from.
            fe_config (str): Configuration of feature extractor. Can be "minimal" or "efficient".
            column_id (str): Name of column representing the ID of the time series.
            column_sort (str): Name of column representing the time of each observation.
            kind_to_fc_parameters (dict): Dictionary containing feature parameters for each feature kind.

        Raises:
            ValueError: If `fe_config` value is not in allowed list.

        Returns:
            pd.DataFrame: Extracted features.
        """        
        if data is None or len(data) == 0: 
            raise ValueError(f"Param [data] cannot be None or empty")
            
        self.check_parameters({'fe_config': fe_config})        
        
        if not (kind_to_fc_parameters is None):
            assert fe_config == None, "Either set fe_config or kind_to_fc_parameters, not both"
                                
        if np.any(pd.isnull(data)):
            self.logger.info(f'Raw time series: Before dropping NaNs: {data.shape}')
            data = data.dropna()
            self.logger.info(f'Raw time series:  Dropped NaNs: {data.shape}') 
        
        if data.index.names == ['component_id', 'timestamp']:
            data.reset_index(inplace=True)
         
        #Create a unique id column since job_id and component_id combo is the unique one
        data['uid'] = data['job_id'].astype(str) + '_' + data['component_id'].astype(str)
        data.drop(columns=['job_id','component_id'],inplace=True)
        
        if kind_to_fc_parameters is None:
            self.logger.info("TSFRESH will use default_fc_parameters")
            data_fe = extract_features(            
                data,
                column_id=column_id,
                column_sort=column_sort,
                default_fc_parameters=EfficientFCParameters() if fe_config == 'efficient' else MinimalFCParameters(),
            )
        else:
            self.logger.info("TSFRESH will use kind_to_fc_parameters")
            data_fe = extract_features(            
                data,
                column_id=column_id,
                column_sort=column_sort,
                kind_to_fc_parameters=kind_to_fc_parameters,
            )            
                                
        data_fe.reset_index(inplace=True)
        data_fe[['job_id', 'component_id']] = data_fe['index'].str.split('_', expand=True)
        data_fe.drop(columns=['index'], inplace=True)
        
        if self.window_size == 0 :
            data_fe.set_index(["job_id", "component_id"],inplace=True)
        else:
            data_fe.set_index(["job_id", "component_id", "timestamp"],inplace=True)
        
        self.logger.info(f'Feature extraction: Before dropping NaNs: {data_fe.shape}')
        data_fe = data_fe.dropna(axis=1, how='any')    
        self.logger.info(f'Feature extraction: Dropped NaNs: {data_fe.shape}') 
                
        self.raw_features = list(data_fe.columns)
        self.fe_features = settings.from_columns(self.raw_features)
        
        return data_fe
    
    def scale_data(self, x_train, x_test=None, save_dir=None):        
        """
        Scales data using MinMaxScaler.

        Args:
            x_train (pd.DataFrame): Training data to scale.
            x_test (pd.DataFrame, optional): Test data to scale. Defaults to None.
            save_dir (str, optional): Directory to save scaler object. Defaults to None.

        Returns:
            pd.DataFrame: Scaled training data.
            pd.DataFrame: Scaled test data.
        """        
    
        scaler = MinMaxScaler(feature_range=(0, 1), clip=True)

        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)        
        if not (x_test is None):
            self.logger.info(f"x_test is not None, scaling")            
            x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
        
        if not (save_dir is None):
            scaler_filename = "scaler.save"
            joblib.dump(scaler, Path(save_dir) / scaler_filename)
            self.logger.info(f"Scaler is saved")            
            
        return x_train, x_test
        
    def _read_data(self, abs_input_path):
        """
        Reads data from HDF5 file.

        Args:
            abs_input_path (str): Absolute path to input data file.

        Returns:
            pd.DataFrame: Input data.
        """        
        
        try:
            data = pd.read_hdf(abs_input_path)
        except FileNotFoundError:
            self.logger.error(f"File not found!: {abs_input_path}")
            return None
        
        return data
                                       
    def _read_label(self, abs_input_path):
        """
        Reads labels from CSV file.

        Args:
            abs_input_path (str): Absolute path to label file.

        Returns:
            pd.DataFrame: Label data.
        """                     
        try:
            label = pd.read_csv(abs_input_path)
            label['job_id'] = label['job_id'].astype('str')
            label['component_id'] = label['component_id'].astype('str') 
            label.set_index(['job_id', 'component_id'], inplace=True)
            
        except FileNotFoundError:
            self.logger.error(f"File not found!: {abs_input_path}")
            return None
        
        return label