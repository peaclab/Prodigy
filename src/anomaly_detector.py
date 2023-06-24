import logging
from pathlib import Path
import joblib
import pandas as pd
import json
import os, sys

from vae import VAE
from data_pipeline import DataPipeline
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np


class AnomalyDetector():
    
    def __init__(self, **kwargs):
        
        #Assuming this class is initialized from the repo's src directory
        curr_dir = Path(os.getcwd()).parent
        self.model_dir = kwargs.get("model_dir", curr_dir / "models/vae")
        self.model_filename = kwargs.get("model_filename", "model.h5") 
        self.model_weights_filename = kwargs.get("model_weights_filename", "model-weights.h5") 
        self.scaler_filename = kwargs.get("scaler_filename", "scaler.save")
        self.deployment_metadata_filename = kwargs.get("deployment_metadata_filename", "deployment_metadata.json")
        self.verbose = kwargs.get("verbose", False)
                
        self.logger = logging.getLogger(__name__)
        
        #Function initialization
        self._prepare_metadata()        
        self._build_prepare_model(input_dim=len(self.raw_column_names))
        
    def _prepare_metadata(self):
        
        #The deployment_metadata file includes the threshold and column names of the model
        with open(Path(self.model_dir) / self.deployment_metadata_filename, "r") as fp: 
            deployment_metadata = json.load(fp)

        self.threshold = deployment_metadata['threshold']    
        self.fe_column_names = deployment_metadata['fe_column_names']
        self.raw_column_names = deployment_metadata['raw_column_names']        
        self.logger.info(f"Feature extraction columns are loaded")

        #Load the scaler
        self.loaded_scaler = joblib.load(Path(self.model_dir) / self.scaler_filename)
        if self.verbose:
            self.logger.info(f"Scaler is loaded")
            self.logger.info(f"The anomaly detection threshold is {self.threshold}")
        
    def _build_prepare_model(self, input_dim):
        
        intermediate_dim = int(input_dim / 2)
        latent_dim = int(input_dim / 3)
        
        self.model = VAE(
                    input_dim=input_dim,
                    intermediate_dim=intermediate_dim,
                    latent_dim=latent_dim,
                    learning_rate=None,
                    verbose=self.verbose,
        )
        
        self.model.load_model_weights(Path(self.model_dir) / self.model_weights_filename)
        self.model.threshold = self.threshold
        
        if self.verbose:
            self.logger.info(f"Built the model and loaded the weights")
                                    
    def calculate_reconstruction_error(self, data):
        
        recon_data = self.model.predict(data)
        return np.mean(np.abs(data - recon_data), axis=1)
    
    def _predict_anomaly(self, data):
        
        mae_data = self.calculate_reconstruction_error(data)
        
        pred = [1 if curr_mae > self.threshold else 0 for curr_mae in mae_data]
        
        return pred[0] if len(pred) == 1 else pred
        

    def prediction_pipeline(self, input_ts):
        
        temp = input_ts.copy(deep=True)
        
        pipeline = DataPipeline(x_train_filename=None, 
                        y_train_filename=None, 
                        x_test_filename=None, 
                        y_test_filename=None)
        
        input_fe = pipeline.tsfresh_generate_features(temp, 
                                                      fe_config=None, 
                                                      kind_to_fc_parameters=self.fe_column_names)
        #print(input_fe.columns)
        input_fe = input_fe[self.raw_column_names]
        result_df = input_fe.index.to_frame(index=False)
        
        ls_scaled_data = self.loaded_scaler.transform(input_fe)
        
        #This is the VAE model imported from VAE.py
        preds, recon_errors = self.model.predict_anomaly(ls_scaled_data)
        result_df.loc[:, 'preds'] = preds
        result_df.loc[:, 'recon_errors'] = recon_errors
                
        return result_df