import os, sys
import pandas as pd
import numpy as np

import joblib
from pathlib import Path
import json
import sys, os
from tensorflow.keras.models import load_model

#Custom module imports
from ai4hpc_deployment.src.utils import transform_dsos_data, tsfresh_extract_features, scale_data, predict_vae
 
class AI4HPCPredict():
    
    def __init__(self, model_folder_path,deployment_metadata_filename,scaler_filename,model_name):
        
        self._prepare_model(model_folder_path,
                            deployment_metadata_filename,
                            scaler_filename,
                            model_name)
                
    def _prepare_model(self, model_folder_path, deployment_metadata_filename,scaler_filename,model_name):
        model_folder_path = Path(model_folder_path)        
        print(f"The model and other files will be loaded from: {model_folder_path}")

        #The deployment_metadata file includes the threshold and column names of the model
        with open(model_folder_path / deployment_metadata_filename, "r") as fp: 
            deployment_metadata = json.load(fp)
            
        self.threshold = deployment_metadata['threshold']    
        self.fc_parameters = deployment_metadata['tsfresh_column_names']

        #Load the scaler
        self.loaded_scaler = joblib.load(model_folder_path / scaler_filename) 
        print(f"Scaler is loaded {self.loaded_scaler}")

        #Load the model to serve
        self.loaded_model = load_model(model_folder_path / model_name, compile=False)
        print(self.loaded_model.summary())      
        
        print(f"Model prep is completed")
        
    def predict_pipeline(self, meminfo_df, vmstat_df, procstat_df):
        
        job_id_df = transform_dsos_data(meminfo_df, vmstat_df, procstat_df)

        fe_job_df = tsfresh_extract_features(job_id_df, 
                                             self.fc_parameters, 
                                             "component_id", 
                                             "timestamp")

        fe_job_df = scale_data(fe_job_df, self.loaded_scaler)

        #Generate predictions
        pred = predict_vae(fe_job_df, self.loaded_model, self.threshold)
        result_df = pd.DataFrame(pred, index=fe_job_df.index, columns=['pred'])

        print(result_df)   
        
        return result_df
