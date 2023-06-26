import logging
import sys
# logging.basicConfig(format='%(asctime)s %(levelname)-7s %(message)s',
#                     stream=sys.stderr, level=logging.INFO)
from pathlib import Path
import pandas as pd
import joblib
import json

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tsfresh.feature_extraction import settings
from sklearn.model_selection import train_test_split

from data_pipeline import DataPipeline
from vae import VAE


def main(repeat_nums, expConfig_nums, data_dir, pre_selected_features_filename, output_dir, verbose=False):
    
    logging.basicConfig(format='%(asctime)s %(levelname)-7s %(message)s', stream=sys.stderr, level=logging.INFO if verbose else logging.DEBUG)
        
    logging.info("This is an info message")
    logging.debug("This is a debug message")
    
    extract_pre_selected_features = pre_selected_features_filename is not None
    
    # Create output directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info("Created outputs directory")
    else:
        logging.info("Output directory already exists")

    if not os.path.exists(output_dir + "/results"):
        os.makedirs(output_dir + "/results")
        logging.info("Created results directory")
    else:
        logging.info("Results directory already exists")

    # Iterate over repeat_nums
    for repeat_num in repeat_nums:
        logging.info(f"######### Repeat Num: {repeat_num}  #########")

        # Iterate over expConfig_nums
        for expConfig_num in expConfig_nums:
            logging.info(f"######### Experimental Configuration: {expConfig_num}  #########")

            if pre_selected_features_filename is not None:
                with open(pre_selected_features_filename, "r") as fp:
                    selected_features_json = json.load(fp)
                fe_selected_features = selected_features_json['tsfresh_column_names']
                logging.info("The previously selected features will be used")

            # Set healthy_test_data_percentage based on expConfig_num
            if expConfig_num == 0:
                healthy_test_data_percentage = 0.2
            elif expConfig_num == 1:
                healthy_test_data_percentage = 0.4
            elif expConfig_num == 2:
                healthy_test_data_percentage = 0.6
            elif expConfig_num == 3:
                healthy_test_data_percentage = 0.8
            elif expConfig_num == 4:
                healthy_test_data_percentage = 0.9
            elif expConfig_num == 5:
                healthy_test_data_percentage = 0.95

            pipeline = DataPipeline()
            x_train, y_train, x_test, y_test, x_val, y_val = pipeline.load_HPC_data(data_dir)

            all_y = pd.concat([y_train, y_test])
            logging.debug(f"Shape of all_y: {all_y.shape}")

            all_x = pd.concat([x_train, x_test])
            all_x.set_index(['job_id', 'component_id'], inplace=True)
            logging.debug(f"Shape of all_x: {all_x.shape}")

            selected_apps = ['exa', 'lammps', 'sw4', 'sw4lite']
            selected_labels = ['none', 'memleak']

            exp_config_dict = {
                'selected_apps': selected_apps,
                'selected_labels': selected_labels,
                'dataset_stats': {
                    'train': {
                        '0': 0,
                        '1': 0
                    },
                    'test': {
                        '0': 0,
                        '1': 0
                    }
                }
            }

            curr_all_y = all_y[(all_y['app_name'].isin(selected_apps)) & (all_y['anom_name'].isin(selected_labels))]

            healthy_labels = curr_all_y[curr_all_y['binary_anom'] == 0]
            anom_labels = curr_all_y[curr_all_y['binary_anom'] != 0]

            # Train test split on the healthy node_ids
            train_label_healthy, test_label_healthy = train_test_split(
                healthy_labels, test_size=healthy_test_data_percentage)

            logging.debug(f"Shape of train_label_healthy: {train_label_healthy.shape}")
            logging.debug(f"Shape of test_label_healthy: {test_label_healthy.shape}")

            exp_config_dict['dataset_stats']['train']['0'] = train_label_healthy.shape[0]
            exp_config_dict['dataset_stats']['test']['0'] = test_label_healthy.shape[0]

            # Train data only have healthy node_ids
            x_train = all_x.loc[train_label_healthy.index]
            y_train = train_label_healthy.copy()

            assert set(x_train.index.get_level_values('component_id')) == set(y_train.index.get_level_values('component_id'))

            logging.debug(f"Shape of train data: {x_train.shape} with {len(set(x_train.index.get_level_values('component_id')))} unique jobid compid combos")
            logging.debug(f"Train label distribution:\n{y_train['binary_anom'].value_counts()}")

            assert len(x_train.index.unique()) == len(y_train)

            # Test data will have some healthy and anomalous node_ids
            test_data_healthy = all_x.loc[test_label_healthy.index]
            assert set(test_data_healthy.index.get_level_values('component_id')) == set(test_label_healthy.index.get_level_values('component_id'))

            anom_data = all_x.loc[anom_labels.index]
            exp_config_dict['dataset_stats']['test']['1'] = anom_labels.shape[0]

            assert set(anom_data.index.get_level_values('component_id')) == set(anom_labels.index.get_level_values('component_id'))

            x_test = pd.concat([test_data_healthy, anom_data])
            y_test = pd.concat([test_label_healthy, anom_labels])

            assert set(x_test.index.get_level_values('component_id')) == set(y_test.index.get_level_values('component_id'))

            logging.debug(f"Shape of test data: {x_test.shape}")
            logging.debug(f"Test label distribution:\n{y_test['binary_anom'].value_counts()}")

            assert len(x_test.index.unique()) == len(y_test)

            x_train.reset_index(inplace=True)
            x_test.reset_index(inplace=True)

            if extract_pre_selected_features:
                x_train_fe = pipeline.tsfresh_generate_features(x_train, fe_config=None, kind_to_fc_parameters=fe_selected_features)
            else:
                x_train_fe = pipeline.tsfresh_generate_features(x_train, fe_config="minimal")

            if y_train is not None:
                y_train = y_train.loc[x_train_fe.index]

            assert all(y_train.index == x_train_fe.index)

            # Generate features for the test data
            if x_test is not None:
                if extract_pre_selected_features:
                    x_test_fe = pipeline.tsfresh_generate_features(x_test, fe_config=None, kind_to_fc_parameters=fe_selected_features)
                else:
                    x_test_fe = pipeline.tsfresh_generate_features(x_test, fe_config="minimal")

                if y_test is not None:
                    y_test = y_test.loc[x_test_fe.index]

                assert all(y_test.index == x_test_fe.index)

                # Make the number of columns and the order equal
                if len(x_test_fe.columns) < len(x_train_fe.columns):
                    x_train_fe = x_train_fe[x_test_fe.columns]
                    logging.debug(f"Shape of x_train_fe: {x_train_fe.shape}")
                elif len(x_test_fe.columns) > len(x_train_fe.columns):
                    x_test_fe = x_test_fe[x_train_fe.columns]
                    logging.debug(f"Shape of x_test_fe: {x_test_fe.shape}")

                x_train_fe = x_train_fe[x_test_fe.columns]
                assert all(x_train_fe.columns == x_test_fe.columns)
                x_test_fe = x_test_fe.loc[y_test.index]

            # Apply scaler to train and test data (if it exists)
            if x_test is not None:
                x_train_scaled, x_test_scaled = pipeline.scale_data(x_train_fe, x_test_fe, save_dir=output_dir)
            else:
                x_train_scaled, x_test_scaled = pipeline.scale_data(x_train_fe, None, save_dir=output_dir)

            logging.debug(f"Shape of x_train_scaled: {x_train_scaled.shape}")

            input_dim = x_train_scaled.shape[1]
            intermediate_dim = int(input_dim / 2)
            latent_dim = int(input_dim / 3)

            if 'vae' in locals():
                logging.debug("Vae exists; will delete to be safe")
                del vae
            else:
                logging.debug("Vae is not defined")

            vae = VAE(
                name="model",
                input_dim=input_dim,
                intermediate_dim=intermediate_dim,
                latent_dim=latent_dim,
                learning_rate=1e-4
            )

            # Train the VAE model
            vae.fit(
                x_train=x_train_scaled,
                epochs=1000,
                batch_size=32,
                validation_split=0.1,
                save_dir=output_dir,
                verbose=0
            )
            
            logging.info("Model training is completed")

            deployment_metadata = {
                'threshold': vae.threshold,
                'raw_column_names': list(x_train_scaled.columns),
                'fe_column_names': settings.from_columns(list(x_train_scaled.columns))
            }

            with open(Path(output_dir) / 'deployment_metadata.json', 'w') as fp:
                json.dump(deployment_metadata, fp)

            y_pred_train, x_train_recon_errors = vae.predict_anomaly(x_train_scaled)

            if y_train is not None:
                logging.debug("Classification Report in Training Data\n")
                logging.debug(classification_report(y_train['binary_anom'].values, y_pred_train))

            if x_test is not None:
                assert all(x_test_scaled.columns == x_train_scaled.columns)
                assert all(x_test_scaled.index == y_test.index)

                y_pred_test, x_test_recon_errors = vae.predict_anomaly(x_test_scaled)
                logging.debug(f"Test data prediction results: {y_pred_test}")
                logging.debug(f"Selected threshold value: {vae.threshold}")

            if y_test is not None:
                y_test['binary_pred'] = y_pred_test
                y_test['recon_errors'] = x_test_recon_errors
                logging.debug("Classification Report in Test Data\n")
                logging.debug(classification_report(y_test['binary_anom'].values, y_pred_test))

                result_dict = classification_report(y_test['binary_anom'].values, y_pred_test, output_dict=True)

            # Writing result_dict to JSON
            with open(Path(output_dir) / "results" / f"expConfig_{expConfig_num}_repeatNum_{repeat_num}_testResults.json", "w") as outfile:
                json.dump(result_dict, outfile)

            # Writing exp_config_dict to JSON
            with open(Path(output_dir) / "results" / f"expConfig_{expConfig_num}_repeatNum_{repeat_num}_dataStats.json", "w") as outfile:
                json.dump(exp_config_dict, outfile)            
                
            logging.info("Results generated and saved to the corresponding directory")

if __name__ == '__main__':
    
    repeat_nums = [0, 1, 2, 3, 4 ]
    expConfig_nums = [0, 1, 2, 3, 4, 5]
    data_dir = "/home/cc/prodigy_artifacts/"
    #If this parameter is set, it will use the previously determined parameters, if it's None, it's going to extract features
    pre_selected_features_filename = "/home/cc/prodigy_artifacts/fe_eclipse_tsfresh_raw_CHI_2000.json"    
    output_dir = "/home/cc/prodigy_ae_output"
    verbose = True  # Set to True to display important logging INFO messages, otherwise it will print all logging messages
    main(repeat_nums, expConfig_nums, data_dir, pre_selected_features_filename, output_dir, verbose)
    
    logging.info("Script is completed")
