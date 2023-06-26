# Prodigy

This repository contains the code for Prodigy: Towards Unsupervised Anomaly Detection in Production HPC Systems.


Maintainer: 
* **Burak Aksar** - *baksar@bu.edu* 

Developers:  
* **Burak Aksar** - *baksar@bu.edu*  & **Efe Sencan** - *esencan@bu.edu*


## Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8079388.svg)](https://doi.org/10.5281/zenodo.8079388)

We released the dataset we used in Section 6.2. We have chosen four applications, namely LAMMPS, sw4, sw4Lite, and ExaMiniMD, to encompass both real and proxy applications. We have executed each application five times on four compute nodes without introducing any anomalies. To showcase our experiment, we have specifically selected the "memleak" anomaly as it is one of the most commonly occurring types. Additionally, we have also executed each application five times with the chosen anomaly. The dataset we have collected consists of a total of 160 samples, with 80 samples labeled as anomalous and 80 samples labeled as healthy. For the details of applications please refer to the paper. A more detailed information regarding synthetic anomalies can be found in [HPAS repository](https://github.com/peaclab/HPAS).

The applications were run on Eclipse, which is situated at Sandia National Laboratories. Eclipse comprises 1488 compute nodes, each equipped with 128GB of memory and two sockets. Each socket contains 18 E5-2695 v4 CPU cores with 2-way hyperthreading, providing substantial computational power for scientific and engineering applications.

## Supercomputing Artifact Evaluation

The goal of this section is to help users get acquianted with the open source framework as soon as possible. We provide a small production dataset, which will replicate the results for Figure 6.

### Computing Environment

We suggest using a Ubuntu-based Linux distribution. We provide specific instructions for Ubuntu 18.04 and Python 3.6.x in Chameleon Cloud. If you are using another cluster, or a local computer please start from **Setup & Requirements section**.

#### Chameleon Cloud (15 human-minutes)

If you haven’t use Chameleon Cloud before, please follow this [guide](https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html#step-1-log-in-to-chameleon) to get started.

To get access to Chameleon resources, you will need to be associated with a project that is assigned a resource allocation. Rest of the guide assumes you are part of a project and have allocation. 

Unlike traditional cloud computing platforms, you cannot immediately launch an instance whenever you want to. Chameleon uses a reservation system, where users must reserve machines beforehand. 

**Reserving a node**

* Go to CHI@TACC and find Leases from the left menu
* Enter “prodigy_sc23_ae_lease” as a Lease Name 
* Leave Start Date and Start Time empty
* Click Next
* Select Reserve Hosts; and 1 instance is enough
* Select compute_haswell_ib as node_type
* Click Next
* Click Create

**Launching an instance**

Please choose an image with Ubuntu18.04. Our suggestion is the following: **CC-Ubuntu18.04-20190626**

When the instance is created successfully, follow the below guidelines to access your instance.
* [Associating an IP address](https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html#associating-an-ip-address)
* [Accessing Your Instance](https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html#accessing-your-instance)


#### Setup & Requirements (2 human-minutes + 10 compute-minutes) 

For the sake of organization, please create a folder named `prodigy_artifacts`. Please make sure you are using Python 3.6.x. The requirements are tested with Python 3.6.8 and 3.6.9.

To install venv in Ubuntu 18.04: (Optional)

```sudo apt-get update```

```sudo apt-get install python3-venv```

Setup the virtual environment and install requirements: 

```python3 -m venv prodigy_py36_venv```

```source prodigy_py36_venv/bin/activate```

```pip install --upgrade pip```  => The version is 21.3.1

```pip install -r py36_deployment_reqs.txt```


Execute the following bash command to download the files from Zenodo or you can manually transfer:

```zenodo_get 10.5281/zenodo.8079388```


Extract the contents of the tar file using the following command:

```tar -xf eclipse_small_prod_dataset.tar```

Your folder should have the following files after the extraction: 

* fe_eclipse_tsfresh_raw_CHI_2000.json
* prod_train_data.hdf
* prod_test_data.hdf
* prod_train_label.csv
* prod_test_label.csv

#### Generate results (2 human-minutes + 2 compute-hours) 



1. Navigate to the `src` directory where `reproducibility_experiments.py` is located.

2. Open the script file `reproducibility_experiments.py` in a text editor.

3. The script contains several parameters that need to be configured before running the experiments. These parameters are defined within the `main` function and should be modified according to your specific requirements. The parameters include:

- `repeat_nums`: A list of integers specifying the repeat numbers for the experiments. **If you want to shorten the duration of the experiment, please do reduce number of repeats**. Default number is 5 repetitions.
- `expConfig_nums`: A list of integers specifying the experimental configuration numbers.
- `data_dir`: The directory path where the dataset is located.
- `pre_selected_features_filename`: The file path of the previously determined selected features (if available). Set it to `None` if feature extraction needs to be performed.
- `output_dir`: The directory path where the experiment outputs will be stored.
- `verbose`: Set it to `True` if you want to see important logging `INFO` messages during the execution of the script. By default, it is set to `False`, which prints all logging messages.

Example configurations 

```python
#Repeats all configurations once (15 - 20 compute-minutes on with one Haswell compute node in Chameleon Cloud)
repeat_nums = [0]
expConfig_nums = [0, 1, 2, 3, 4, 5]

#Repeats all configurations once (2 compute-hours on with one Haswell compute node in Chameleon Cloud)
repeat_nums = [0, 1, 2, 3, 4, 5]
expConfig_nums = [0, 1, 2, 3, 4, 5]

#Disclaimer: For the paper results, we repeat the experiments ten times. If you want to obtain the same results please set `repeat_nums = [0,1,2,3,4,5,6,7,8,9]` in the previous stage, otherwise, it's normal to observe small differences. 

```

**Note**: Please ensure that the specified directories exist before running the script. If they do not exist, the script will attempt to create them.

4. Once you have configured the parameters in the `main` function, you can run the script using the following command:

```
python reproducibility_experiments.py
```

5. Go grab a coffee or have lunch

6. The script generates the following outputs:

- Output directories:
  - The `output_dir` directory (specified in the parameters) will be created if it does not exist.
  - The `results` directory will be created within the `output_dir` to store the experiment results.

- Experiment results:
  - For each combination of `repeat_nums` and `expConfig_nums`, the script generates two JSON files:
    - A classification report for the test results (`expConfig_{expConfig_num}_repeatNum_{repeat_num}_testResults.json`).
    - Dataset statistics (`expConfig_{expConfig_num}_repeatNum_{repeat_num}_dataStats.json`).
  
- Model and metadata:
  - The trained model and deployment metadata are saved in the `output_dir`.
  - The trained model is saved as `model.h5`.
  - The deployment metadata is saved as `deployment_metadata.json`.

**Note**: Ensure that you have write permissions for the specified output directories.

#### Plot results (2 human-minutes + 3 compute-minutes)

This script allows you to plot results from experiment data stored in JSON files. It generates a bar plot of the macro average F1-scores based on different experimental configurations and the number of healthy samples in the training data.

1. Navigate to the `src` directory where `reproducibility_plots.py` is located.

2. Open the script file `reproducibility_plots.py` in a text editor.

3. Modify the following variables according to your needs:

   - `results_dir`: The directory path where the experiment results are stored.
   - `plot_output_dir`: The directory path where the plot output will be saved.
   - `verbose`: Set it to `True` if you want to see additional information during the execution of the script.
      
Make sure to place the experiment data files (e.g., `expConfig_0_repeatNum_0_dataStats.json`) in the `results` directory and specify the correct paths for `results_dir` and `plot_output_dir` in the script.

4. Save the modifications made to the script file.

5. Open a command prompt or terminal and navigate to the directory where the script file is located.

6. Run the script using the command:

```python
python plot_results.py
```

The script will read the experiment data from the specified `results_dir`, generate a bar plot, and save it in the specified `plot_output_dir` as `prodigy_increasing_num_samples_results.pdf`.

7. The plot will be displayed on the screen if your terminal supports it, nevertheless, the plot will be saved as a PDF to the designated output. 

8. You can open the saved PDF file to view and analyze the plotted results.


### Using Prodigy with Your Pipeline

During our experiments we use Lightweight Distributed Metric Service [LDMS](https://github.com/ovis-hpc/ldms-containers) to collect telemetry data using the available samplers: "meminfo", "vmstat", and "procstat". The flow of the designed pipeline shown below, and feel free to update according to your needs.

#### Data Format 

If you have your own sampler data collected from another telemetry frameworks, you can convert them to the necessary format shown below.

Train or test data must have the following 3 common columns: **job_id, component_id, and timestamp**. component_id column is also known as node_id. However, since the node_ids are not unique, we also require job_id. In other words, 
job_id and component_id combination gives us a unique representation which will be matched with labels.

|    |   job_id |   component_id |   timestamp |   MemTotal::meminfo |   Active::meminfo |   processes::procstat |
|---:|---------:|---------------:|------------:|--------------------:|------------------:|----------------------:|
|  0 |       66 |              1 |  1678928719 |             2031012 |            496632 |                  3430 |
|  1 |       66 |              1 |  1678928722 |             2031012 |            496632 |                  3430 |
|  2 |       66 |              1 |  1678928723 |             2031012 |            496632 |                  3430 |
|  3 |       66 |              1 |  1678928724 |             2031012 |            496632 |                  3430 |
|  4 |       66 |              1 |  1678928725 |             2031012 |            496632 |                  3430 |



Train or test label must have the component_id and job_id columns, and a label colum, which is **binary_anom**.

|    |   component_id |   job_id |   binary_anom |
|---:|---------------:|---------:|--------------:|
|  0 |              1 |       66 |             0 |
|  0 |              2 |       66 |             0 |
|  0 |              3 |       66 |             0 |
|  0 |              4 |       66 |             0 |


By looking at the above data structures, we can interpret the data as following: job_id 66 run on 4 compute nodes and all compute nodes were healthy.



#### Predictions 

If you want to use the trained model in your pipeline, you need to use **AnomalyDetector** class. 

The code below will read a sample CSV file from **input_data_path**. You can try with other DataFrames as long as it is the same as the training data format specified above. You can also check **local_predict.py.**

```python

from anomaly_detector import AnomalyDetector

#input_timeseries must have the same format shown above
model_dir = "ai4hpc_deployment/models/sample_vae_model"
anomaly_detector = AnomalyDetector(model_dir=model_dir)
preds = anomaly_detector.prediction_pipeline(input_timeseries)

```

The preds is a dataframe with where each job_id and component_id combination has a binary prediction value. 


|    |   job_id |   component_id |   pred |
|---:|---------:|---------------:|-------:|
|  0 |        0 |              1 |      0 |
|  1 |        0 |              2 |      0 |
|  2 |        0 |              3 |      0 |
|  3 |        0 |              4 |      0 |


### License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details
