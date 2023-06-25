# Prodigy

This repository contains the code for Prodigy: Towards Unsupervised Anomaly Detection in Production HPC Systems.

### Reproducibility Experiments

The goal of this section is to help users get acquianted with the open source framework as soon as possible. We provide a small production dataset, which will help replicate the results for Figure 6.

#### Requirements



#### Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8079388.svg)](https://doi.org/10.5281/zenodo.8079388)

We have chosen four applications, namely LAMMPS, sw4, sw4Lite, and ExaMiniMD, to encompass both real and proxy applications. We have executed each application five times on four compute nodes without introducing any anomalies. To showcase our experiment, we have specifically selected the "memleak" anomaly as it is one of the most commonly occurring types. Additionally, we have also executed each application five times with the chosen anomaly. The dataset we have collected consists of a total of 160 samples, with 80 samples labeled as anomalous and 80 samples labeled as healthy. For the details of applications please refer to the paper. A more detailed information regarding synthetic anomalies can be found in [HPAS repository](https://github.com/peaclab/HPAS).

The applications were run on Eclipse, which is situated at Sandia National Laboratories. Eclipse comprises 1488 compute nodes, each equipped with 128GB of memory and two sockets. Each socket contains 18 E5-2695 v4 CPU cores with 2-way hyperthreading, providing substantial computational power for scientific and engineering applications.

#### Generate results

1. Navigate to the `src` directory where `reproducibility_experiments.py` is located.

2. Open the script file `reproducibility_experiments.py` in a text editor.

3. The script contains several parameters that need to be configured before running the experiments. These parameters are defined within the `main` function and should be modified according to your specific requirements. The parameters include:

- `repeat_nums`: A list of integers specifying the repeat numbers for the experiments.
- `expConfig_nums`: A list of integers specifying the experimental configuration numbers.
- `data_dir`: The directory path where the dataset is located.
- `pre_selected_features_filename`: The file path of the previously determined selected features (if available). Set it to `None` if feature extraction needs to be performed.
- `output_dir`: The directory path where the experiment outputs will be stored.
- `verbose`: Set it to `True` if you want to see important logging `INFO` messages during the execution of the script. By default, it is set to `False`, which prints all logging messages.

**Note**: Please ensure that the specified directories exist before running the script. If they do not exist, the script will attempt to create them.

4. Once you have configured the parameters in the `main` function, you can run the script using the following command:

```
python reproducibility_experiments.py
```

5. The script generates the following outputs:

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

#### Plot results

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

7. The plot will be displayed on the screen, and if `verbose` is set to `True`, it will print "Saved the plot" after saving the plot. If you are using the 

8. You can open the saved PDF file to view and analyze the plotted results.