# Replicability of the experimentation

## Items to replicate
1. Training process for baseline model and our model
1. Figures from table VI, VII, VIII
1. t-statistics and p-values for experiments
    1. Proposed model vs Baseline model 
    1. Proposed model at 15% of train data vs Baseline at 100% data
    1. Proposed model simultaneous transfer learning vs Sequential Transfer learning

## 0. Prerequisite
1. A GPU of 4Go
1. Cuda 1.0.1
1. A copy of the repository
    * https://anonymous.4open.science/repository/dsaa-2020/
    * This includes code and datasets
1. Installed packages from requirements.txt
    * ``` pip install -r requirements.txt ```    
## 1. Training process
Run the command ``` python3 train_model.py ```

Use the following arguments in the CLI
* `-m han` for HAN baseline model
* `-m mlhan` for MLHAN proposed model
* `-r <seed>` for a specific random starting point (default is 42). We alternatively used 42, 420, 26, 1337, 101 for seeds


Other training parameters are hardcoded 

## 2. Figure from tables & t-statistics
Run the command ``` python3 compare_prediction_for_experimentations.py ```

Items will be presented in the output
