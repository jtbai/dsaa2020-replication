# Replicability of the experimentation

## Items to replicate
1. Figures from table VI, VII, VIII
1. t-statistics and p-values for experiments
    1. Proposed model vs Baseline model 
    1. Proposed model at 15% of train data vs Baseline at 100% data
    1. Proposed model simultaneous transfer learning vs Sequential Transfer learning
1. Training process for baseline model and our model

## 0. Prerequisite
1. Python 3.7.x
1. A GPU of 4Go
1. Cuda 10.1
1. Git large file system extention (git-lfs)
    * https://packagecloud.io/github/git-lfs/install#bash-deb 
1. A copy of the repository
    * https://anonymous.4open.science/repository/dsaa-2020/
    * This includes code and datasets
1. Installed packages from requirements.txt
    * ``` pip install -r requirements.txt ```   
    
    

Other training parameters are hardcoded 

## 1. Figure from tables & t-statistics
To generate figures for tables VI, VII and VIII, simply run the command 

``` python3 generate_officiel_figures.py ```

Items from the will be presented in the output

     
## 2. Training process
### Train models
Run the command: ``` python3 train_model.py ```

Use the following arguments in the CLI
* `-m <model>`:  use `HAN` for baseline model and `MLHAN` for proposed model
* `-r <seed>`: for a specific random starting point (default is `42`). We alternatively used `42`, `420`,`26`, `1337`, `101` for seeds
* `-e <experiment-name>`: use `fullcorpus` for full experiment, `10pct`, `15pct`,`30pct`, for generability experiment at 10%, 15% and 30% of the train dataset, `xlearn` for transfer learning experiment 

Examples:
- `python3 train.py -m han -e fullcorpus -r 1337`
- `python3 train.py -m mlhan -e fullcorpus -r 1337`
- `python3 train.py -m han -e 15pct -r 1203`
- `python3 train.py -m mlhan -e 15pct -r 1203`
- `python3 train.py -m mlhan -e xlearn -r 1234`

### Evaluate models
Run the command: ```python3 evaluate_models.py```

It will recursively evaluate all trained model
