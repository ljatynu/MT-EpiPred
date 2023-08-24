# MT-EpiPred
MT-EpiPred: Multitask learning for prediction of small-molecule epigenetic modulators 
# MT-EpiPred: Multitask learning for prediction of small-molecule epigenetic modulators

## Introduction
This is the source code and dataset for the following paper: 

**MT-EpiPred: Multitask learning for prediction of small-molecule epigenetic modulators**

## Datasets
The datasets uploaded can be downloaded to train our model directly.

Before trainning, please run the python script *data_process.py* , which is under the *dataset* directory.
## Usage

### Installation
We used the following Python packages for the development by python 3.8.
```
- python = 3.8.11
- torch = 1.9.0
- rdkit = 2022.9.3
- scikit-learn = 0.23.1
- tqdm = 4.50.0
- numpy = 1.18.5
- pandas = 1.0.5
```

### Run code

Before trainning , please run the python script *data_process.py* , which is under the *dataset* directory.
```
python dataset/data_process.py
```

Run the main script *main.py* to train model, and you can change the setting (such as layer size) in the config file *config.py*
```
python main.py
```

To get the model's performance on each task after trainging, run the *test_valid_per_task.py* script. You can find the result in the path *result/\*_test_valid_per_task/*, the * will be replaced by layer_size(default: moderate).
```
python test_valid_per_task.py
```


## Reference
```
@article{MT-EpiPred,
  title={MT-EpiPred: Multitask learning for prediction of small-molecule epigenetic modulators},
  author={Ruihan Zhang‡, Xingran Xie‡, Dongxuan Ni‡, Jing Li*, Weilie Xiao*},
  journal={xxx},
  year={2023}
}
```
