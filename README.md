# GlobalAIHack

## Installation libraries

```
conda create --name GlobalAIHack37 python=3.7
conda activate GlobalAIHack37
conda install rdkit -c rdkit
pip install xgboost catboost
pip install -U scikit-learn
conda install -c dglteam dgl
conda install pytorch  -c pytorch
pip install dgllife
pip install -U imbalanced-learn
pip install biopandas 
```

## Running baseline

```(bash)
$ python3 src.baselines.run_baseline
```

## Running saved model

All saved models are in **models** folder.\
Use script **runner.py** to load model and make predictions with it.

Example:

```(bash)
$ ./runner.py -m ./models/MACCSkeys_xgboost_score_0_4731 -f MACCS
```
