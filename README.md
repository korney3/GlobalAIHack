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
pip install notebook
conda install matplotlib
```

## Running baseline

```(bash)
$ python3 src.baselines.run_baseline
```

## Running saved model

All saved models are in **models** folder.\
Use script **runner.py** to load model and make predictions with it.

Options:
- **-m/--model**\
Path to file with model

- **-f/--fingerprints**\
Type of fingerprints. Can be one of ['TOPOTORSION', 'MACCS', 'RDKitFP', 'PATTERN', 'ATOMPAIR', 'ECFP4']

- **-o/--output** (optional)\
Name of submission file. Will be saved in **predictions** folder

Example:

```(bash)
$ ./runner.py -m ./models/MACCSkeys_xgboost_score_0_4731 -f MACCS
```

## Results visualization



According to feature importance, these 3 MACCS substructures are the most important:
```
1. $(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1) #ACH2CH2A > 1
2. [#16]~*(~*)~* #SA(A)A > 1
3. [#7]~*~[#8] #NAO > 0
4. [#7;!H0] # NH > 0
5. [#8]~[#16]~[#8] # OSO > 0
6. [#8]!:*:* # O not %A%A
7. '[#16]!:*:*' # S not %A%A
```

![](./download.png)
