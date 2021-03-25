# ZS-BERT
This repository contains the implementation of the paper "Towards Zero-Shot Relation Extraction with Attribute Representation Learning".

# DataSet
You can download the datasets employed in our work from the following link:
- [WikiZSL (Daniil Sorokin and Iryna Gurevych, 2017)](https://drive.google.com/file/d/1Ju5ngSL9Bpb7K0br11ocyPWzDA8N5-9v/view?usp=sharing)
- [FewRel (Xu et al., 2018)](https://drive.google.com/file/d/1q_q1TA-c_xSCj0-CscWxmvvExr46ZcQS/view?usp=sharing)

and place them to the `/data` folder.

# Structure
```
ZS-BERT/
├── model
    ├── model.py
    ├── data_helper.py
    ├── evaluation.py
    ├── train_wiki.py
    └── train_fewrel.py
└── resources/
    ├── property_list.html
└── data/
    ├── wiki_train_new.json
    └── fewrel_all.json
```

# Requirements
python >= 3.7.0
torch >= 1.4.0
or simply run:
```
pip install -r requirements.txt
```

# Train ZS-BERT
If you wish to train on the wiki dataset, run:
```
python3 train_wiki.py --seed 300 --n_unseen 10 --gamma 7.5 --alpha 0.4 --dist_func 'inner' --batch_size 4 --epochs 10
```
Otherwise to train on FewRel dataset, you can run:
```
python3 train_fewrel.py --seed 300 --n_unseen 10 --gamma 7.5 --alpha 0.4 --dist_func 'inner' --batch_size 4 --epochs 10
```
inside the `/model` folder.
