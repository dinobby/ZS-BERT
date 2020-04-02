# ZS-BERT
Officail implementation of "ZS-BERT: Towards Zero-Shot Learning on Relation Extraction"

# DataSet
You can download the datasets present in our work from the original link:
- [WikiZSL](https://www.informatik.tu-darmstadt.de/ukp/research_6/data/lexical_resources/wikipedia_wikidata_relations/index.en.jsp)
- [FewRel](https://thunlp.github.io/1/fewrel1.html])

# Structure
```
ZS-BERT/
├── model
    └── ZSBERT.py
├── train.py
├── eval.py
├── utils
│   └── io.py
└── resources/
    ├── property_list.html
└── data/
    ├── wiki_train.json
    ├── wiki_test.json
    ├── fewrel_train.json
    └── fewrel_test.json
```

# Requirements
python >= 3.7.0
torch >= 1.4.0
```
pip install -r requirements.txt
```
