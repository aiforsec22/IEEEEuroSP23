## Looking Beyond IoCs: Automatically Extracting Attack Patterns from External CTI

This repository contains implementation fotr the paper "Looking Beyond IoCs: Automatically Extracting Attack Patterns from External CTI".

### Install
Run `pip install -r requirements.txt` on a virtual environment


### Demo
Open the notebook in `notebooks/attack-pattern-extraction.ipynb` in Google colab for demo on attack pattern extraction and mapping from CTI texts.   
Open the notebook in `notebooks/malware-similarity.ipynb` in Google colab for malware similarity analysis from all triples

### How to Run

#### Attack Pattern Extraction
Change directory to `attack_pattern`  
Run `pip install -r requirements.txt` to install the dependencies in a virtual environment

To train the sentence classification model: `python finetune_sentence_classification.py --save-path=logs/sentence_classification`  
Edit the `params_dict` values in the python file to fine tune other combination of hyperparameters.

To train the entity extraction model: `python finetune_entity_extraction.py --save-path=logs/entity_extraction`  
Edit the `params_dict` values in the python file to fine tune other combination of hyperparameters.


#### Named Entity Recognition
Change directory to `ner`  
Run `pip install -r requirements.txt` to install the dependencies in a virtual environment

To train NER model run: `python train_ner.py`
cfgEdit `cfg` values in the python file to fine tune other combination of hyperparameters.

`infer.py` shows examples on how to create annotation in BRAT format with trained model.


#### Relation Extraction
Change directory to `relation_extraction`  
Run `pip install -r requirements.txt` to install the dependencies in a virtual environment

To train Relation Extraction model run: `python train_supervised.py`  

#### Prediction with Tucker
Run the notebook `train_class_specific.ipynb` to get prediction result for different classes  
Run the notebook `tucker_malkg_usecase.ipynb` to get atack pattern prediction for FluBot  
 


#### Annotated Dataset
Text file and corresponding BRAT annotations are provided in `data/1-150` directory.
