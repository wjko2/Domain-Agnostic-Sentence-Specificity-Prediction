# Domain Agnostic Real-Valued Specificity Prediction
Code for
Wei-Jen Ko, Greg Durrett and Junyi Jessy Li, "Domain Agnostic Real-Valued Specificity Prediction", The AAAI Conference on Artificial Intelligence (AAAI), 2019

This is a text specificity predictor for any domain. 

**Citation:**
```
@InProceedings{ko2019domain,
  author    = {Ko, Wei-Jen and Durrett, Greg and Li, Junyi Jessy},
  title     = {Domain Agnostic Real-Valued Specificity Prediction},
  booktitle = {AAAI},
  year      = {2019},
}
```


## Dependencies
-Pytorch (Tested on 1.0.0)

-Numpy

## Data and resources
The glove vector file (840B.300d) is required. Download it and set the glove path in train.py and test.py
```
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```
The twitter, yelp, and movies data and annotations used in the paper is in dataset/data


## Running 
Training command:

python train.py  --gpu_id 0 --test_data twitter

Testing command:

python test.py  --gpu_id 0 --test_data twitter

## Using it on a new domain
To use it on a new domain, unlabeled sentences of the new domain is required.

When training,change the s1['unlab']['path'] in data2.py and the path of xsu in train.py and test.py to the unlabeled data file.

When testing, change the s1['test']['path'] in data2.py and the path of xst in test.py to the test sentences file.(And make sure s1['unlab']['path'] in data2.py and the path of xsu in test.py is the same file)

<b>The first line in the testing data is ignored.</b>



