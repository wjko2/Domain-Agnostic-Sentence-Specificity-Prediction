# Domain Agnostic Real-Valued Specificity Prediction
Code for
Wei-Jen Ko, Greg Durrett and Junyi Jessy Li, "Domain Agnostic Real-Valued Specificity Prediction", The AAAI Conference on Artificial Intelligence (AAAI), 2019

**Citation:**
```
@InProceedings{ko2019domain,
  author    = {Ko, Wei-Jen and Durrett, Greg and Li, Junyi Jessy},
  title     = {Domain Agnostic Real-Valued Specificity Prediction,
  booktitle = {AAAI},
  year      = {2019},
}
```


The twitter, yelp, and movies data and annotations used in the paper is in dataset/pdtb
This is a text specificity predictor for any domain. 


The glove vector file (840B.300d) is required. Download it and set the glove path in train.py and test.py



To use it on a new domain, unlabeled sentences of the new domain is required.

Please change the s1['unlab']['path'] in data2.py and the path of xsu in train.py and test.py to the unlabeled data file.
Also change the s1['test']['path'] in data2.py and the path of xst in test.py to the test sentences file.

The first line in the testing data is ignored.


Training command:
python train.py  --gpu_id 0 --test_data twitter

Testing command:
python test.py  --gpu_id 0 --test_data twitter


