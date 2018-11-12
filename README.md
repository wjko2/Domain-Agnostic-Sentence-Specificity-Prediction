Code for
Wei-Jen Ko, Greg Durrett and Junyi Jessy Li, "Domain Agnostic Real-Valued Specificity Prediction", The AAAI Conference on Artificial Intelligence (AAAI), 2019

The twitter, yelp, and movies data and annotations used in the paper is in dataset/pdtb

This is a text specificity predictor for any domain. 
To use it on a new domain, unlabeled sentences of the new domain is required.

Please change the s1['unlab']['path'] in data2.py and the path of xsu in train.py and test.py to the unlabeled data file.
Also change the s1['test']['path'] in data2.py and the path of xst in test.py to the test sentences file.
The output file path in line 328 of test.py.

The first line in the testing data is ignored.


Training command:
python train.py  --gpu_id 0 --test_data twitter

Testing command:
python test.py  --gpu_id 0 --test_data twitter
