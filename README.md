Code for
Wei-Jen Ko, Greg Durrett and Junyi Jessy Li, "Domain Agnostic Real-Valued Specificity Prediction", The AAAI Conference on Artificial Intelligence (AAAI), 2019


python train.py  --d 0.999 --optimizer adam,lr=0.0001 --enc_lstm_dim 100 --fc_dim 100 --gnoise 0.1 --dprob 0.15 --iprob 0.15  --cprob 0.15 --sf 1 --sptrain 0 --esize 4342 --dom 1 --tv 200 --wf 1 --th 0.5  --nonlinear_fc 1 --dpout_fc 0.5 --gpu_id 0 --cth 0 --seed 508 --dpout_model 0.5 --lrshrink 1 --loss 0  --test_data movie --wed 300  --gnoise2 0.2 --norm 1 --se 4  --n_epochs 5 --uss 5000 --sss  50  --c 0 --uss2 5000 --rmu 0 --ne0 100  --md 0 --c2 0 --me 31

python test.py  --d 0.999 --optimizer adam,lr=0.0001 --enc_lstm_dim 100 --fc_dim 100 --gnoise 0.1 --dprob 0.15 --iprob 0.15  --cprob 0.15 --sf 1 --sptrain 0 --esize 4342 --dom 1 --tv 200 --wf 1 --th 0.5  --nonlinear_fc 1 --dpout_fc 0.5 --gpu_id 0 --cth 0 --seed 508 --dpout_model 0.5 --lrshrink 1 --loss 0  --test_data movie --wed 300  --gnoise2 0.2 --norm 1 --se 4  --n_epochs 5 --uss 5000 --sss  50  --c 0 --uss2 5000 --rmu 0 --ne0 100  --md 0 --c2 0 --me 31
