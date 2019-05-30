import pickle
import os
import sys
import time
import argparse
import math
import random
import numpy as np
import scipy.stats
from generatefeatures import ModelNewText
from features import Space

import utils
import torch
from torch.autograd import Variable
import torch.nn as nn

from data2 import get_nli, get_batch, build_vocab,get_pdtb
from mutils import get_optimizer
from models import NLINet,PDTBNet
from torch.nn import functional as F
global_step=0

parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='dataset/data/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--c", type=float, default='1000')
parser.add_argument("--c2", type=float, default='100')
parser.add_argument("--tv", type=int, default='1')

parser.add_argument("--d", type=float, default='0.999')
parser.add_argument("--cth", type=float, default='0')
parser.add_argument("--klmiu", type=float, default='0.65')
parser.add_argument("--klsig", type=float, default='0.65')

parser.add_argument("--loss", type=float, default='0')
parser.add_argument("--rmu", type=int, default='0')
parser.add_argument("--md", type=int, default='0')


# training
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--esize", type=int, default=4342)
parser.add_argument("--dom", type=int, default=1)
parser.add_argument("--norm", type=int, default=1)

parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--se", type=int, default=4)
parser.add_argument("--me", type=int, default=31)

parser.add_argument("--dpout_model", type=float, default=0.5, help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0.5, help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=1, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=1, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--gnoise", type=float, default=0.1, help="max norm (grad clipping)")
parser.add_argument("--gnoise2", type=float, default=0.2, help="max norm (grad clipping)")
parser.add_argument("--dprob", type=float, default=0.15, help="max norm (grad clipping)")
parser.add_argument("--iprob", type=float, default=0.15, help="max norm (grad clipping)")
parser.add_argument("--sprob", type=float, default=0.05, help="max norm (grad clipping)")
parser.add_argument("--sf", type=float, default=1, help="max norm (grad clipping)")
parser.add_argument("--wf", type=float, default=1, help="max norm (grad clipping)")
parser.add_argument("--test_data", type=str, default="twitter", help="max norm (grad clipping)")
# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=100, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=100, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=2, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
parser.add_argument("--sptrain", type=str, default='0', help="max or mean")
parser.add_argument("--th", type=float, default='0.5', help="max or mean")
parser.add_argument("--uss", type=int, default='5000', help="max or mean")
parser.add_argument("--uss2", type=int, default='5000', help="max or mean")

parser.add_argument("--sss", type=int, default='50', help="max or mean")
parser.add_argument("--ne0", type=int, default='100', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=1, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")
parser.add_argument("--wed", type=int, default=300, help="seed")
parser.add_argument("--bb", type=int, default=0, help="seed")
parser.add_argument("--eeps", type=float, default=0.1, help="seed")




params, _ = parser.parse_known_args()
if params.wed==300:
    GLOVE_PATH = "glove.840B.300d.txt"

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
RT = "./"

BRNCLSTSPACEFILE = RT+"cotraining_models/brnclst1gram.space"
SHALLOWSCALEFILE = RT+"cotraining_models/shallow.scale"
SHALLOWMODELFILE = RT+"cotraining_models/shallow.model"
#NEURALBRNSCALEFILE = RT+"cotraining_models/neuralbrn.scale"
#NEURALBRNMODELFILE = RT+"cotraining_models/neuralbrn.model"

def initBrnSpace():
    s = Space(101)
    s.loadFromFile(BRNCLSTSPACEFILE)
    return s

def readScales(scalefile):
    scales = {}
    with open(scalefile) as f:
        for line in f:
            k,v = line.strip().split("\t")
            scales[int(k)] = float(v)
        f.close()
    return scales

#brnclst = utils.readMetaOptimizeBrownCluster()
#embeddings = utils.readMetaOptimizeEmbeddings()
#brnspace = initBrnSpace()
scales_shallow = readScales(SHALLOWSCALEFILE)
#scales_neuralbrn = readScales(NEURALBRNSCALEFILE)
#model_shallow = ll.load_model(SHALLOWMODELFILE)
#model_neuralbrn = ll.load_model(NEURALBRNMODELFILE)
def getFeatures(fin):
#    aligner = ModelNewText(brnspace,brnclst,embeddings)
    aligner = ModelNewText(0,0,0)
    aligner.loadFromFile(fin)
    #aligner.loadSentences('1',fin)
    aligner.fShallow()
    #aligner.fNeuralVec()
    #aligner.fBrownCluster()
    y,xs = aligner.transformShallow()
    print (len(xs))
    
    #_,xw = aligner.transformWordRep()
    return y,xs

train,valid, test,unlab ,trainu= get_pdtb(params.nlipath,params.dom,params.test_data,params.tv)
#train['label']=train['label']*0.8+0.1
#print(train['label'])

#print (train[s1].shape)
_,xsl = getFeatures(os.path.join(params.nlipath,'data.txt'))

if params.test_data=="twitter":
    _,xst= getFeatures(os.path.join(params.nlipath,'twitters.txt'))
    _,xsu = getFeatures('dataset/data/twitteru.txt')


elif params.test_data=="yelp":
    _,xst= getFeatures(os.path.join(params.nlipath,'yelps.txt'))
    _,xsu = getFeatures('dataset/data/yelpu.txt')

elif params.test_data=="movie":
    _,xst= getFeatures(os.path.join(params.nlipath,'movies.txt'))
    _,xsu = getFeatures('dataset/data/movieu.txt')


_,xslu= getFeatures(os.path.join(params.nlipath, 'aaai15unlabeled/all.60000.sents'))

mmm=(np.mean(np.asarray(xsu),axis=0))  
vvv=(np.var(np.asarray(xsu),axis=0))
vvv[vvv==0]=1
if params.norm==1:
    for i in range(len(xst)):
        xst[i]=(xst[i]-mmm)/vvv
    for i in range(len(xsl)):
        xsl[i]=(xsl[i]-mmm)/vvv
    for i in range(len(xsu)):
        xsu[i]=(xsu[i]-mmm)/vvv
    for i in range(len(xslu)):
        xslu[i]=(xslu[i]-mmm)/vvv
xstt=xst[params.tv:]
xst=xst[:params.tv]

word_vec = build_vocab(train['s1']+ #+ train['s2'] +
                      # valid['s1'] + valid['s2'] +
                       #test['s1'] + test['s2']
                       unlab['s1']+trainu['s1'], GLOVE_PATH)


if params.sptrain==1:
    train['s1']=train['s1']+trainu['s1']
    train['label']=np.concatenate((train['label'],trainu['label']))
    xsl=np.concatenate((xsl,xslu))
#for split in ['s1', 's2']:

for split in ['s1']:
    for data_type in ['train', 'valid', 'test', 'unlab', 'trainu']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] 
            #+            ['</s>']
            for sent in eval(data_type)[split]])
params.word_emb_dim = params.wed
params.klmiu=0.42
params.klsig=0.23

"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}
print(config_nli_model)
# model
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)

pdtb_net = PDTBNet(config_nli_model)
pdtb_net2 = PDTBNet(config_nli_model)
print(pdtb_net)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
if params.loss==0:
    loss_fn = nn.CrossEntropyLoss(weight=weight)
else:
    loss_fn = nn.SoftMarginLoss()
loss_fn=nn.BCELoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(pdtb_net.parameters(), **optim_params)

# cuda by default
if config_nli_model['use_cuda']:
    pdtb_net.cuda()
    pdtb_net2.cuda()
    loss_fn.cuda()


"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None
def get_batch_aug(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)+3
    embed = np.zeros((max_len, len(batch), params.wed))

    for i in range(len(batch)):
        os=0
        for j in range(max_len):
#            print(word_vec[batch[i][j]])
            qq=random.random()
            if qq<params.dprob:
                os=os+1                
                if j+os<len(batch[i]) and j<max_len:
                    embed[j, i, :] = word_vec[batch[i][j+os]]+ np.random.normal(0, params.gnoise, params.word_emb_dim )
            elif qq<params.dprob+params.iprob:
                os=os-1
                #embed[j, i, :] = np.random.uniform(-1,1,params.word_emb_dim)
#                embed[j, i, :] = word_vec[random.choice(list(word_vec.keys()))]+ np.random.normal(0, params.gnoise, params.word_emb_dim )
            elif qq<params.dprob+params.iprob+params.sprob:
#                embed[j, i, :] = word_vec[random.choice(list(word_vec.keys()))]+ np.random.normal(0, params.gnoise, params.word_emb_dim )
                #embed[j, i, :] = np.random.uniform(-1,1,params.word_emb_dim)
                wqwq=0
            else:
                if j+os<len(batch[i]) and j<max_len:
                    embed[j, i, :] = word_vec[batch[i][j+os]]+ np.random.normal(0, params.gnoise, params.word_emb_dim )

    return torch.from_numpy(embed).float(), lengths

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def evaluate(epoch, eval_type='valid', final_eval=False):
    kko=open('predictions.txt','w')
        

    pdtb_net.eval()
    pdtb_net2.eval()
    correcttp = 0.
    correcttn = 0.
    correctfp = 0.
    correctfn = 0.
    global val_acc_best, lr, stop_training, adam_stop


    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
 #   s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']
    targetv = valid['labelv'] if eval_type == 'valid' else test['labelv']
    for i in range(0, len(s1), params.batch_size):
    
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec,params.wed)
        if eval_type == 'valid':
            s1_batchf=torch.from_numpy(xst[i:i + params.batch_size]).float()*params.sf
        else:
            s1_batchf=torch.from_numpy(xstt[i:i + params.batch_size]).float()*params.sf
        if config_nli_model['use_cuda']:
            s1_batch= Variable(s1_batch).cuda()*params.wf
            s1_batchf= Variable(s1_batchf).cuda()
          
            tgt_batch = Variable(torch.LongTensor(target[: params.batch_size])).cuda()
            tgtv_batch = Variable(torch.FloatTensor(targetv[: params.batch_size])).cuda()
        else:
            s1_batch= Variable(s1_batch)*params.wf#.cuda()
            s1_batchf= Variable(s1_batchf)#.cuda()
            tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size]))#.cuda()
            tgtv_batch = Variable(torch.FloatTensor(targetv[i:i + params.batch_size]))#.cuda()

        
        # model forward
        output = pdtb_net2((s1_batch, s1_len),s1_batchf) 
        ou2 = F.softmax(output, dim=1)
            
        for sis in range(output.size(0)):
            kko.write(str(ou2.data[sis,1])+'\n')
        
        

    return 0



"""
Train model on Natural Language Inference task
"""
epoch = 1
pdtb_net2 = torch.load('savedir/3os'+params.outputmodelname)

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(0, 'test', True)
# Save encoder instead of full model
