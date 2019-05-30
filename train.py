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

parser.add_argument("--batch_size", type=int, default=32)
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
if  params.test_data=='pdtb':
    params.esize=2784
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
            elif qq<params.dprob+params.iprob+params.sprob:
                wqwq=0
            else:
                if j+os<len(batch[i]) and j<max_len:
                    embed[j, i, :] = word_vec[batch[i][j+os]]+ np.random.normal(0, params.gnoise, params.word_emb_dim )

    return torch.from_numpy(embed).float(), lengths

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def trainepoch(epoch):
    
    print('\nTRAINING : Epoch ' + str(epoch))
    pdtb_net.train()
    pdtb_net2.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    if params.uss==-1:
        params.uss=len(unlab['s1'])
    permutation = np.random.permutation(len(train['s1']))
    permutationu = np.random.permutation(params.uss)
    permutationu=permutationu%50
    s1 = train['s1'][permutation]
    s1f = xsl[permutation]
    s_u = unlab['s1'][permutationu]
    suf = xsu[permutationu]
    target = train['label'][permutation]
    print (target)

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, params.esize, params.batch_size):
    
        s1_batch, s1_len = get_batch_aug(s1[stidx:stidx + params.batch_size],
                                     word_vec)
        s1_batch2, s1_len2 = get_batch_aug(s1[stidx:stidx + params.batch_size],
                                     word_vec)
        
        s1_batchf=torch.from_numpy(s1f[stidx:stidx + params.batch_size]+ np.random.normal(0, params.gnoise2, 14 )).float()*params.sf
        s1_batchf2=torch.from_numpy(s1f[stidx:stidx + params.batch_size]+ np.random.normal(0, params.gnoise2, 14 )).float()*params.sf
        
        su_batch, su_len = get_batch_aug(s_u[stidx:stidx + params.batch_size],
                                     word_vec)
        su_batchf=torch.from_numpy(suf[stidx:stidx + params.batch_size]+ np.random.normal(0, params.gnoise2, 14 )).float()*params.sf
        su_batch2, su_len2 = get_batch_aug(s_u[stidx:stidx + params.batch_size],
                                     word_vec)
        su_batchf2=torch.from_numpy(suf[stidx:stidx + params.batch_size]+ np.random.normal(0, params.gnoise2, 14 )).float()*params.sf
        if config_nli_model['use_cuda']:
            s1_batch= Variable(s1_batch).cuda()*params.wf
            s1_batchf= Variable(s1_batchf).cuda()
            s1_batch2= Variable(s1_batch2).cuda()*params.wf
            s1_batchf2= Variable(s1_batchf2).cuda()
            su_batch= Variable(su_batch).cuda()*params.wf
            su_batchf= Variable(su_batchf).cuda()
            su_batchf2= Variable(su_batchf2).cuda()
            
            su_batch2= Variable(su_batch2).cuda()*params.wf
            tgt_batch = Variable(torch.FloatTensor(target[stidx:stidx + params.batch_size])).cuda()
        else:
            s1_batch= Variable(s1_batch)*params.wf
            s1_batchf= Variable(s1_batchf)
            s1_batch2= Variable(s1_batch2)*params.wf
            s1_batchf2= Variable(s1_batchf2)

            su_batch= Variable(su_batch)*params.wf
            su_batchf= Variable(su_batchf)
            su_batchf2= Variable(su_batchf2)
            
            su_batch2= Variable(su_batch2)*params.wf
            tgt_batch = Variable(torch.FloatTensor(target[stidx:stidx + params.batch_size]))
        k = s1_batch.size(1)  # actual batch size
        output = pdtb_net((s1_batch, s1_len),s1_batchf)
        output2 = pdtb_net((s1_batch2, s1_len2),s1_batchf2)
        outputu = pdtb_net((su_batch, su_len),su_batchf)
        outputu2 = pdtb_net2((su_batch2, su_len2),su_batchf2)
        if params.loss==0:
            pred = output.data.max(1)[1]
        else:
            pred=output.data[:,0]>0
        

        assert len(pred) == len(s1[stidx:stidx + params.batch_size])
        if params.loss==0:
            ou = F.softmax(outputu, dim=1)
            
            ou2 = F.softmax(outputu2, dim=1)
            sou = F.softmax(output, dim=1)
            
            sou2 = F.softmax(output2, dim=1)
 
            a,_=torch.max(ou,1)
            sa,_=torch.max(sou,1)

            a=(a.detach()>params.th).view(-1,1).float()
            sa=(sa.detach()>params.th).view(-1,1).float()
            ou=ou*  torch.cat((a,a), 1)
            ou2=ou2*  torch.cat((a,a), 1)
            sou=sou*  torch.cat((sa,sa), 1)
            sou2=sou2*  torch.cat((sa,sa), 1)
        
        else:
            ou=outputu[:,0]
            ou2=outputu2[:,0]
            a=(ou.detach()>params.th).view(-1,1).float()
            ou=ou*  a
            ou2=ou2* a

        ou2.require_grad=False
        sou2.require_grad=False
        loss2=( F.mse_loss(ou, ou2.detach(), size_average=False)+F.mse_loss(sou, sou2.detach(), size_average=False)) / params.n_classes/params.batch_size
        # loss
        if params.loss==0:
            tgt_batch=torch.cat([1.0-tgt_batch.view(-1,1),tgt_batch.view(-1,1) ],dim=1)
            oop=F.softmax(output, dim=1)
            oop2=F.softmax(outputu, dim=1)
            loss3=0
            pppp=Variable(torch.FloatTensor([1/oop.size(0)]).cuda())
            dmiu=torch.mean(oop2[:,1])
            dstd=torch.std(oop2[:,1])
            loss3=loss3+torch.abs(torch.mean(oop2[:,1])-params.klmiu)+torch.abs(torch.std(oop2[:,1])-params.klsig)
            
            kss=float(params.klsig)
            
            
            loss1 = loss_fn(oop, tgt_batch.float())
        else:
            loss1 = loss_fn(output[:,0], (tgt_batch*2-1).float())
        if epoch>=params.se:
            loss=loss1+params.c*loss2+params.c2*loss3
        else:
            loss=loss1+params.c2*loss3
        all_costs.append(loss.item())
        words_count += (s1_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in pdtb_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm.cpu())

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        global global_step
        global_step =global_step +1
        gs=global_step
        update_ema_variables(pdtb_net, pdtb_net2, params.d, gs)

        optimizer.param_groups[0]['lr'] = current_lr
        if len(all_costs) == 10:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; '.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time))))
            print(logs[-1])
            #print (loss3)

            last_time = time.time()
            words_count = 0
            all_costs = []
    
    return 0






"""
Train model on Natural Language Inference task
"""
epoch = 1
for jpp in range (params.sss):
    epochh = 1
    gg=params.ne0
    while not stop_training and epochh <= gg:
        if params.bb==1:    
            train_acc = trainepochb(epoch)
        else:
            train_acc = trainepoch(epoch)
        epoch += 1
        if epoch== params.me:
            stop_training=1
            #fffa=open('opt'+params.test_data+str(params.c)+'ll'+str(params.c2)+'ll'+str(params.gnoise2)+'.txt','w')                                        
            #for ee in range(q.size(0)):
            #    fffa.write(str(q.data[ee,1])+'\n')
            #fffa.close()
        epochh+=1
    gg=params.n_epochs

print('\nTEST : Epoch {0}'.format(epoch))

# Save encoder instead of full model
torch.save(pdtb_net,
           os.path.join(params.outputdir, params.outputmodelname ))
torch.save(pdtb_net2,
           os.path.join(params.outputdir, '3os'+params.outputmodelname ))       
