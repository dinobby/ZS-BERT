import json
import random
import numpy as np
import pandas as pd
import data_helper

import torch
import random
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer
from evaluation import extract_relation_emb, evaluate
from model import ZSBert

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-s", "--seed", help="random seed", type=int, default=500, dest="seed")
parser.add_argument("-m", "--n_unseen", help="number of unseen classes", type=int, default=10, dest="m")
parser.add_argument("-g", "--gamma", help="margin factor gamma", type=float, default=7.5, dest="gamma")
parser.add_argument("-a", "--alpha", help="balance coefficient alpha", type=float, default=0.4, dest="alpha")
parser.add_argument("-d", "--dist_func", help="distance computing function", type=str, default='inner', dest="dist_func")
parser.add_argument("-b", "--batch_size", type=int, default=4, dest="batch_size")
parser.add_argument("-e", "--epochs", type=int, default=10, dest="epochs")

args = parser.parse_args()
# set randam seed, this affects the data spliting.
random.seed(args.seed) 

# load data
with open('../data/fewrel_all.json') as f:
    raw_train = json.load(f)

# random sample k unseen classes for dev/test
dev_keys = random.sample(list(raw_train.keys()), k=5)
dev_values = [raw_train[k] for k in dev_keys]
dev_keys_else = set(raw_train.keys()) - set(dev_keys)
dev_values_else = [raw_train[k] for k in dev_keys_else]
raw_dev = dict(zip(dev_keys, dev_values))

raw_train = dict(zip(dev_keys_else, dev_values_else))
test_keys = random.sample(list(raw_train.keys()), k=args.m)
test_values = [raw_train[k] for k in test_keys]
raw_test = dict(zip(test_keys, test_values))
keys_else = set(raw_train.keys()) - set(test_keys)
values_else = [raw_train[k] for k in keys_else]
raw_train = dict(zip(keys_else, values_else))

training_data = []
for k, v in raw_train.items():
    for i in v:
        i['relation'] = k
        training_data.append(i)

validation_data = []
for k, v in raw_dev.items():
    for i in v:
        i['relation'] = k
        validation_data.append(i)

test_data = []
for k, v in raw_test.items():
    for i in v:
        i['relation'] = k
        test_data.append(i)

train_label = list(raw_train.keys())
dev_label = list(raw_dev.keys())
test_label = list(raw_test.keys())
print('there are {} kinds of relation in train.'.format(len(set(train_label))))
print('there are {} kinds of relation in dev.'.format(len(set(dev_label))))
print('there are {} kinds of relation in test.'.format(len(set(test_label))))
print('number of union of train and dev: {}'.format(len(set(train_label) & set(dev_label))))
print('number of union of dev and test: {}'.format(len(set(dev_label) & set(test_label))))
print('number of union of train and test: {}'.format(len(set(train_label) & set(test_label))))

property2idx, idx2property, pid2vec = data_helper.generate_attribute(train_label, dev_label, test_label, att_dim=1024)
print(len(training_data))
print(len(validation_data))
print(len(test_data))

bertconfig = BertConfig.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',
                                        num_labels=len(set(train_label)),
                                        finetuning_task='fewrel-zero-shot')
bertconfig.relation_emb_dim = 1024
bertconfig.margin = args.gamma
bertconfig.alpha = args.alpha
bertconfig.dist_func = args.dist_func

model = ZSBert.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', config=bertconfig)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)

trainset = data_helper.FewRelDataset('train', training_data, pid2vec, property2idx)
trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=data_helper.create_mini_batch,shuffle=True)

val_y_attr, val_y = [], []
test_y_attr, test_y = [], []
val_idxmap, test_idxmap = {}, {}

for i, dev in enumerate(validation_data):
    label = int(property2idx[dev['relation']])
    val_y.append(label)
    val_idxmap[i] = label

for i in set(dev_label):
    val_y_attr.append(pid2vec[i])

val_y_attr = np.array(val_y_attr)
val_y = np.array(val_y)

print(val_y_attr.shape)
print(val_y.shape)

for i, test in enumerate(test_data):
    label = int(property2idx[test['relation']])
    test_y.append(label)
    test_idxmap[i] = label

for i in set(test_label):
    test_y_attr.append(pid2vec[i])

test_y_attr = np.array(test_y_attr)
test_y = np.array(test_y)

print(test_y_attr.shape)
print(test_y.shape)

devset = data_helper.FewRelDataset('dev', validation_data, pid2vec, property2idx)
devloader = DataLoader(devset, batch_size=256, 
                        collate_fn=data_helper.create_mini_batch)

testset = data_helper.FewRelDataset('test', test_data, pid2vec, property2idx)
testloader = DataLoader(testset, batch_size=256, 
                        collate_fn=data_helper.create_mini_batch)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

best_p = 0.0
best_r = 0.0
best_f1 = 0.0
for epoch in range(args.epochs):
    print(f'============== TRAIN ON THE {epoch+1}-th EPOCH ==============')
    running_loss = 0.0
    for step, data in enumerate(trainloader):

        tokens_tensors, segments_tensors, marked_e1, marked_e2, \
        masks_tensors, relation_emb, labels = [t.to(device) for t in data]
        optimizer.zero_grad()

        outputs, out_relation_emb = model(input_ids=tokens_tensors, 
                                    token_type_ids=segments_tensors,
                                    e1_mask=marked_e1,
                                    e2_mask=marked_e2,
                                    attention_mask=masks_tensors,
                                    input_relation_emb=relation_emb,
                                    labels=labels)

        loss = outputs[0]
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 1000 == 0:
            print(f'[step {step}]' + '=' * (step//1000))

    print('============== EVALUATION ON DEV DATA ==============')
    preds = extract_relation_emb(model, devloader).cpu().numpy()
    p, r, f1 = evaluate(preds, val_y_attr, val_y, val_idxmap, len(set(train_label)), args.dist_func)
    print(f'loss: {running_loss:.2f}, precision: {p:.4f}, recall: {r:.4f}, f1 score: {f1:.4f}')

    print('============== EVALUATION ON TEST DATA ==============')
    preds = extract_relation_emb(model, testloader).cpu().numpy()
    pt, rt, f1t = evaluate(preds, test_y_attr, test_y, test_idxmap, len(set(train_label))+len(set(dev_label)), args.dist_func)
    print(f'[testing performance] precision: {pt:.4f}, recall: {rt:.4f}, f1 score: {f1t:.4f}')

    if f1 > best_f1:
        best_p = p
        best_r = r
        best_f1 = f1
        torch.save(model, f'best_f1_{best_f1}_fewrel_epoch_{epoch}_m_{args.m}_alpha_{args.alpha}_gamma_{args.gamma}')
    print(f'[best val] precision: {best_p:.4f}, recall: {best_r:.4f}, f1 score: {best_f1:.4f}')