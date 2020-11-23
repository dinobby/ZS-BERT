import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import f1_score, precision_recall_fscore_support
# from sklearn.metrics import precision_score, recall_score, f1_score

def compute_macro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
    '''
    This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
    code borrowed from the following link:
    https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
    '''
    if i == -1:
        i = len(predicted_idx)

    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = (predicted_idx[:i] == r)
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec+avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1

def extract_relation_emb(model, testloader):
    out_relation_embs = None
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for data in testloader:
        tokens_tensors, segments_tensors, marked_e1, marked_e2, \
        masks_tensors, relation_emb = [t.to(device) for t in data if t is not None]

        with torch.no_grad():
            outputs, out_relation_emb = model(input_ids=tokens_tensors, 
                                        token_type_ids=segments_tensors,
                                        e1_mask=marked_e1,
                                        e2_mask=marked_e2,
                                        attention_mask=masks_tensors,
                                        input_relation_emb=relation_emb)
            logits = outputs[0]
        if out_relation_embs is None:
            out_relation_embs = out_relation_emb
        else:
            out_relation_embs = torch.cat((out_relation_embs, out_relation_emb))    
    return out_relation_embs

def evaluate(preds, y_attr, y_label, idxmap, num_train_y, dist_func='inner'):
    assert dist_func in ['inner', 'euclidian', 'cosine']
    if dist_func == 'inner':
        tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=lambda a, b: -(a@b))
    elif dist_func == 'euclidian':
        tree = NearestNeighbors(n_neighbors=1)
    elif dist_func == 'cosine':
        tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=lambda a, b: -((a@b) / (( (a@a) **.5) * ( (b@b) ** .5) )))
    tree.fit(y_attr)
    predictions = tree.kneighbors(preds, 1, return_distance=False).flatten() + num_train_y
    p_macro, r_macro, f_macro = compute_macro_PRF(predictions, y_label)
    return p_macro, r_macro, f_macro