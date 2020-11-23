import torch
import random
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer

class ZSBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.relation_emb_dim = config.relation_emb_dim
        self.margin = torch.tensor(config.margin)
        self.alpha = config.alpha
        self.dist_func = config.dist_func
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fclayer = nn.Linear(config.hidden_size*3, self.relation_emb_dim)
        self.classifier = nn.Linear(self.relation_emb_dim, self.config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        e1_mask=None,
        e2_mask=None,
        head_mask=None,
        inputs_embeds=None,
        input_relation_emb=None,
        labels=None,
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0] # Sequence of hidden-states of the last layer.
        pooled_output   = outputs[1] # Last layer hidden-state of the [CLS] token further processed 
                                     # by a Linear layer and a Tanh activation function.

        def extract_entity(sequence_output, e_mask):
            extended_e_mask = e_mask.unsqueeze(1)
            extended_e_mask = torch.bmm(
                extended_e_mask.float(), sequence_output).squeeze(1)
            return extended_e_mask.float()
        
        e1_h = extract_entity(sequence_output, e1_mask)
        e2_h = extract_entity(sequence_output, e2_mask)
        context = self.dropout(pooled_output)
        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)
        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.fclayer(pooled_output)
        relation_embeddings = torch.tanh(pooled_output)
        relation_embeddings = self.dropout(relation_embeddings)
        logits = self.classifier(relation_embeddings) # [batch_size x hidden_size]
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            gamma = self.margin.to(device)
            ce_loss = nn.CrossEntropyLoss()
            loss = (ce_loss(logits.view(-1, self.num_labels), labels.view(-1))) * self.alpha
            zeros = torch.tensor(0.).to(device)
            for a, b in enumerate(relation_embeddings):
                max_val = torch.tensor(0.).to(device)
                for i, j in enumerate(input_relation_emb):
                    if a==i:
                        if self.dist_func == 'inner':
                            pos = torch.dot(b, j).to(device)
                        elif self.dist_func == 'euclidian':
                            pos = torch.dist(b, j, 2).to(device)
                        elif self.dist_func == 'cosine':
                            pos = torch.cosine_similarity(b, j, dim=0).to(device)
                    else:
                        if self.dist_func == 'inner':
                            tmp = torch.dot(b, j).to(device)
                        elif self.dist_func == 'euclidian':
                            tmp = torch.dist(b, j, 2).to(device)
                        elif self.dist_func == 'cosine':
                            tmp = torch.cosine_similarity(b, j, dim=0).to(device)
                        if tmp > max_val:
                            if labels[a] != labels[i]:
                                max_val = tmp
                            else:
                                continue
                neg = max_val.to(device)
#                 print(f'neg={neg}')
#                 print(f'neg-pos+gamma={neg - pos + gamma}')
#                 print('===============')
                if self.dist_func == 'inner' or self.dist_func == 'cosine':
                    loss += (torch.max(zeros, neg - pos + gamma) * (1-self.alpha))
                elif self.dist_func == 'euclidian':
                    loss += (torch.max(zeros, pos - neg + gamma) * (1-self.alpha))
            outputs = (loss,) + outputs
        return outputs, relation_embeddings  # (loss), logits, (hidden_states), (attentions)