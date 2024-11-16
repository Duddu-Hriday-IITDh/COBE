from cProfile import label
from random import betavariate
from re import S
import re
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_opset9 import dim, unsqueeze
from transformers import BertModel, XLNetModel
from transformers.modeling_utils import (
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from typing import Optional
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from bert import BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Function
import math
from torch.autograd import Variable

def get_inverse_sqrt_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases with inverse_sqrt
    from the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to
    the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    decay_factor = num_warmup_steps ** 0.5 if num_warmup_steps > 0 else 1

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return decay_factor * float(current_step + 1) ** -0.5

    return LambdaLR(optimizer, lr_lambda, last_epoch)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss

class BertCon(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        :param bert_config: configuration for bert model
        """
        super(BertCon, self).__init__(bert_config)
        self.bert_config = bert_config
        self.bert = BertModel(bert_config)
        penultimate_hidden_size = bert_config.hidden_size
        
        # Define the RNN layer
        self.rnn = nn.RNN(input_size=penultimate_hidden_size, 
                          hidden_size=192, 
                          num_layers=1, 
                          batch_first=True, 
                          dropout=0.2)  # You can adjust dropout as needed
        
        # Final classification layer
        self.sent_cls = nn.Linear(192, bert_config.num_labels)  # For sentiment classification
        
        # Domain classification loss
        self.dom_loss1 = CrossEntropyLoss()
        self.dom_cls = nn.Linear(192, bert_config.domain_number)
        
        # Temperature for softmax (if needed for domain classification)
        self.tem = torch.tensor(0.05)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_labels=None,
                position_ids=None, head_mask=None, dom_labels=None, meg='train'):
        # Get BERT output
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        hidden = outputs[0]  # BERT last hidden states

        batch_num = hidden.shape[0]
        
        # Use the hidden states from the [CLS] token for the RNN input
        rnn_input = hidden[:, 0, :].unsqueeze(1)  # Take the [CLS] token's representation
        
        # Pass through RNN layer
        rnn_out, _ = self.rnn(rnn_input)
        rnn_out = rnn_out.squeeze(1)  # Squeeze out the time dimension
        
        if meg == 'train':
            # Sentiment classification loss
            if sent_labels is not None:
                sent_preds = self.sent_cls(rnn_out)
                sent_loss = CrossEntropyLoss()(sent_preds, sent_labels)
                
                # Domain classification loss (if needed)
                dom_preds = self.dom_cls(rnn_out)
                dom_loss = self.dom_loss1(dom_preds)
                
                # Total loss
                total_loss = sent_loss + dom_loss
                return total_loss
            else:
                return None  # If no labels are provided, return None (no training loss)
        
        elif meg == 'source':
            # For inference, return the RNN output (normalized)
            return F.normalize(rnn_out, p=2, dim=1)


