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

class BertCon(nn.Module):  # Changed from BertPreTrainedModel to nn.Module
    def __init__(self, bert_config):
        """
        :param bert_config: configuration for bert model
        """
        super(BertCon, self).__init__()
        self.bert_config = bert_config
        self.bert = BertModel(bert_config)
        
        # CNN layers for feature extraction
        self.cnn1 = nn.Conv1d(bert_config.hidden_size, 256, kernel_size=3, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.cnn3 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.fc = nn.Linear(64, 192)  # Final fully connected layer after CNN

        # Temperature scaling parameter (not used in contrastive loss now)
        self.tem = torch.tensor(0.05)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_labels=None,
                position_ids=None, head_mask=None, meg='train'):
        
        # Forward pass through BERT model
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        hidden = outputs[0]  # Get the hidden states from BERT
        
        # Apply CNN on BERT's output
        x = hidden.permute(0, 2, 1)  # Change shape for CNN [batch_size, hidden_size, seq_len]
        x = F.relu(self.cnn1(x))
        x = self.pool(x)
        x = F.relu(self.cnn2(x))
        x = self.pool(x)
        x = F.relu(self.cnn3(x))
        x = self.pool(x)
        
        # Flatten the CNN output and pass through fully connected layer
        x = x.flatten(1)  # Flatten all dimensions except batch
        x = self.fc(x)
        
        if meg == 'train':
            # If training, calculate loss (no contrastive learning anymore)
            loss = 0  # You can add your custom loss here if needed
            return loss

        elif meg == 'source':
            # During inference or source domain prediction, return normalized CNN features
            return F.normalize(x, p=2, dim=1)
