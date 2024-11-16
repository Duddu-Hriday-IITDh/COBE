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

# class BertCon(BertPreTrainedModel):
    # def __init__(self, bert_config):
    #     """
    #     :param bert_config: configuration for bert model
    #     """
    #     super(BertCon, self).__init__(bert_config)
    #     self.bert_config = bert_config
    #     self.bert = BertModel(bert_config)
    #     penultimate_hidden_size = bert_config.hidden_size
        
    #     # Shared encoder for feature extraction
    #     self.shared_encoder = nn.Sequential(
    #         nn.Linear(penultimate_hidden_size, penultimate_hidden_size // 2),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(0.3),
    #         nn.Linear(penultimate_hidden_size // 2, 192),
    #     )

    #     # Decoder to reconstruct the original hidden representation
    #     self.decoder = nn.Sequential(
    #         nn.Linear(192, penultimate_hidden_size // 2),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(0.3),
    #         nn.Linear(penultimate_hidden_size // 2, penultimate_hidden_size),
    #     )

    #     # Reconstruction loss (MSE)
    #     self.mse_loss = nn.MSELoss()

    # def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_labels=None,
    #             position_ids=None, head_mask=None, dom_labels=None, meg='train'):
    #     outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
    #                         attention_mask=attention_mask, head_mask=head_mask)
    #     hidden = outputs[0]  # Shape: (batch_size, sequence_length, hidden_size)
        
    #     batch_num = hidden.shape[0]
    #     w = hidden[:, 0, :]  # Taking the representation for the [CLS] token
    #     h = self.shared_encoder(w)  # Passing through shared encoder

    #     if meg == 'train':
    #         # Reconstruct the hidden state using the decoder
    #         reconstructed_h = self.decoder(h)
            
    #         # Compute reconstruction loss (MSE)
    #         loss = self.mse_loss(reconstructed_h, w)  # Loss between original input and reconstruction
    #         return loss

    #     elif meg == 'source':
    #         return F.normalize(h, p=2, dim=1)


class BertCon(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        :param bert_config: configuration for bert model
        """
        super(BertCon, self).__init__(bert_config)
        self.bert = BertModel(bert_config)
        penultimate_hidden_size = bert_config.hidden_size
        self.shared_encoder = nn.Sequential(
            nn.Linear(penultimate_hidden_size, penultimate_hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(penultimate_hidden_size // 2, 192),
        )
        self.dom_cls = nn.Linear(192, bert_config.domain_number)
        self.tem = nn.Parameter(torch.tensor(0.05, requires_grad=True))  # Temperature for similarity (used in meta-learning)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_labels=None,
                position_ids=None, head_mask=None, dom_labels=None, meg='train'):
        # Get BERT embeddings
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        hidden = outputs[0]
        batch_num = hidden.shape[0]
        w = hidden[:, 0, :]
        
        # Shared Encoder to transform the embeddings
        h = self.shared_encoder(w)
        
        # Normalize the output embeddings (optional depending on your task)
        h = F.normalize(h, p=2, dim=1)
        
        if meg == 'train':
            if fast_adaptation:
                # During meta-training, simulate a quick adaptation to new tasks
                loss = self.meta_loss(h, task_labels, num_adaptation_steps)
                return loss
            else:
                # During normal training, return the output embeddings (h)
                return h

        elif meg == 'source':
            return h  # Return the transformed hidden states for use in other tasks

    def meta_loss(self, embeddings, task_labels, num_adaptation_steps):
        """
        Simulate the meta-learning loss with fast adaptation on a task using the given embeddings.

        Args:
            embeddings: Encoded representations after the shared encoder
            task_labels: Ground truth labels for the task
            num_adaptation_steps: Number of gradient steps for task adaptation

        Returns:
            Loss for meta-training
        """
        # Simulate gradient updates for a few steps (meta-learning)
        task_loss = CrossEntropyLoss()(embeddings, task_labels)
        for _ in range(num_adaptation_steps):
            # Apply a dummy gradient step (this is where meta-learning adaptation would occur)
            task_loss.backward()
            # Here, you would normally perform an optimization step, but for the sake of the example, we don't.
        
        return task_loss  # Return the task loss (after meta-adaptation)