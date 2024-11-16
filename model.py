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

class BertCon(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        :param bert_config: configuration for bert model
        """
        super(BertCon, self).__init__(bert_config)
        self.bert_config = bert_config
        self.bert = BertModel(bert_config)
        penultimate_hidden_size = bert_config.hidden_size
        self.shared_encoder = nn.Sequential(
                        nn.Linear(penultimate_hidden_size, penultimate_hidden_size // 2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(penultimate_hidden_size // 2, 192),
                    )

        self.dom_loss1 = CrossEntropyLoss()
        self.dom_cls = nn.Linear(192, bert_config.domain_number)
        # self.tem = bert_config.tem
        # self.tem = torch.tensor(0.05)
        self.tem = nn.Parameter(torch.tensor(0.05, requires_grad=True))
        # self.tem.requires_grad = True
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_labels=None,
            position_ids=None, head_mask=None, dom_labels=None, meg='train', epsilon=1e-5):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        hidden = outputs[0]  # Shape: (batch_size, seq_len, hidden_size)
        batch_num = hidden.shape[0]
        w = hidden[:, 0, :]  # CLS token representation
        h = self.shared_encoder(w)
    
        if meg == 'train':
            # Normalize original embeddings
            h = F.normalize(h, p=2, dim=1)
    
            # Step 1: Initialize random noise
            noise = torch.randn_like(h).detach()
            noise = Variable(noise, requires_grad=True)
    
            # Step 2: Compute perturbed embeddings
            for _ in range(1):  # Single-step adversarial training
                perturbed_h = h + epsilon * noise
                perturbed_h = F.normalize(perturbed_h, p=2, dim=1)
    
                # Compute similarity with adversarial embeddings
                sent_labels = sent_labels.unsqueeze(0).repeat(batch_num, 1).T
                rev_sent_labels = sent_labels.T
                similarity_mat = torch.matmul(perturbed_h, h.T)
    
                # Compute adversarial loss
                equal_mat = (sent_labels == rev_sent_labels).float()
                loss_adv = self.dom_loss1(similarity_mat, equal_mat.argmax(dim=-1))
    
                # Backpropagate adversarial gradients
                grad = torch.autograd.grad(loss_adv, noise, retain_graph=True)[0]
                noise = grad.detach()
    
            # Final perturbed embeddings
            h_adv = h + epsilon * noise
            h_adv = F.normalize(h_adv, p=2, dim=1)
    
            # Combine original and adversarial losses
            similarity_mat = torch.matmul(h, h_adv.T)
            eye = torch.eye(batch_num).to(h.device)
            equal_mat = (sent_labels == rev_sent_labels).float()
            a = ((equal_mat - eye) * similarity_mat).sum(dim=-1) + 1e-5
            b = ((torch.ones(batch_num, batch_num) - eye).to(h.device) * similarity_mat).sum(dim=-1) + 1e-5
    
            loss = -(torch.log(a / b)).mean(-1) + loss_adv
            return loss
    
        elif meg == 'source':
            return F.normalize(h, p=2, dim=1)

