from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

def E_Reg_loss(outputs, outputs_doc):
    loss = 0
    return loss


def E_CL(outputs, outputs_doc):
    loss = 0
    return loss


def doc_reweight_loss(outputs, outputs_doc):
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    unweighted_loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), outputs.labels.view(-1))
    weights = (F.normalize(outputs.encoder_last_hidden_state[:,0,:], dim=-1, p=2) * F.normalize(outputs_doc.encoder_last_hidden_state[:, 0, :], dim=-1, p=2)).sum(dim=-1).detach()
    loss = ( weights.reshape(-1, 1) * unweighted_loss.reshape(outputs.logits.shape[0], -1) ).sum()
    return loss



def loss_zoo(args, outputs, outputs_doc, prev_loss):
    if args.contrastive_variant == 'doc_Reweight':
        loss = doc_reweight_loss(outputs, outputs_doc)
        return loss