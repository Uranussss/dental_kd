# coding = uft-8

import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_fn_kd(teacher_out, student_out, labels, t_s_map_dict, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """

    alpha = params.alpha
    T = params.temperature

    m = nn.Sigmoid()
    bce = nn.BCELoss()

    # print(F.log_softmax(student_out['logits']/T).shape)
    # print(F.log_softmax(student_out['logits']/T, dim=1).shape)
    # print(m(student_out['logits']/T).shape)

    # print(m(teacher_out['logits']/T).shape)

    # print(F.log_softmax(teacher_out['logits']/T).shape)
    # print(F.log_softmax(teacher_out['logits']/T, dim=1).shape)

    # former part
    # print(nn.KLDivLoss()(m(student_out['logits']/T), m(student_out['logits']/T)))
#     print(nn.KLDivLoss()(m(student_out['logits'] / T).log(), m(teacher_out['logits'] / T)) * (alpha * T * T))

    # later part
    # F.cross_entropy would get error: RuntimeError: CUDA error: device-side assert triggered
    # print(bce(student_out['final'], labels)* (1. - alpha)

#     print(bce(m(student_out['final']), labels) * (1. - alpha))

#     print(nn.KLDivLoss()(m(student_out['logits'] / T).log(), m(teacher_out['logits'] / T)) * (alpha * T * T) \
#           + bce(student_out['final'], labels) * (1. - alpha))
    # exit(0)

    # todo: sigmoid to activate logits
    return nn.KLDivLoss()(m(student_out['logits'] / T).log(), m(teacher_out['logits'] / T)) * (alpha * T * T) \
           + bce(student_out['final'], labels) * (1. - alpha)


def loss_fn_fitnet(teacher_out, student_out, labels, t_s_map_dict, params):
    alpha = params.alpha
    T = params.temperature
    beta = params.beta
    m = nn.Sigmoid()
    bce = nn.BCELoss()

    a, b = ([student_out[t_s_map_dict['student']], teacher_out[t_s_map_dict['teacher']]])
    a1, a2, a3, a4 = a.shape
    z = torch.zeros(a.size())
    if a.shape == b.shape:
        for i in range(0, a1):
            z[i] = a[i] - b[i]
    loss1 = 1 / 2 * (z.norm())
    # print(nn.KLDivLoss()(nn.Sigmoid(student_out['logits']/T, dim=1), nn.Sigmoid(teacher_out['logits']/T, dim=1)) * (alpha * T * T))
    # print(nn.BCELoss(student_out['logits'], labels)* (1. - alpha))
    # exit(0)

    #
    loss2 = nn.KLDivLoss()(m(student_out['logits'] / T).log(), m(teacher_out['logits'] / T)) * (alpha * T * T) \
            + bce(student_out['final'], labels) * (1. - alpha)

#     print(loss1.cuda() * beta)
#     print(loss2)
    # exit(0)
    return loss1.cuda() * beta + loss2 * 100


def loss_fn_at(teacher_out, student_out, labels, t_s_map_dict, params):
    m = nn.Sigmoid()
    bce = nn.BCELoss()

    # p = m(student_out['logits'] / params.temperature)
    # q = m(teacher_out['logits'] / params.temperature)
    # l_kl = F.kl_div(p, q, size_average=False) * (params.alpha * params.temperature * params.temperature)
    # l_ce = bce(student_out['logits'], labels)* (1. - params.alpha)
    # logits_loss = l_kl * params.alpha + l_ce * (1. - params.alpha)

    logits_loss = nn.KLDivLoss()(m(student_out['logits'] / params.temperature).log(),
                                 m(teacher_out['logits'] / params.temperature)) * (
                              params.alpha * params.temperature * params.temperature) \
                  + bce(student_out['final'], labels) * (1. - params.alpha)

    def at_loss(x, y):
        return (F.normalize(x.pow(2).mean(1).view(x.size(0), -1)) - F.normalize(
            y.pow(2).mean(1).view(y.size(0), -1))).pow(2).mean()

    g_s = student_out['feas']
    g_t = teacher_out['feas']
    loss_groups = at_loss(g_s, g_t)

#     print(logits_loss)
#     print(loss_groups)
#     print(params.gama * loss_groups)
#     print(logits_loss + params.gama * loss_groups)
    # exit(0)
    return logits_loss + params.gama * loss_groups


def loss_fn_KC(teacher_out, student_out, labels, t_s_map_dict, params):
    m = nn.Sigmoid()
    bce = nn.BCELoss()

    # p = m(student_out['logits'] / params.temperature)
    # q = m(teacher_out['logits'] / params.temperature)

    # l_kl = F.kl_div(p, q, size_average=False) * (params.alpha * params.temperature * params.temperature)
    # l_ce = bce(student_out['logits'], labels)* (1. - params.alpha)
    # logits_loss = l_kl * params.alpha + l_ce * (1. - params.alpha)

    logits_loss = nn.KLDivLoss()(m(student_out['logits'] / params.temperature).log(),
                                 m(teacher_out['logits'] / params.temperature)) * (
                              params.alpha * params.temperature * params.temperature) \
                  + bce(student_out['final'], labels) * (1. - params.alpha)

    # loss function for attention
    def at_loss(x, y):
        return (F.normalize(x.pow(2).mean(1).view(x.size(0), -1)) - F.normalize(
            y.pow(2).mean(1).view(y.size(0), -1))).pow(2).mean()

    g_s = student_out['feas']
    g_t = teacher_out['feas']
    loss_groups = at_loss(g_s, g_t)
    # print(logits_loss)
    # print(params.beta * loss_groups )
    # print(logits_loss + params.beta * loss_groups )
    # exit(0)

    return logits_loss + params.gama * loss_groups
