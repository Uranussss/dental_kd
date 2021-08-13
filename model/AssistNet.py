# coding = uft-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class KD(nn.Module):
    def __init__(self, teacher_out, student_out, t_s_map_dict = {}):
        super(KD, self).__init__()

    def forward(self, teacher_out, student_out, t_s_map_dict = {}):
        return teacher_out, student_out


class FitnetConvReg(nn.Module):
    """
    Convolutional regression for FitNet
    """

    def __init__(self, teacher_out, student_out, t_s_map_dict, use_relu=True):
        super(FitnetConvReg, self).__init__()
        s_N, s_C, s_H, s_W = student_out[t_s_map_dict['student']].shape
        t_N, t_C, t_H, t_W = teacher_out[t_s_map_dict['teacher']].shape
        # print(student_out[t_s_map_dict['student']].shape)
        # print(teacher_out[t_s_map_dict['teacher']].shape)
        # print(s_H)
        # print(t_H)
        # exit(0)
        self.use_relu = use_relu
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        #  TODO without bn
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)
        self.t_s_map_dict = t_s_map_dict

    def forward(self, teacher_out, student_out, t_s_map_dict):
        x = student_out[self.t_s_map_dict['student']]
        x = self.conv(x)
        if self.use_relu:
            student_out[self.t_s_map_dict['student']] = self.relu(x)
        return teacher_out, student_out




class AT(nn.Module):
    def __init__(self, teacher_out, student_out, t_s_map_dict):
        super(AT, self).__init__()
        self.t_s_map_dict = t_s_map_dict

    def forward(self, teacher_out, student_out, t_s_map_dict):
        # student_H, teacher_H = student_out[self.t_s_map_dict['student']].shape[2], teacher_out[self.t_s_map_dict['teacher']].shape[2]
        # if(student_H > teacher_H):
        #     student_out[self.t_s_map_dict['student']] = F.adaptive_avg_pool2d(student_out[self.t_s_map_dict['teacher']], (teacher_H, teacher_H))
        # elif(student_H < teacher_H):
        #     teacher_out[self.t_s_map_dict['student']] = F.adaptive_avg_pool2d(teacher_out[self.t_s_map_dict['teacher']], (student_H, student_H))
        return teacher_out, student_out


class KC(nn.Module):
    def __init__(self, teacher_out, student_out, t_s_map_dict = {}, fix_p=False, instance_bn=True, kernel_size=3, padding=1,
                 layers=2, bn=False, affine=False, state=0):
        super(KC, self).__init__()
        s_N, s_C, s_H, s_W = student_out[t_s_map_dict['student']].shape
        input_size = output_size = (s_C, s_H, s_W)
        self.input_size = input_size
        self.output_size = output_size
        inCh = outCh = input_size[0]
        self.affine = affine
        self.state = state
        self.layers = layers
        self.bn = bn
        self.t_s_map_dict = t_s_map_dict
        self.nonLinearLayers_p_pre = nn.Parameter(torch.tensor([0.0, 0.0]).cuda(), requires_grad=(not fix_p))
        self.nonLinearLayers_p = self.get_p()
        self.instance_bn = instance_bn
        if self.bn:
            if not self.instance_bn:
                self.linearLayers_bn = nn.BatchNorm2d(inCh, affine=self.affine, track_running_stats=False)
            else:
                self.linearLayers_bn = nn.InstanceNorm2d(inCh, affine=self.affine, track_running_stats=False)
        linearLayers_conv = []
        nonLinearLayers_ReLU = []
        for x in range(self.layers):
            linearLayers_conv += [nn.Conv2d(inCh, outCh, kernel_size=kernel_size, padding=padding, bias=False)]
            nonLinearLayers_ReLU += [nn.ReLU(inplace=True)]
        self.linearLayers_conv = nn.ModuleList(linearLayers_conv)
        self.nonLinearLayers_ReLU = nn.ModuleList(nonLinearLayers_ReLU)

        if not instance_bn:
            self.nonLinearLayers_norm = nn.Parameter(torch.ones(self.layers, self.output_size[0]),
                                                     requires_grad=False)
            self.running_times = nn.Parameter(torch.zeros(self.layers, dtype=torch.long), requires_grad=False)

        else:
            self.nonLinearLayers_norm = torch.ones(self.layers - 1, 1, self.output_size[0]).cuda()
            # self.nonLinearLayers_norm = torch.ones(self.layers - 1).cuda(self.gpu_id)

    def get_p(self):
        return nn.Sigmoid()(self.nonLinearLayers_p_pre)

    def forward(self, teacher_out, student_out, t_s_map_dict = {}):
        self.nonLinearLayers_p = self.get_p()
        x = student_out['feas']
        if self.bn:
            x = self.linearLayers_bn(x)
        else:
            x = self.my_bn(self.layers - 1, x)

        out = self.linear(self.state, x, torch.zeros_like(x))
        for i in range(1 + self.state, self.layers):
            out = self.nonLinear(i - 1, out)
            out = self.linear(i, x, out)
        student_out['feas'] = out
        return teacher_out, student_out

    def my_bn(self, i, out, momentum=0.1, eps=1e-5, rec=False, yn=False):
        if not self.instance_bn:
            if self.training:
                a = out.transpose(0, 1).reshape([out.shape[1], -1]).var(-1).sqrt() + eps
                if self.running_times[i] == 0:
                    self.nonLinearLayers_norm[i] = a
                else:
                    self.nonLinearLayers_norm[i] = (1 - momentum) * self.nonLinearLayers_norm[i] + momentum * a
                self.running_times[i] += 1
                a_ = a.reshape(1, out.shape[1], 1, 1)
            else:
                a_ = self.nonLinearLayers_norm[i].reshape(1, out.shape[1], 1, 1)

            a_ = a_.repeat(out.shape[0], 1, out.shape[2], out.shape[3])
            out = out / a_
            return out
        else:
            if not yn:

                # out = student_out['feas']
                # print(out.shape)
                # print(*out.shape[:-2])

                # print(self.output_size)
                # print(self.output_size[1])
                # print(self.output_size[2])
                # print(self.output_size[1]*self.output_size[2])


                a = out.data.reshape([*out.shape[:-2], self.output_size[1]*self.output_size[2]])
                # a = out.data.reshape([*out.shape[:-3],-1]).var(-1).sqrt() \
                #     + eps
                if a.size()[-1] == 1:
                    a = torch.ones_like(a)
                    if rec:
                        self.nonLinearLayers_norm[i] = a.reshape([*a.shape[:-1]])
                else:
                    a = a.var(-1).sqrt() + eps
                    if rec:
                        self.nonLinearLayers_norm[i] = a.squeeze(0)
            else:
                a = self.nonLinearLayers_norm[i]
            a = a.reshape([*out.shape[:-2], 1, 1])
            # a = a.reshape([out.shape[0],1, 1, 1])
            out = out / a
            return out

    def nonLinear(self, i, out, rec=False):
        out = self.my_bn(i, out, rec=rec)
        out = self.nonLinearLayers_ReLU[i](out)
        if rec:
            self.nonLinearLayersRecord[i] = torch.gt(out, 0)#.reshape(self.input_size)
        out = self.nonLinearLayers_p[i] * out
        return out

    def linear(self, i, x, out):
        out = x + out
        out = self.linearLayers_conv[i](out)
        return out