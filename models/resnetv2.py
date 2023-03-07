# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.route import *

class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW."""
    if conv_weights.ndim == 4:
        conv_weights = conv_weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        out = self.relu(self.gn1(x))

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)

        # Unit's branch
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual

    def load_from(self, weights, prefix=''):
        convname = 'standardized_conv2d'
        with torch.no_grad():
            self.conv1.weight.copy_(tf2th(weights[f'{prefix}a/{convname}/kernel']))
            self.conv2.weight.copy_(tf2th(weights[f'{prefix}b/{convname}/kernel']))
            self.conv3.weight.copy_(tf2th(weights[f'{prefix}c/{convname}/kernel']))
            self.gn1.weight.copy_(tf2th(weights[f'{prefix}a/group_norm/gamma']))
            self.gn2.weight.copy_(tf2th(weights[f'{prefix}b/group_norm/gamma']))
            self.gn3.weight.copy_(tf2th(weights[f'{prefix}c/group_norm/gamma']))
            self.gn1.bias.copy_(tf2th(weights[f'{prefix}a/group_norm/beta']))
            self.gn2.bias.copy_(tf2th(weights[f'{prefix}b/group_norm/beta']))
            self.gn3.bias.copy_(tf2th(weights[f'{prefix}c/group_norm/beta']))
            if hasattr(self, 'downsample'):
                w = weights[f'{prefix}a/proj/{convname}/kernel']
                self.downsample.weight.copy_(tf2th(w))


class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        n_b = x.data.size(0)
        x = x.view(n_b, -1)
        return x

    def __repr__(self):
        return 'view(nB, -1)'

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False, fix_backbone=False, num_block_open=0, k=None):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        self.fix = fix_backbone
        # self.num_block_open = num_block_open

        if num_block_open == 1:
            self.fix_parts = ['root', 'block1', 'block2', 'block3']
            self.fix_gn1 = 'block4'
        elif num_block_open == 2:
            self.fix_parts = ['root', 'block1', 'block2']
            self.fix_gn1 = 'block3'
        elif num_block_open == 3:
            self.fix_parts = ['root', 'block1']
            self.fix_gn1 = 'block2'
        elif num_block_open == 4:
            self.fix_parts = ['root']
            self.fix_gn1 = 'block1'
        else:
            self.fix_parts = None
            self.fix_gn1 = None

        # The following will be unreadable if we split lines.
        # pylint: disable=line-too-long
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, 64 * wf, kernel_size=7, stride=2, padding=3, bias=False)),
            ('pad', nn.ConstantPad2d(1, 0)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            # The following is subtly not the same!
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=64 * wf, cout=256 * wf, cmid=64 * wf))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=256 * wf, cout=256 * wf, cmid=64 * wf)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=256 * wf, cout=512 * wf, cmid=128 * wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=512 * wf, cout=512 * wf, cmid=128 * wf)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=512 * wf, cout=1024 * wf, cmid=256 * wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=1024 * wf, cout=1024 * wf, cmid=256 * wf)) for i in
                 range(2, block_units[2] + 1)],
            ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=1024 * wf, cout=2048 * wf, cmid=512 * wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=2048 * wf, cout=2048 * wf, cmid=512 * wf)) for i in
                 range(2, block_units[3] + 1)],
            ))),
        ]))
        # pylint: enable=line-too-long

        self.zero_head = zero_head

        self.before_head = nn.Sequential(OrderedDict([
            ('gn', nn.GroupNorm(32, 2048 * wf)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
        ]))

        if k is None:
            self.head = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(2048 * wf, head_size, kernel_size=1, bias=True)),
            ]))
        else:
            self.head = nn.Sequential(OrderedDict([
                ('fcview', FCView()),
                ('conv', RouteDICE(2048 * wf, head_size, bias=True, topk=k, conv1x1=True, info=np.load('cache/imagenet2012_subset_BiT-S-R101x1_wa_stat.npy')))
            ]))

        if self.fix:
            for param in self.root.parameters():
                param.requires_grad = False
            for param in self.body.parameters():
                param.requires_grad = False
            for param in self.before_head.parameters():
                param.requires_grad = False
        elif self.fix_parts:
            if 'root' in self.fix_parts:
                print('Fixing root......')
                for param in self.root.parameters():
                    param.requires_grad = False
            for bname, block in self.body.named_children():
                if bname in self.fix_parts:
                    print('Fixing {}......'.format(bname))
                    for param in block.parameters():
                        param.requires_grad = False
                elif bname == self.fix_gn1:
                    print('Fixing gn1 in {}......'.format(bname))
                    for param in block.unit01.gn1.parameters():
                        param.requires_grad = False

    def feature_list(self, x):
        out_list = []

        out = self.root(x)
        out_list.append(out)
        out = self.body.block1(out)
        out_list.append(out)
        out = self.body.block2(out)
        out_list.append(out)
        out = self.body.block3(out)
        out_list.append(out)
        out = self.body.block4(out)
        out_list.append(out)

        out = self.head(self.before_head(out))

        return out[..., 0, 0], out_list

    def intermediate_forward(self, x, layer_index):
        out = self.root(x)
        if layer_index == 1:
            out = self.body.block1(out)
        elif layer_index == 2:
            out = self.body.block1(out)
            out = self.body.block2(out)
        elif layer_index == 3:
            out = self.body.block1(out)
            out = self.body.block2(out)
            out = self.body.block3(out)
        elif layer_index == 4:
            out = self.body.block1(out)
            out = self.body.block2(out)
            out = self.body.block3(out)
            out = self.body.block4(out)

        return out

    def forward(self, x):
        if self.fix:
            with torch.no_grad():
                x = self.before_head(self.body(self.root(x)))
            x = self.head(x)

        elif self.fix_parts:
            if 'root' in self.fix_parts:
                with torch.no_grad():
                    x = self.root(x)
            else:
                x = self.root(x)

            for bname, block in self.body.named_children():
                if bname in self.fix_parts:
                    with torch.no_grad():
                        x = block(x)
                else:
                    x = block(x)
            x = self.head(self.before_head(x))

        else:
            x = self.head(self.before_head(self.body(self.root(x))))

        # assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x.squeeze()

    def load_state_dict_custom(self, state_dict):
        state_dict_new = {}
        for k, v in state_dict.items():
            state_dict_new[k[len("module."):]] = v
        self.load_state_dict(state_dict_new, strict=True)

    def load_from(self, weights, prefix='resnet/'):
        with torch.no_grad():
            self.root.conv.weight.copy_(
                tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))  # pylint: disable=line-too-long
            self.before_head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
            self.before_head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
            if self.zero_head:
                nn.init.zeros_(self.head.conv.weight)
                nn.init.zeros_(self.head.conv.bias)
            else:
                self.head.conv.weight.copy_(
                    tf2th(weights[f'{prefix}head/conv2d/kernel']))  # pylint: disable=line-too-long
                self.head.conv.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.head.train(mode)
        if self.fix:
            self.root.eval()
            self.body.eval()
            self.before_head.eval()
        elif self.fix_parts:
            if 'root' in self.fix_parts:
                self.root.eval()
            else:
                self.root.train(mode)
            for bname, block in self.body.named_children():
                if bname in self.fix_parts:
                    block.eval()
                elif bname == self.fix_gn1:
                    block.train(mode)
                    block.unit01.gn1.eval()
                else:
                    block.train(mode)
            self.before_head.train(mode)
        else:
            self.root.train(mode)
            self.body.train(mode)
            self.before_head.train(mode)
        return self


KNOWN_MODELS = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])
