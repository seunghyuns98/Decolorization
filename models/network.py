import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from models.basis import ResBlk, conv1x1


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Sequential(conv(3, 64), nn.ReLU())
        self.conv1_2 = nn.Sequential(conv(64, 64), nn.ReLU())
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv2_1 = nn.Sequential(conv(64, 128), nn.ReLU())
        self.conv2_2 = nn.Sequential(conv(128, 128), nn.ReLU())
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv3_1 = nn.Sequential(conv(128, 256), nn.ReLU())
        self.conv3_2 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.conv3_3 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.conv3_4 = nn.Sequential(conv(256, 256), nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.conv4_1 = nn.Sequential(conv(256, 512), nn.ReLU())
        self.conv4_2 = nn.Sequential(conv(512, 512), nn.ReLU())

    def load_model(self, model_file):
        vgg19_dict = self.state_dict()
        pretrained_dict = torch.load(model_file)

        vgg19_keys = vgg19_dict.keys()
        pretrained_keys = pretrained_dict.keys()
        for k, pk in zip(vgg19_keys, pretrained_keys):
            vgg19_dict[k] = pretrained_dict[pk]
        self.load_state_dict(vgg19_dict)

    def forward(self, input_images):
        feature = {}
        feature['conv1_1'] = self.conv1_1(input_images)
        feature['conv1_2'] = self.conv1_2(feature['conv1_1'])
        feature['pool1'] = self.pool1(feature['conv1_2'])
        feature['conv2_1'] = self.conv2_1(feature['pool1'])
        feature['conv2_2'] = self.conv2_2(feature['conv2_1'])
        feature['pool2'] = self.pool2(feature['conv2_2'])
        feature['conv3_1'] = self.conv3_1(feature['pool2'])
        feature['conv3_2'] = self.conv3_2(feature['conv3_1'])
        feature['conv3_3'] = self.conv3_3(feature['conv3_2'])
        feature['conv3_4'] = self.conv3_4(feature['conv3_3'])
        feature['pool3'] = self.pool3(feature['conv3_4'])
        feature['conv4_1'] = self.conv4_1(feature['pool3'])
        x = self.conv4_2(feature['conv4_1'])

        return x


class Resnet(nn.Module):
    def __init__(self, dim):
        super(Resnet, self).__init__()

        self.layers = [4, 4, 4, 4]
        self.planes = [64, 128, 256, 512]

        self.num_layers = sum(self.layers)
        self.inplanes = self.planes[0]
        self.conv0 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, self.planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.actv = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ResBlk, self.planes[0], self.layers[0])
        self.layer2 = self._make_layer(ResBlk, self.planes[1], self.layers[1], stride=2)
        self.layer3 = self._make_layer(ResBlk, self.planes[2], self.layers[2], stride=2)
        self.layer4 = self._make_layer(ResBlk, self.planes[3], self.layers[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.bias2 = nn.Parameter(torch.zeros(1))

        self.fc = nn.Linear(self.planes[3], dim)

        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, ResBlk):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(
                    2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(
                        2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = conv1x1(self.inplanes, planes, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.actv(x + self.bias1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_x = self.gap(x)
        max_x = self.gmp(x)

        x = (max_x + avg_x).flatten(1)
        x = self.fc(x + self.bias2)

        x = F.normalize(x, p=2, dim=1)

        return x


class Mapping(nn.Module):
    def __init__(self, in_dim):
        super(Mapping, self).__init__()
        self.layers = 2
        self.planes = 512

        self.mlp_in = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU()
        )

        self.mlp_out = nn.ModuleList()
        for _ in range(self.layers):
            self.mlp_out.append(
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.PReLU(),
                    nn.Linear(512, self.planes * 2),
                    nn.Sigmoid()
                )
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp_in(x)

        s_list = []
        for i in range(self.layers):
            out = self.mlp_out[i](x).view(x.size(0), -1, 1, 1)
            s_list.append(list(torch.chunk(out, chunks=2, dim=1)))

        return s_list


class DualAdainResBlk(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DualAdainResBlk, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.actv1 = nn.PReLU()
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bias2a = nn.Parameter(torch.zeros(1))
        self.actv2 = nn.PReLU()
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.scale = nn.Parameter(torch.ones(1))

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, sa, sb):
        identity = x

        out = self.actv1(x + self.bias1a)

        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out + self.bias1b)
        alpha = sb[0] / sa[0]
        beta = sb[1] - sa[1] * alpha
        out = out * alpha + beta

        out = self.actv2(out + self.bias2a)
        out = self.conv2(out + self.bias2b)

        out = out * self.scale

        out += identity

        return out


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):
        output = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        return output


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.PReLU,
                 batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Header(nn.Module):
    def __init__(self, dim, cN):
        super(Header, self).__init__()
        self.fc = Parameter(torch.Tensor(dim, cN))
        torch.nn.init.xavier_normal_(self.fc)

    def forward(self, input):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)

        return simInd


class Content(nn.Module):

    def __init__(self, dim, obj):
        super(Content, self).__init__()
        self.encoder = Resnet(dim)
        self.cproxy = Header(dim, obj)

    def forward(self, x):
        x = self.encoder(F.adaptive_avg_pool2d(x, (224, 224)))
        out = self.cproxy(x)

        return x, out


class Style(nn.Module):

    def __init__(self, dim, style):
        super(Style, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, 3, 1, 1)
        self.encoder = VGG19()
        self.encoder.load_model('/seunghyun/vgg19-dcbb9e9d.pth')
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(512, 512)
        self.cproxy = Header(dim, style)

    def forward(self, x):

        x = self.encoder(self.conv1(F.adaptive_avg_pool2d(x, (224, 224))))
        avg_x = self.gap(x)
        max_x = self.gmp(x)

        x = (max_x + avg_x).flatten(1)
        x = self.fc(x + self.bias2)

        x = F.normalize(x, p=2, dim=1)

        out = self.cproxy(x)

        return x, out


class Mixer(nn.Module):
    def __init__(self, dim):
        super(Mixer, self).__init__()
        self.fc1 = Parameter(torch.Tensor(dim * 2, dim))
        self.fc2 = Parameter(torch.Tensor(dim, dim))
        self.fc3 = Parameter(torch.Tensor(dim, dim))
        torch.nn.init.xavier_normal_(self.fc1)
        torch.nn.init.xavier_normal_(self.fc2)
        torch.nn.init.xavier_normal_(self.fc3)

    def forward(self, x):
        x = x.matmul(self.fc1)
        x = x.matmul(self.fc2)
        x = x.matmul(self.fc3)
        x = F.normalize(x, p=2, dim=1)
        return x


class Encoder(nn.Module):
    def __init__(self, dim, style, obj):
        super(Encoder, self).__init__()
        self.style_head = Style(dim, style)
        self.object_head = Content(dim, obj)
        self.mixer = Mixer(dim)

    def forward(self, x):
        style_proxy, style_id = self.style_head(x)
        obj_proxy, obj_id = self.object_head(x)
        x = torch.cat([style_proxy, obj_proxy], dim=1)
        x = self.mixer(x)
        return x, style_id, obj_id


class Bilateral_grid(nn.Module):
    def __init__(self, dims):
        super(Bilateral_grid_res, self).__init__()
        self.layers = [2, 2, 2, 2]
        self.planes = [64, 128, 256, 512]
        self.dims = dims

        self.num_layers = sum(self.layers)
        self.inplanes = self.planes[0]

        self.conv1 = nn.Conv2d(3, self.planes[0], kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.actv = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(*self._make_layer(ResBlk, self.planes[0], self.layers[0]))
        self.layer2 = nn.Sequential(*self._make_layer(ResBlk, self.planes[1], self.layers[1]))
        self.layer3 = nn.Sequential(*self._make_layer(ResBlk, self.planes[2], self.layers[2], stride=2))
        self.layer4 = self._make_layer(DualAdainResBlk, self.planes[3], self.layers[3], stride=2)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(256, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.L1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.L1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.L2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, padding_mode='reflect')
        self.L2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.L3_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, padding_mode='reflect')
        self.L3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.G1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, padding_mode='reflect')
        self.G2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, padding_mode='reflect')
        self.G2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, padding_mode='reflect')
        self.G2_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2, padding_mode='reflect')
        self.G3 = nn.Linear(1024, 256)
        self.G4 = nn.Linear(256, 128)
        self.G5 = nn.Linear(128, 64)
        self.G6 = nn.Linear(64, 64)
        self.F = nn.Conv2d(256, 64, 1, padding=0)
        self.T = nn.Conv2d(64, 32, 7, padding=3, padding_mode='reflect')
        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, ResBlk) or isinstance(m, DualAdainResBlk):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(
                    2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(
                        2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = conv1x1(self.inplanes, planes, stride)

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def forward(self, x, sa, sb):
        x = self.conv1(x)
        x = self.actv(x + self.bias1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.layers[3]):
            x = self.layer4[i](x, sa[i], sb[i])

        x = self.actv(self.conv2(x))
        x = self.actv(self.conv3(x))

        L = self.actv(self.L1_1(x))
        L = self.actv(self.L1_2(L))
        L2 = self.actv(self.L2_1(L))
        L2 = self.actv(self.L2_2(L2))
        L3 = self.actv(self.L3_1(L2))
        L3 = self.actv(self.L3_2(L3))

        G = self.actv(self.G1(x))
        G = self.actv(self.G2(G))
        G = self.actv(self.G2_2(G))
        G = self.actv(self.G2_3(G))
        G = G.reshape((G.shape[0], -1))
        G = self.actv(self.G3(G))
        G = self.actv(self.G4(G))
        G = self.actv(self.G5(G))
        G = self.actv(self.G6(G))
        G = G.reshape(G.shape + (1, 1)).expand(G.shape + (64, 64))
        L2 = F.interpolate(L2, (64, 64), mode='bilinear')
        L3 = F.interpolate(L3, (64, 64), mode='bilinear')

        f = torch.cat((L, L2, L3, G), dim=1)
        f = self.actv(self.F(f))
        f = self.T(f)
        return f


class RGB2GRAY(nn.Module):
    def __init__(self, dim):
        super(RGB2GRAY, self).__init__()
        self.mapping = Mapping(dim)
        self.bilateral_grid = Bilateral_grid(dim)
        self.guide = GuideNN()
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, cont, cont_feat, proxy, gray_img, ori_img=None):
        cont_feat = self.mapping(cont_feat)
        style_feat = self.mapping(proxy)
        b_grid = self.bilateral_grid(cont, cont_feat, style_feat)
        b_grid = b_grid.reshape(b_grid.shape[0], 4, -1, b_grid.shape[-2], b_grid.shape[-1])

        slice_coeffs = self.slice(b_grid, gray_img)
        if ori_img is not None:
            slice_coeffs = F.interpolate(slice_coeffs, (ori_img.shape[2], ori_img.shape[3]), mode='bilinear')
            out = self.apply_coeffs(slice_coeffs, ori_img)
        else:
            out = self.apply_coeffs(slice_coeffs, cont)

        return out, slice_coeffs

