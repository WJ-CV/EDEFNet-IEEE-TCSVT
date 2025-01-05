import torch
from torch import nn
import torch.nn.functional as F
import cv2
import pvt_v2
from torch.nn import Conv2d, Parameter, Softmax
import numpy as np
import os

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU()
    )

def knn(x, k):  # N, c, num
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # N, num, num
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # N, 1, num
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=10, idx=None, dim9=False):  # N, c, num  k=10
    batch_size = x.size(0)  # N
    num_points = x.size(2)  # 100
    x = x.view(batch_size, -1, num_points)  # N, 128, 100
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # (N, 1, 1)
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="bchw"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["bchw", "bhwc"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "bhwc":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "bchw":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class CA(nn.Module):
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()

    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)

class ASPP(nn.Module): # deeplab
    def __init__(self, dim):
        super(ASPP, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(4 * dim, dim, 3,padding=1),nn.BatchNorm2d(dim),nn.PReLU())
        down_dim = dim // 4
        self.conv1 = nn.Sequential(nn.Conv2d(dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(dim, down_dim, kernel_size=3, dilation=2, padding=2),
                                   nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(dim, down_dim, kernel_size=3, dilation=4, padding=4),
                                   nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(dim, down_dim, kernel_size=3, dilation=6, padding=6),
                                   nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU())
        self.fuse = nn.Sequential(nn.Conv2d(5 * down_dim, dim, kernel_size=1), nn.BatchNorm2d(dim), nn.PReLU())

        self.squeeze_body_edge_high = SqueezeBodyEdge_high(dim)
    def forward(self, rgb, t):
        x = torch.cat((torch.cat((rgb, t), 1), 0.5 * (rgb + t), rgb * t), 1)
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.interpolate(self.conv5(F.adaptive_avg_pool2d(x, 2)), x.size()[2:], mode='bilinear', align_corners=True)
        glo_edge, glo_body = self.squeeze_body_edge_high(self.fuse(torch.cat((conv1, conv2, conv3,conv4, conv5), 1)))
        return glo_edge, glo_body

class Edge_CMF(nn.Module):
    def __init__(self, in_1, in_2):
        super(Edge_CMF, self).__init__()
        self.att = CA(in_1)
        self.in_channels = 128
        self.conv_down = convblock(in_1, 64, 3, 1, 1)
        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(self.in_channels * 2, self.in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_globalinfo = convblock(in_1, in_1, 1, 1, 0)
        self.conv_fus = convblock(128, in_1, 3, 1, 1)
        self.conv_out = convblock(2 * in_1, in_2, 3, 1, 1)

    def forward(self, rgb, t, global_info):
        att_rgb = self.conv_down(self.att(rgb))
        att_t = self.conv_down(self.att(t))
        n, c, h, w = att_rgb.size()   #N, 64, H, W,
        att_in = (torch.cat((att_rgb, att_t), 1)).view(n, 128, -1)   #N, 128, H, W  -  N, 128, HW

        graph_edge = get_graph_feature(att_in, k=10)     #N, 256, HW, 10
        graph_edge = self.conv1(graph_edge)             #N, 128, HW, 10
        graph_edge = graph_edge.max(dim=-1, keepdim=False)[0]        #N, 128, HW

        graph_edge = get_graph_feature(graph_edge, k=10)
        graph_edge = self.conv2(graph_edge)
        graph_edge = graph_edge.max(dim=-1, keepdim=False)[0]

        graph_edge = get_graph_feature(graph_edge, k=10)
        graph_edge = self.conv3(graph_edge)
        graph_edge = graph_edge.max(dim=-1, keepdim=False)[0]   #N, 128, HW

        graph_edge = graph_edge.view(n, 128, h, w)
        cmf_out = self.conv_fus(graph_edge)

        global_info = self.conv_globalinfo(F.interpolate(global_info,  (h, w), mode='bilinear', align_corners=True))
        return self.conv_out(torch.cat((cmf_out, global_info), 1))

class GMLP(nn.Module):
    def __init__(self, dim):
        super(GMLP, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim // 4)
        self.fc1 = nn.Linear(dim, dim // 4)
        self.fc3 = nn.Linear(dim // 4, dim)
        self.act = nn.GELU()
        self.conv3d = nn.Conv1d(dim // 4, dim // 4, 3, 1, int((3 - 1) / 2))
        self.conv5d = nn.Conv1d(dim // 4, dim // 4, 5, 1, int((5 - 1) / 2))
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, groups=dim),
            nn.Conv2d(dim, dim, 1),
        )
        nn.init.constant_(self.conv3d.bias, 1.0)
        nn.init.constant_(self.conv5d.bias, 1.0)
        self.conv_fus = convblock(2 * dim, dim, 3, 1, 1)

    def forward(self, rgb0, t0):
        b, c, h, w = rgb0.size()
        rgb = self.ln1(rgb0.view(b, c, -1).permute(0, 2, 1))
        t = self.ln1(t0.view(b, c, -1).permute(0, 2, 1))
        rgb = self.ln2(self.act(self.fc1(rgb))).permute(0, 2, 1)
        t = self.ln2(self.act(self.fc1(t))).permute(0, 2, 1)

        r3 = self.conv3d(rgb)
        t3 = self.conv3d(t)
        rt3 = (r3 + t3 + r3 * t3).permute(0, 2, 1)
        rt3 = self.dw(self.fc3(rt3).permute(0, 2, 1).view(b, c, h, w))

        r5 = self.conv5d(rgb)
        t5 = self.conv5d(t)
        rt5 = (r5 + t5 + r5 * t5).permute(0, 2, 1)
        rt5 = self.dw(self.fc3(rt5).permute(0, 2, 1).view(b, c, h, w))

        rt_out = torch.cat((rt3, rt5), 1)

        return self.conv_fus(rt_out)

class Body_CMF(nn.Module):
    def __init__(self, in_1, in_2):
        super(Body_CMF, self).__init__()
        self.gmlp = GMLP(in_1)
        self.casa1 = CA(in_1)
        self.conv_globalinfo = convblock(in_1, in_1, 1, 1, 0)
        self.conv_out = convblock(2 * in_1, in_2, 3, 1, 1)
        self.rt_fus = nn.Sequential(
            nn.Conv2d(in_1, in_1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, cu_p, cu_s, global_info):
        cur_size = cu_p.size()[2:]
        cu_p = self.casa1(cu_p)
        cu_s = self.casa1(cu_s)
        cm_fus = self.gmlp(cu_p, cu_s)
        global_info = self.conv_globalinfo(F.interpolate(global_info, cur_size, mode='bilinear', align_corners=True))

        cross_cat = cm_fus + torch.add(cm_fus, torch.mul(cm_fus, self.rt_fus(global_info)))
        global_info = global_info + torch.add(global_info, torch.mul(global_info, self.rt_fus(cm_fus)))
        return self.conv_out(torch.cat((cm_fus, global_info), 1))

class SqueezeBodyEdge_low(nn.Module):
    def __init__(self, inplane):
        super(SqueezeBodyEdge_low, self).__init__()
        self.conv1 = convblock(inplane, inplane, 3, 1, 1)
        self.conv2 = convblock(inplane, inplane // 4, 3, 1, 1)
        self.conv3 = nn.Conv2d(inplane // 4, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        boundary = self.edge_canny(x)
        out = self.conv1(boundary)
        out = self.conv2(out)
        out = self.sigmoid(self.conv3(out))
        canny_edge = x * out.expand_as(x)
        canny_body = x - canny_edge
        return canny_edge, canny_body

    def edge_canny(self, inp):
        x_size = inp.size()
        im_arr = inp.detach().cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).float().to(inp.device)
        return canny.repeat(1, x_size[1], 1, 1)

class SqueezeBodyEdge_high(nn.Module):
    def __init__(self, inplane):
        super(SqueezeBodyEdge_high, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            nn.BatchNorm2d(inplane),
            nn.ReLU(inplace=True)
        )
        self.flow_make = nn.Conv2d(inplane *2 , 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        flow_body = self.flow_warp(x, flow, size)
        flow_edge = x - flow_body

        return flow_edge, flow_body

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new                        1,h               h,1              h, w
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)  # h, w, 2
                       #  n, h, w, 2
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

class SqueezeBodyEdge(nn.Module):
    def __init__(self):
        super(SqueezeBodyEdge, self).__init__()
        self.Squeeze_low1 = SqueezeBodyEdge_low(64)
        self.Squeeze_low2 = SqueezeBodyEdge_low(128)
        self.Squeeze_high3 = SqueezeBodyEdge_high(320)
        self.Squeeze_high4 = SqueezeBodyEdge_high(512)

    def forward(self, rgb_f, t_f):
        rgb_edge_1, rgb_body_1 = self.Squeeze_low1(rgb_f[0])
        rgb_edge_2, rgb_body_2 = self.Squeeze_low2(rgb_f[1])
        rgb_edge_3, rgb_body_3 = self.Squeeze_high3(rgb_f[2])
        rgb_edge_4, rgb_body_4 = self.Squeeze_high4(rgb_f[3])

        t_edge_1, t_body_1 = self.Squeeze_low1(t_f[0])
        t_edge_2, t_body_2 = self.Squeeze_low2(t_f[1])
        t_edge_3, t_body_3 = self.Squeeze_high3(t_f[2])
        t_edge_4, t_body_4 = self.Squeeze_high4(t_f[3])

        rgb_edge = [rgb_edge_1, rgb_edge_2, rgb_edge_3, rgb_edge_4]
        rgb_body = [rgb_body_1, rgb_body_2, rgb_body_3, rgb_body_4]
        t_edge = [t_edge_1, t_edge_2, t_edge_3, t_edge_4]
        t_body = [t_body_1, t_body_2, t_body_3, t_body_4]
        return rgb_edge, rgb_body, t_edge, t_body

class BodyEdge_CLF(nn.Module):
    def __init__(self):
        super(BodyEdge_CLF, self).__init__()
        self.sig = nn.Sigmoid()
        self.up_4 = convblock(512, 320, 3, 1, 1)
        self.up_3 = convblock(320, 128, 3, 1, 1)
        self.up_2 = convblock(128, 64, 3, 1, 1)
        self.up_1 = convblock(64, 64, 1, 1, 0)

        self.conv3 = convblock(128, 64, 3, 1, 1)
        self.conv1 = convblock(64, 64, 1, 1, 0)

        self.conv_fus = convblock(64 * 4, 64, 3, 1, 1)

    def forward(self, fus_0, fus_1, fus_2, fus_3, fus_g):
        s3 = fus_3 + torch.mul(fus_3, self.sig(self.up_4(fus_g)))
        up_3 = self.up_3(F.interpolate(s3, fus_2.size()[2:], mode='bilinear', align_corners=True))
        s2 = fus_2 + torch.mul(fus_2, self.sig(up_3))
        up_2 = self.up_2(F.interpolate(s2, fus_1.size()[2:], mode='bilinear', align_corners=True))
        s1 = fus_1 + torch.mul(fus_1, self.sig(up_2))
        up_1 = self.up_1(F.interpolate(s1, fus_0.size()[2:], mode='bilinear', align_corners=True))
        s0 = fus_0 + torch.mul(fus_0, self.sig(up_1))

        d3_up = self.conv3(F.interpolate(up_3, s0.size()[2:], mode='bilinear', align_corners=True))
        d2_up = self.conv1(F.interpolate(up_2, s0.size()[2:], mode='bilinear', align_corners=True))
        d1_up = self.conv1(up_1)
        d0_up = self.conv1(s0)

        clf_out = self.conv_fus(torch.cat((d3_up, d2_up, d1_up, d0_up), 1))
        return clf_out
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.squeeze_body_edge = SqueezeBodyEdge()
        self.glo = ASPP(512)
        self.sig = nn.Sigmoid()
        self.body_fus3 = Body_CMF(512, 320)
        self.body_fus2 = Body_CMF(320, 128)
        self.body_fus1 = Body_CMF(128, 64)
        self.body_fus0 = Body_CMF(64, 64)

        self.edge_fus3 = Edge_CMF(512, 320)
        self.edge_fus2 = Edge_CMF(320, 128)
        self.edge_fus1 = Edge_CMF(128, 64)
        self.edge_fus0 = Edge_CMF(64, 64)

        self.bodyedge_clf = BodyEdge_CLF()

    def forward(self, rgb_f, t_f):
        rgb_edge, rgb_body, t_edge, t_body = self.squeeze_body_edge(rgb_f, t_f)
        glo_edge, glo_body = self.glo(rgb_f[3], t_f[3])
        fus_body3 = self.body_fus3(rgb_body[3], t_body[3], glo_body)
        fus_body2 = self.body_fus2(rgb_body[2], t_body[2], fus_body3)
        fus_body1 = self.body_fus1(rgb_body[1], t_body[1], fus_body2)
        fus_body0 = self.body_fus0(rgb_body[0], t_body[0], fus_body1)

        fus_edge3 = self.edge_fus3(rgb_edge[3], t_edge[3], glo_edge)
        fus_edge2 = self.edge_fus2(rgb_edge[2], t_edge[2], fus_edge3)
        fus_edge1 = self.edge_fus1(rgb_edge[1], t_edge[1], fus_edge2)
        fus_edge0 = self.edge_fus0(rgb_edge[0], t_edge[0], fus_edge1)

        fusion_all = self.bodyedge_clf(fus_edge0 + fus_body0, fus_edge1 + fus_body1, fus_edge2 + fus_body2, fus_edge3 + fus_body3, glo_edge + glo_body)

        return fus_body0, fus_edge0, fusion_all

class Transformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.encoder = getattr(pvt_v2, backbone)()
        if pretrained:
            checkpoint = torch.load('../pvt_v2_b3.pth', map_location='cpu')
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = self.encoder.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            self.encoder.load_state_dict(checkpoint_model, strict=False)

def Encoder():
    model = Transformer('pvt_v2_b3', pretrained=True)
    return model

class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        model = Encoder()
        self.rgb_net = model.encoder
        self.t_net = model.encoder
        self.decoder = Decoder()

    def forward(self, rgb, t):
        rgb_f = []
        t_f = []
        rgb_f = self.rgb_net(rgb)
        t_f = self.t_net(t)

        fus_body0, fus_edge0, fusion_all = self.decoder(rgb_f, t_f)

        return fus_body0, fus_edge0, fusion_all