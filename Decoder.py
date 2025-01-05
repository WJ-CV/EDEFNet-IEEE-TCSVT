from decoders import *
from Encoder import Mnet
def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU()
    )
def convblock2(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )

def Aux_decoders():
    vat_decoder = [VATDecoder(4, 64, xi=1e-6, eps=2.0) for _ in range(2)]  # 1e-6 2.0   2
    # drop_decoder = [DropOutDecoder(4, 96, drop_rate=0.5, spatial_dropout=True) for _ in range(2)]  ## 0.5   True   6
    # cut_decoder = [CutOutDecoder(4, 96, erase=0.4) for _ in range(2)]  # # 0.4  6
    # context_m_decoder = [ContextMaskingDecoder(4, 96) for _ in range(1)]  # 2
    object_masking = [ObjectMaskingDecoder(4, 64) for _ in range(3)]  # 2
    feature_drop = [FeatureDropDecoder(4, 64) for _ in range(2)]  # 6
    feature_noise = [FeatureNoiseDecoder(4, 64, uniform_range=0.3) for _ in range(2)]  # # 0.3  6

    # Aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder,
    #                                *context_m_decoder, *object_masking, *feature_drop, *feature_noise])
    Aux_decoders = nn.ModuleList([*object_masking, *vat_decoder, *feature_drop, *feature_noise])
    return Aux_decoders

class Main_Decoder(nn.Module):
    def __init__(self):
        super(Main_Decoder, self).__init__()
        self.score_body = nn.Conv2d(64, 1, 1, 1, 0)
        self.score_edge = nn.Conv2d(64, 1, 1, 1, 0)
        self.score_1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.score = nn.Conv2d(64, 1, 1, 1, 0)
        # self.sig = nn.Sigmoid()

    def forward(self, fus_body0, fus_edge0, fusion_all):
        score_body = self.score_body(F.interpolate(fus_body0, (384, 384), mode='bilinear', align_corners=True))
        score_edge = self.score_edge(F.interpolate(fus_edge0, (384, 384), mode='bilinear', align_corners=True))
        score_1 = self.score_1(F.interpolate(fus_body0 + fus_edge0, (384, 384), mode='bilinear', align_corners=True))
        score = self.score(F.interpolate(fusion_all, (384, 384), mode='bilinear', align_corners=True))

        return score_body, score_edge, score_1, score

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.encoder = Mnet()
        self.decoder = Main_Decoder()

    def forward(self, rgb, t):
        fus_body0, fus_edge0, fusion_all = self.encoder( rgb, t)
        score_body, score_edge, score_1, score = self.decoder(fus_body0, fus_edge0, fusion_all)
        return score_body, score_edge, score_1, score
