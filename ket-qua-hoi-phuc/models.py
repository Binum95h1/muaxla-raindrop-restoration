import math
import torch
from torch import nn
from torch.nn import functional as F
from restormer_arch import Restormer
class DoubleConv(nn.Sequential):
    def __init__(self, incoming, outgoing):
        groups = math.gcd(8,outgoing)
        super().__init__(nn.Conv2d(incoming,outgoing,3,padding=1,bias=False),
                         nn.GroupNorm(groups,outgoing),nn.ReLU(inplace=True),
                         nn.Conv2d(outgoing,outgoing,3,padding=1,bias=False),
                         nn.GroupNorm(groups,outgoing),nn.ReLU(inplace=True))

class MaskUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        channels = [base*2**i for i in range(5)]
        self.encoder = nn.ModuleList([DoubleConv(3,channels[0])]+
                                     [DoubleConv(channels[i-1],channels[i]) for i in range(1,5)])
        self.reduce = nn.ModuleList([nn.Conv2d(channels[i],channels[i-1],1) for i in range(4,0,-1)])
        self.decoder = nn.ModuleList([DoubleConv(2*channels[i-1],channels[i-1]) for i in range(4,0,-1)])
        self.head = nn.Conv2d(base,1,1)
    def forward(self,x):
        height,width = x.shape[-2:]
        x = F.pad(x,(0,(-width)%16,0,(-height)%16),mode='replicate')
        skips = []
        for i,layer in enumerate(self.encoder):
            if i:
                x = F.max_pool2d(x,2)
            x = layer(x)
            if i<4:
                skips.append(x)
        for reduce,decode,skip in zip(self.reduce,self.decoder,reversed(skips)):
            x = reduce(F.interpolate(x,size=skip.shape[-2:],mode='nearest'))
            x = decode(torch.cat([x,skip],dim=1))
        return self.head(x)[...,:height,:width]

class DetailRefiner(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        self.enc1=DoubleConv(7,width);self.enc2=DoubleConv(width,width*2)
        self.center=DoubleConv(width*2,width*4)
        self.dec2=DoubleConv(width*6,width*2);self.dec1=DoubleConv(width*3,width)
        self.head=nn.Conv2d(width,3,3,padding=1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        a=self.enc1(x);b=self.enc2(F.avg_pool2d(a,2));c=self.center(F.avg_pool2d(b,2))
        d=self.dec2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        e=self.dec1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.head(e)

class RestorationSystem(nn.Module):
    def __init__(self,cfg,mode,mask_base=32):
        super().__init__();self.mode=mode
        a=cfg['architecture']
        self.backbone=Restormer(inp_channels=3 if mode=='rgb' else 4,dim=a['dim'],
            num_blocks=a['blocks'],num_refinement_blocks=a['refinement'],heads=a['heads'],
            ffn_expansion_factor=a['expansion'],bias=False,LayerNorm_type='WithBias')
        nn.init.zeros_(self.backbone.output.weight)
        self.detector=MaskUNet(mask_base) if mode!='rgb' else None
        self.refiner=DetailRefiner(a['detail_width']) if mode=='hybrid' else None
        self.freeze_detector()
    def freeze_detector(self):
        if self.detector is not None:
            self.detector.requires_grad_(False);self.detector.eval()
    def train(self,mode=True):
        super().train(mode);self.freeze_detector();return self
    def forward(self,x,return_aux=False):
        if self.detector is None:p=x[:,0:1]*0
        else:
            with torch.no_grad():p=self.detector(x).sigmoid()
        # Soft guidance extends context but does not hard-limit image corrections.
        guide=F.max_pool2d(p,9,stride=1,padding=4)
        coarse=self.backbone(x if self.mode=='rgb' else torch.cat([x,guide],1))
        final=coarse if self.refiner is None else coarse+self.refiner(torch.cat([x,coarse,guide],1))
        return (final,coarse,p) if return_aux else final
