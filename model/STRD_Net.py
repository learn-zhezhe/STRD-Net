import torch
from SoftPool import SoftPool2d
from model.resnet50 import resnet50
import torch.nn as nn
import torch.nn.functional as F
from model.swin_transformer import swin_transformer
from fightingcv_attention.attention.CBAM import CBAMBlock
from model.deform_conv_v2 import DeformConv2d
import time
from thop import profile


class EASPP(nn.Module):
    """
    EASPP feature extraction module
    Utilize dilated convolutions with different dilation rates for feature extraction
    Use DeformConv2d convolutions to improve the accuracy of feature extraction after fusion
    """
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(EASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_pool = SoftPool2d(kernel_size=1, stride=1)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            DeformConv2d(dim_out*5, dim_out, 1, 0, 1, True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()

        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = self.branch5_pool(x)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)

        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class ST_R50_connect (nn.Module):
    """
    Fuse the extraction results of the Swin-transformer and ResNet50 encoders
    """
    def __init__(self, pretrained=True, channel_1=2048, channel_2=1024, downsample_factor=16):
        super(ST_R50_connect, self).__init__()

        self.backbone_1 = resnet50(pretrained=pretrained)
        self.backbone_2 = swin_transformer()

        # Perform feature extraction using the EASPP feature extraction module
        self.easpp = EASPP(dim_in=channel_1, dim_out=256, rate=16 // downsample_factor)
        self.cls_conv_1 = nn.Conv2d(256, 64, 1, stride=1)

        """Further extract global features from the up-sampled results of Swin-transformer 
        using CBAM and change the number of channels"""
        self.cabm = CBAMBlock(channel=channel_2, reduction=16, kernel_size=9)
        self.stchannel_change = nn.Sequential(
            nn.Conv2d(in_channels=channel_2, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Concatenate the results of the processed deep features extracted from the two branches
        self.st_r50_cat_conv = nn.Sequential(
            DeformConv2d(128 + 64, 64, 3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            DeformConv2d(64, 64, 3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

    def forward(self, x1, x2):

        lowfeature, x_1 = self.backbone_1(x1)
        x_1 = self.easpp(x_1)
        x_1 = F.interpolate(x_1, size=(lowfeature.size(2)//4, lowfeature.size(3)//4),
                            mode='bilinear', align_corners=True)
        x_1 = self.cls_conv_1(x_1)

        x_2 = self.backbone_2(x2)
        x_2 = self.cabm(x_2)
        x_2 = F.interpolate(x_2, size=(x_1.size(2), x_1.size(3)), mode='bilinear',align_corners=True)
        x_2 = self.stchannel_change(x_2)

        x_3 = self.st_r50_cat_conv(torch.cat((x_1, x_2), dim=1))

        return x_3, lowfeature  #(2,64,64,64), (2,128,256,256)


class STRD_Net(nn.Module):
    def __init__(self, num_classes=3, low_level_channels=128):
        super(STRD_Net, self).__init__()

        # Obtain shallow features and the fused deep features
        self.str50con = ST_R50_connect()

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            DeformConv2d(48 + 64, 64, 3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            DeformConv2d(64, 64, 3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )

        self.cls_conv_2 = nn.Conv2d(64, num_classes, 1, stride=1)

    def forward(self, x1, x2):
        # H, W = 512, 512
        H, W = x1.size(2), x1.size(3)

        x, lowfeature = self.str50con(x1, x2)

        x = F.interpolate(x, size=(lowfeature.size(2), lowfeature.size(3)),
                          mode='bilinear', align_corners=True)

        lowfeature = self.shortcut_conv(lowfeature)

        x = self.cat_conv(torch.cat((x, lowfeature), dim=1))

        x = self.cls_conv_2(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x1 = torch.rand(2, 3, 256, 256)
    x2 = torch.rand(2, 3, 256, 256)
    x1 = x1.to(device)
    x2 = x2.to(device)
    net = STRD_Net()
    net = net.to(device)

    # Ensure the model is in evaluation mode
    model = STRD_Net().eval()
    model = model.to('cuda')
    # Calculate GFLOPs, latency, FPS, and params
    flops, params = profile(model, (x1,x2, ), verbose=False)
    start_time = time.perf_counter()
    output = model(x1, x2)
    end_time = time.perf_counter()
    latency = end_time - start_time
    fps = 1 / latency

    print(output.shape)
    print('Total GFLOPS: %.3f' % (flops / 1e9))
    print('Total params: %d' % params)
    print(f'Latency for single inference: {latency:.4f} seconds')
    print(f'Calculated FPS: {fps:.2f} frames per second')