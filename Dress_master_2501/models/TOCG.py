import torch.nn.functional as F
from utils import *
from .ResBlock import ResBlock


class ConditionGenerator(nn.Module):
    def __init__(self, opt, input1_nc, input2_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.warp_feature = opt.warp_feature
        self.out_layer_opt = opt.out_layer

        self.ClothEncoder = nn.Sequential(
            ResBlock(input1_nc, ngf, norm_layer=norm_layer, scale='down'),  # 128
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),  # 64
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),  # 32
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 16
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')  # 8
        )

        self.PoseEncoder = nn.Sequential(
            ResBlock(input2_nc, ngf, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')
        )

        self.conv = ResBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, scale='same')

        if opt.warp_feature == 'T1':
            # in_nc -> skip connection + T1, T2 channel
            self.SegDecoder = nn.Sequential(
                ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
                ResBlock(ngf * 4 * 2 + ngf * 4, ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
                ResBlock(ngf * 4 * 2 + ngf * 4, ngf * 2, norm_layer=norm_layer, scale='up'),  # 64
                ResBlock(ngf * 2 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up'),  # 128
                ResBlock(ngf * 1 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up')  # 256
            )
        if opt.warp_feature == 'encoder':
            # in_nc -> [x, skip_connection, warped_cloth_encoder_feature(E1)]
            self.SegDecoder = nn.Sequential(
                ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
                ResBlock(ngf * 4 * 3, ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
                ResBlock(ngf * 4 * 3, ngf * 2, norm_layer=norm_layer, scale='up'),  # 64
                ResBlock(ngf * 2 * 3, ngf, norm_layer=norm_layer, scale='up'),  # 128
                ResBlock(ngf * 1 * 3, ngf, norm_layer=norm_layer, scale='up')  # 256
            )
        if opt.out_layer == 'relu':
            self.out_layer = ResBlock(ngf + input1_nc + input2_nc, output_nc, norm_layer=norm_layer, scale='same')
        if opt.out_layer == 'conv':
            self.out_layer = nn.Sequential(
                ResBlock(ngf + input1_nc + input2_nc, ngf, norm_layer=norm_layer, scale='same'),
                nn.Conv2d(ngf, output_nc, kernel_size=1, bias=True)
            )

        # Cloth Conv 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        # Person Conv 1x1
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        self.flow_conv = nn.ModuleList([
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        )

        self.bottleneck = nn.Sequential(
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
        )

        self.DEVICE = torch.device("cuda:0" if opt.cuda else "cpu")

    def normalize(self, x):
        return x

    def forward(self, opt, input1, input2, upsample='bilinear'):
        E1_list = []
        E2_list = []
        flow_list = []
        # warped_grid_list = []

        # Feature Pyramid Network
        for i in range(5):
            if i == 0:
                E1_list.append(self.ClothEncoder[i](input1))
                E2_list.append(self.PoseEncoder[i](input2))
            else:
                E1_list.append(self.ClothEncoder[i](E1_list[i - 1]))
                E2_list.append(self.PoseEncoder[i](E2_list[i - 1]))

        # Compute Clothflow
        for i in range(5):
            N, _, iH, iW = E1_list[4 - i].size()
            grid = make_grid(N, iH, iW, opt).to(self.DEVICE)

            if i == 0:
                T1 = E1_list[4 - i]  # (ngf * 4) x 8 x 6
                T2 = E2_list[4 - i]
                E4 = torch.cat([T1, T2], 1)

                flow = self.flow_conv[i](self.normalize(E4)).permute(0, 2, 3, 1)
                flow_list.append(flow)

                x = self.conv(T2)
                x = self.SegDecoder[i](x)

            else:
                T1 = F.interpolate(T1, scale_factor=2, mode=upsample) + self.conv1[4 - i](E1_list[4 - i])
                T2 = F.interpolate(T2, scale_factor=2, mode=upsample) + self.conv2[4 - i](E2_list[4 - i])

                flow = F.interpolate(flow_list[i - 1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2,
                                                                                                                  3,
                                                                                                                  1)  # upsample n-1 flow
                flow_norm = torch.cat(
                    [flow[:, :, :, 0:1] / ((iW / 2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH / 2 - 1.0) / 2.0)], 3)
                warped_T1 = F.grid_sample(T1, flow_norm + grid, padding_mode='border')

                flow = flow + self.flow_conv[i](
                    self.normalize(torch.cat([warped_T1, self.bottleneck[i - 1](x)], 1))).permute(0, 2, 3, 1)  # F(n)
                flow_list.append(flow)

                if self.warp_feature == 'T1':
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4 - i], warped_T1], 1))
                if self.warp_feature == 'encoder':
                    warped_E1 = F.grid_sample(E1_list[4 - i], flow_norm + grid, padding_mode='border')
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4 - i], warped_E1], 1))

        N, _, iH, iW = input1.size()
        grid = make_grid(N, iH, iW, opt)

        flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
        flow_norm = torch.cat(
            [flow[:, :, :, 0:1] / ((iW / 2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH / 2 - 1.0) / 2.0)], 3)
        warped_input1 = F.grid_sample(input1, flow_norm + grid, padding_mode='border')

        x = self.out_layer(torch.cat([x, input2, warped_input1], 1))

        warped_c = warped_input1[:, :-1, :, :]
        warped_cm = warped_input1[:, -1:, :, :]

        return flow_list, x, warped_c, warped_cm