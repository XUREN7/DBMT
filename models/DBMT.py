import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.HVI_transform import RGB_HVI
from models.transformer_utils import *
from models.MTM import *


class DBMT(nn.Module):
    def __init__(self, norm=False):
        super(DBMT, self).__init__()

        [ch1, ch2, ch3, ch4] = [36, 36, 72, 144]
        [head1, head2, head3, head4] = [1, 2, 4, 8]

        # HV路线
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )

        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # I路线
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        self.HV_down_MTMixer1 = HVI_MTMixer(ch2)
        self.HV_down_CAB1 = HV_CAB(ch2, head2)
        self.HV_down_MTMixer2 = HVI_MTMixer(ch3)
        self.HV_down_CAB2 = HV_CAB(ch3, head3)
        self.HV_down_MTMixer3 = HVI_MTMixer(ch4)
        self.HV_down_CAB3 = HV_CAB(ch4, head4)

        self.HV_up_MTMixer3 = HVI_MTMixer(ch4)
        self.HV_up_CAB3 = HV_CAB(ch4, head4)
        self.HV_up_MTMixer2 = HVI_MTMixer(ch3)
        self.HV_up_CAB2 = HV_CAB(ch3, head3)
        self.HV_up_MTMixer1 = HVI_MTMixer(ch2)
        self.HV_up_CAB1 = HV_CAB(ch2, head2)

        self.I_down_MTMixer1 = HVI_MTMixer(ch2)
        self.I_down_CAB1 = I_CAB(ch2, head2)
        self.I_down_MTMixer2 = HVI_MTMixer(ch3)
        self.I_down_CAB2 = I_CAB(ch3, head3)
        self.I_down_MTMixer3 = HVI_MTMixer(ch4)
        self.I_down_CAB3 = I_CAB(ch4, head4)

        self.I_up_MTMixer3 = HVI_MTMixer(ch4)
        self.I_up_CAB3 = I_CAB(ch4, head4)
        self.I_up_MTMixer2 = HVI_MTMixer(ch3)
        self.I_up_CAB2 = I_CAB(ch3, head3)
        self.I_up_MTMixer1 = HVI_MTMixer(ch2)
        self.I_up_CAB1 = I_CAB(ch2, head2)

        self.trans = RGB_HVI().cuda()

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        
        # 进入CV之前的步骤
        i_down_0 = self.IE_block0(i)
        i_down_1 = self.IE_block1(i_down_0)
        hv_down_0 = self.HVE_block0(hvi)
        hv_down_1 = self.HVE_block1(hv_down_0)
        i_jump0 = i_down_0
        hv_jump0 = hv_down_0

        # 进入第一列CV
        i_down_1 = self.I_down_MTMixer1(i_down_1, "I")
        hv_down_1 = self.HV_down_MTMixer1(hv_down_1, "HV")

        i_down_2 = self.I_down_CAB1(i_down_1, hv_down_1)
        hv_down_2 = self.HV_down_CAB1(hv_down_1, i_down_1)
        v_jump1 = i_down_2
        hv_jump1 = hv_down_2
        i_down_2 = self.IE_block2(i_down_2)
        hv_down_2 = self.HVE_block2(hv_down_2)

        # 进入第二列CV
        i_down_2 = self.I_down_MTMixer2(i_down_2, "I")
        hv_down_2 = self.HV_down_MTMixer2(hv_down_2, "HV")

        i_down_3 = self.I_down_CAB2(i_down_2, hv_down_2)
        hv_down_3 = self.HV_down_CAB2(hv_down_2, i_down_2)
        v_jump2 = i_down_3
        hv_jump2 = hv_down_3
        i_down_3 = self.IE_block3(i_down_3)
        hv_down_3 = self.HVE_block3(hv_down_3)

        # CV_bottleNeck
        i_down_3 = self.I_down_MTMixer3(i_down_3, "I")
        hv_down_3 = self.HV_down_MTMixer3(hv_down_3, "HV")
        i_down_4 = self.I_down_CAB3(i_down_3, hv_down_3)
        hv_down_4 = self.HV_down_CAB3(hv_down_3, i_down_3)

        # CV_bottleNeck
        i_down_4 = self.I_up_MTMixer3(i_down_4, "I")
        hv_down_4 = self.HV_up_MTMixer3(hv_down_4, "HV")
        hv_up_4 = self.HV_up_CAB3(hv_down_4, i_down_4)
        i_up_4 = self.I_up_CAB3(i_down_4, hv_down_4)

        # 进入第四列CV
        hv_up_4 = self.HVD_block3(hv_up_4, hv_jump2)
        i_up_4 = self.ID_block3(i_up_4, v_jump2)

        i_up_4 = self.I_up_MTMixer2(i_up_4, "I")
        hv_up_4 = self.HV_up_MTMixer2(hv_up_4, "HV")
        hv_up_3 = self.HV_up_CAB2(hv_up_4, i_up_4)
        i_up_3 = self.I_up_CAB2(i_up_4, hv_up_4)

        # 进入第五列CV
        hv_up_3 = self.HVD_block2(hv_up_3, hv_jump1)
        i_up_3 = self.ID_block2(i_up_3, v_jump1)
        
        i_up_3 = self.I_up_MTMixer1(i_up_3, "I")
        hv_up_3 = self.HV_up_MTMixer1(hv_up_3, "HV")
        hv_up_2 = self.HV_up_CAB1(hv_up_3, i_up_3)
        i_up_2 = self.I_up_CAB1(i_up_3, hv_up_3)

        # 走完CV之后
        i_up_1 = self.ID_block1(i_up_2, i_jump0)
        i_up_0 = self.ID_block0(i_up_1)
        hv_up_1 = self.HVD_block1(hv_up_2, hv_jump0)
        hv_up_0 = self.HVD_block0(hv_up_1)

        output_hvi = torch.cat([hv_up_0, i_up_0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi
