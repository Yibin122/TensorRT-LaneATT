import torch

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector


class GANetOnnx(torch.nn.Module):
    def __init__(self, model):
        super(GANetOnnx, self).__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.bbox_head

    def forward(self, img):
        output = self.backbone(img)
        output = self.neck(output)

        # key points hm
        f_hm = output['features'][0]
        z_ = self.head.keypts_head(f_hm)
        kpts_hm = z_['hm']
        kpts_hm = torch.clamp(kpts_hm.sigmoid(), min=1e-4, max=1 - 1e-4)
        # offset map
        f_hm = output['aux_feat']
        o = self.head.offset_head(f_hm)
        pts_offset = o['offset_map']
        o_ = self.head.reg_head(f_hm)
        int_offset = o_['offset_map']

        # nms
        hmax = torch.nn.functional.max_pool2d(kpts_hm, (1, 3), stride=(1, 1), padding=(0, 1))
        keep = (hmax == kpts_hm).float()  # false:0 true:1
        heat_nms = kpts_hm * keep

        return heat_nms, pts_offset, int_offset


def export_onnx(onnx_file_path):
    # https://github.com/Wolfwjs/GANet
    config_file = 'configs/culane/final_exp_res18_s8.py'
    checkpoint_file = './ganet_culane_resnet18.pth'

    # Load pretrained model
    cfg = Config.fromfile(config_file)
    cfg.model.pretrained = None
    model = build_detector(cfg.model)
    load_checkpoint(model, checkpoint_file, map_location='cpu')
    model = model.cuda().eval()

    # Export to ONNX
    ganet_model = GANetOnnx(model)
    dummy_img = torch.randn(1, 3, 320, 800, device='cuda:0')
    torch.onnx.export(ganet_model, dummy_img, onnx_file_path, opset_version=11, enable_onnx_checker=False)
    print('Saved GANet to onnx file: {}'.format(onnx_file_path))


if __name__ == '__main__':
    export_onnx('./ganet.onnx')
