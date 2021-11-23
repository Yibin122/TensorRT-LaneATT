import torch

from lib.models.laneatt import LaneATT


class LaneATTONNX(torch.nn.Module):
    def __init__(self, model):
        super(LaneATTONNX, self).__init__()
        # Params
        self.fmap_h = model.fmap_h  # 11
        self.fmap_w = model.fmap_w  # 20
        self.anchor_feat_channels = model.anchor_feat_channels  # 64
        self.anchors = model.anchors
        self.cut_xs = model.cut_xs
        self.cut_ys = model.cut_ys
        self.cut_zs = model.cut_zs
        self.invalid_mask = model.invalid_mask
        # Layers
        self.feature_extractor = model.feature_extractor
        self.conv1 = model.conv1
        self.cls_layer = model.cls_layer
        self.reg_layer = model.reg_layer
        self.attention_layer = model.attention_layer

        # Exporting the operator eye to ONNX opset version 11 is not supported
        attention_matrix = torch.eye(1000)
        self.non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        self.non_diag_inds = self.non_diag_inds[:, 1] + 1000 * self.non_diag_inds[:, 0]  # 999000

    def forward(self, x):
        batch_features = self.feature_extractor(x)
        batch_features = self.conv1(batch_features)
        # batch_anchor_features = self.cut_anchor_features(batch_features)
        batch_anchor_features = batch_features[0].flatten()
        # h, w = batch_features.shape[2:4]  # 12, 20
        batch_anchor_features = batch_anchor_features[self.cut_xs + 20 * self.cut_ys + 12 * 20 * self.cut_zs].\
            view(1000, self.anchor_feat_channels, self.fmap_h, 1)
        # batch_anchor_features[self.invalid_mask] = 0
        batch_anchor_features = batch_anchor_features * torch.logical_not(self.invalid_mask)

        # Join proposals from all images into a single proposals features batch
        batch_anchor_features = batch_anchor_features.view(-1, self.anchor_feat_channels * self.fmap_h)

        # Add attention features
        softmax = torch.nn.Softmax(dim=1)
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores)
        attention_matrix = torch.zeros(1000 * 1000, device=x.device)
        attention_matrix[self.non_diag_inds] = attention.flatten()  # ScatterND
        attention_matrix = attention_matrix.view(1000, 1000)
        attention_features = torch.matmul(torch.transpose(batch_anchor_features, 0, 1),
                                          torch.transpose(attention_matrix, 0, 1)).transpose(0, 1)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=1)

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        # Add offsets to anchors (1000, 2+2+73)
        reg_proposals = torch.cat([softmax(cls_logits), self.anchors[:, 2:4], self.anchors[:, 4:] + reg], dim=1)

        return reg_proposals


def export_onnx(onnx_file_path):
    # e.g. laneatt_r18_culane
    backbone_name = 'resnet18'
    checkpoint_file_path = 'experiments/laneatt_r18_culane/models/model_0015.pt'
    anchors_freq_path = 'culane_anchors_freq.pt'

    # Load specified checkpoint
    model = LaneATT(backbone=backbone_name, anchors_freq_path=anchors_freq_path, topk_anchors=1000)
    checkpoint = torch.load(checkpoint_file_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Export to ONNX
    onnx_model = LaneATTONNX(model)
    dummy_input = torch.randn(1, 3, 360, 640)
    torch.onnx.export(onnx_model, dummy_input, onnx_file_path, opset_version=11)


if __name__ == '__main__':
    export_onnx('./LaneATT_test.onnx')
