import yaml

import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet

with open("config/config.yaml", "r") as f:
    MODEL_CONFIG = yaml.safe_load(f)["model"]


def create_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=MODEL_CONFIG["in_channels"],
        out_channels=MODEL_CONFIG["out_channels"],
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    return model, loss_function, optimizer, dice_metric, device
