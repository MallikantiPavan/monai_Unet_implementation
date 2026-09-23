from dataset.data import test_org_transforms, post_test_transforms

import yaml
import os
import glob
from model.model import create_model
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine
import matplotlib.pyplot as plt
from monai.data import DataLoader, Dataset
import torch
from monai.transforms import LoadImage

loader = LoadImage()

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
def test():
    test_images=sorted(glob.glob(os.path.join(config['paths']['data_dir'],"imagesTs", "*.nii.gz")))
    test_data=[{"image": image_name} for image_name in test_images]
    test_transform=test_org_transforms()
    test_org_ds=Dataset(data=test_data, transform=test_transform)
    test_org_loader=DataLoader(test_org_ds, batch_size=1, num_workers=config['data']["num_workers"])
    model, loss_function, optimizer, dice_metric, device = create_model()
    model.load_state_dict(
        torch.load(
            os.path.join(config['paths']['checkpoint_dir'], "best_metric_model.pth"),
            weights_only=True
        )
    )
    model.eval()
    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs = test_data["image"].to(device)
            roi_size, sw_batch_size = config['training']["roi_size"], config['training']["sw_batch_size"]
            test_data["pred"] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model
            )
            post_test_transform = post_test_transforms(test_transform)
            test_data = [post_test_transform(item) for item in decollate_batch(test_data)]
            test_output = from_engine(["pred"])(test_data)
            original_image=loader(test_output[0].meta["filename_or_obj"])
            plt.figure("check", (18, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image[:, :, 20], cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_output[0].detach().cpu()[1, :, :, 20])
            plt.savefig(
                os.path.join(
                    "/storage/projects/vinkle/ez_compass_imaging/code/monai_unet_test/curves",
                    f"test_output_{test_output[0].meta['filename_or_obj'].split('/')[-1]}.png"
                ),
                bbox_inches="tight"
            )
            plt.show()

if __name__== "__main__":
    test()