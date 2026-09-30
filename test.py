from dataset.data import get_test_files,test_org_transforms, post_test_transforms

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
from monai.transforms import LoadImage,Compose, AsDiscrete
import pandas as pd

loader = LoadImage()

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
def test():
    test_data=get_test_files()
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
    post_pred=Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label=Compose([AsDiscrete(to_onehot=2)])
    csv_list=[['subject_id','test_mean_dice']]
    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs = test_data["image"].to(device)
            test_labels = test_data["label"].to(device)
            roi_size, sw_batch_size = config['training']["roi_size"], config['training']["sw_batch_size"]
            test_data["pred"] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model
            )
            sub_id=os.path.basename(os.path.dirname(test_data['image'].meta['filename_or_obj'][0]))
            output_name = f"{sub_id}_FLAIR_segment.nii.gz"
            test_pred=[
                post_pred(i) for i in decollate_batch(test_data["pred"])
            ]
            test_label=[
                post_label(i) for i in decollate_batch(test_labels)
            ]
            test_subjects=decollate_batch(test_data)
            saved_subjects=[]
            for subject in test_subjects:
                subject = post_test_transforms(test_transform,output_name)(subject)
                saved_subjects.append(subject)
            test_output = from_engine(["pred"])(saved_subjects)
            dice_metric(y_pred=test_pred, y=test_label)
            subject_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"{sub_id} Dice: {subject_dice:.4f}")
            csv_list.append([sub_id,subject_dice])
            original_image=loader(test_output[0].meta["filename_or_obj"])
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(original_image[:, :, 135], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.imshow(test_data['label'].detach().cpu()[0,0, :, :, 135])
            plt.subplot(1, 3, 3)
            plt.imshow(torch.argmax(test_data['pred'], dim=1).detach().cpu()[0, :, :, 135])
            plt.savefig(
                os.path.join(
                    f"/storage/projects/vinkle/ez_compass_imaging/code/monai_unet_test/curves_fcd",
                    f"test_output_{sub_id}_{os.path.basename(test_output[0].meta['filename_or_obj'][0])}.png"
                ),
                bbox_inches="tight"
            )
            plt.show()
    dice_df=pd.DataFrame(csv_list[1:],columns=csv_list[0])
    test_mean_dice=dice_df['test_mean_dice'].mean()
    print(f"test mean dice: {test_mean_dice:.4f}")
    dice_df.to_csv(f"/storage/projects/vinkle/ez_compass_imaging/code/monai_unet_test/curves_fcd/test_mean_dice.csv", index=False)


if __name__== "__main__":
    test()