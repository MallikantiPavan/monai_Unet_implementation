import glob
import os
import yaml
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
import pandas as pd
import nibabel as nib
import numpy as np

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def read_csv(file_path):
    base_root=config['paths']['data_dir']
    data=[]
    csv_file=pd.read_csv(file_path)
    for _,row in csv_file.iterrows():
        participant_id=row['participant_id']
        fcd_patients=row['group']
        if fcd_patients=="fcd":
            image_path=os.path.join(base_root,participant_id,"FLAIR_brain.nii.gz")
            label_path=os.path.join(base_root,participant_id,"FLAIR_roi.nii.gz")
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue
            # if  os.path.exists(label_path):
            #     label=label_path
            # else:
            #     flair_shape=nib.load(image_path).shape
            #     label=np.zeros(flair_shape,dtype=np.uint8)
            if os.path.exists(label_path):
                data.append({"image": image_path, "label":  label_path})
    return data


    

        

def get_data_files():
    
    data_dicts = read_csv(config['paths']['train_csv'])
    train_files, val_files = data_dicts[:-config['data']["val_size"]], data_dicts[-config['data']["val_size"]:]
    return train_files, val_files


def get_test_files():
    test_dirs=read_csv(config['paths']['test_csv'])
    return test_dirs


def train_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.9966772212646902,
                a_max=551.0037992522124,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 96, 64),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )



def val_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.9966772212646902,
                a_max=551.0037992522124,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            
        ]
    )


# def get_original_spacing_transforms():
#     return Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             Orientationd(keys=["image"], axcodes="RAS"),
#             Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
#             ScaleIntensityRanged(
#                 keys=["image"],
#                 a_min=-57,
#                 a_max=164,
#                 b_min=0.0,
#                 b_max=1.0,
#                 clip=True,
#             ),
#             CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
#         ]
#     )


def val_org_transforms():
    return Compose(
    [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0.9966772212646902,
                    a_max=551.0037992522124,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]
        )

def post_transforms(transform):
    return Compose(
    [
            Invertd(
                keys="pred",
                transform=transform,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
                device="cpu",
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=2),
            AsDiscreted(keys="label", to_onehot=2),
        ]
    )




def test_org_transforms():
    return Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.9966772212646902,
            a_max=551.0037992522124,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        
    ]
)

def post_test_transforms(transform,output_name):
    return Compose(
    [
        Invertd(
            keys="pred",
            transform=transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=config['paths']['output_dir'], output_postfix=f"{output_name}_FLAIR_seg", resample=False),
    ]
)
