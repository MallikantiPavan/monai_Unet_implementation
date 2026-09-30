import glob
import os
from sympy import Lambda
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
    Lambda,
)
import random
import pandas as pd
import nibabel as nib
import numpy as np
import torch

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def read_csv(file_path):
    base_root=config['paths']['data_dir']
    data=[]
    csv_file=pd.read_csv(file_path)
    for _,row in csv_file.iterrows():
        participant_id=row['participant_id']
        t1w_path=os.path.join(base_root,participant_id,"T1w_brain.nii.gz")
        flair_path=os.path.join(base_root,participant_id,"FLAIR_brain.nii.gz")
        label_path=os.path.join(base_root,participant_id,"FLAIR_roi.nii.gz")
        if not os.path.exists(flair_path) and  os.path.exists(t1w_path):
            print(f"Image file not found: {t1w_path} or {flair_path}. Skipping this entry.")
            continue
        # if  os.path.exists(label_path):
        #     label=label_path
        # else:
        #     flair_shape=nib.load(image_path).shape
        #     label=np.zeros(flair_shape,dtype=np.uint8)
        if os.path.exists(label_path):
            data.append({"t1w": t1w_path,"flair": flair_path, "label":  label_path})
    return data


    

        

def get_data_files():
    
    data_dicts = read_csv(config['paths']['train_csv'])
    random.Random(config['training']['seed']).shuffle(data_dicts)
    val_size = config['data']["val_size"]
    train_files=data_dicts[:-val_size]
    val_files=data_dicts[-val_size:]
    return train_files, val_files


def get_test_files():
    test_dirs=read_csv(config['paths']['test_csv'])
    return test_dirs


def combined_transforms():
    return Compose(
        [
            LoadImaged(keys=["flair","t1w", "label"]),
            EnsureChannelFirstd(keys=["flair","t1w", "label"]),
            #flair transforms
            ScaleIntensityRanged(
                keys=["flair"],
                a_min=0.9966772212646902,
                a_max=551.0037992522124,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            # CropForegroundd(keys=["flair", "label"], source_key="flair", allow_smaller=True),
            Orientationd(keys=["flair","t1w", "label"], axcodes="RAS"),

            #t1w transforms
            ScaleIntensityRanged(
                keys=["t1w"],
                a_min=0.7421839237213135,
                a_max=1606.20263671875,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            #combine

            Lambda(
            func=lambda x: {
                "image": torch.cat([
                    x["flair"].float(),
                    x["t1w"].float()
                ], dim=0),
                "label": x["label"]
            }
        )
    ])





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
        LoadImaged(keys=["flair","t1w", "label"]),
        EnsureChannelFirstd(keys=["flair","t1w", "label"]),
        Orientationd(keys=["flair","t1w", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["flair"],
            a_min=0.9966772212646902,
            a_max=551.0037992522124,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ScaleIntensityRanged(
            keys=["t1w"],
            a_min=0.7421839237213135,
            a_max=1606.20263671875,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Lambda(
            func=lambda x: {
                "image": torch.cat([
                    x["flair"].float(),
                    x["t1w"].float()
                ], dim=0),
                "label": x["label"]
            }
        )
        
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
