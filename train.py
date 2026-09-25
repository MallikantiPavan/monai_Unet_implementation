import os

import torch
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import CacheDataset
import yaml
from monai.data import DataLoader,Dataset
import logging

from model.model import create_model
from plottings.plot import plot_training_history
from dataset.data import get_data_files, train_transforms, val_transforms,val_org_transforms,post_transforms
import matplotlib.pyplot as plt
from monai.handlers.utils import from_engine

def train():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    set_determinism(seed=config['training']['seed'])
    train_files, val_files = get_data_files()
    train_ds = CacheDataset(data=train_files, transform=train_transforms(), cache_rate=1.0, num_workers=4)
    train_loader=DataLoader(train_ds, batch_size=config['data']["batch_size"], shuffle=True, num_workers=config['data']["num_workers"])
    val_ds = CacheDataset(data=val_files, transform=val_transforms(), cache_rate=1.0, num_workers=4)
    val_loader=DataLoader(val_ds, batch_size=1, num_workers=config['data']["num_workers"])

    model, loss_function, optimizer, dice_metric, device = create_model()
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    for epoch in range(config['training']["max_epochs"]):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{config['training']['max_epochs']}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % config['training']["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_outputs = sliding_window_inference(
                        val_inputs, config['training']["roi_size"], config['training']["sw_batch_size"], model
                    )
                    val_outputs = [post_pred(item) for item in decollate_batch(val_outputs)]
                    val_labels = [post_label(item) for item in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)
            logging.info(f"epoch {epoch + 1} current mean dice: {metric:.4f}")
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(config['paths']["checkpoint_dir"], "best_metric_model.pth"))
                print("saved new best metric model")
                logging.info(f"saved new best metric model at epoch {best_metric_epoch}")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )
            logging.info(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    plot_training_history(epoch_loss_values, metric_values, config['training']["val_interval"])

    model.load_state_dict(
        torch.load(
            os.path.join(config['paths']['checkpoint_dir'], "best_metric_model.pth"),
            weights_only=True
        )
    )

    model.eval()

    with torch.no_grad():
        for i, val_data in enumerate(val_loader):

            roi_size = config['training']["roi_size"]
            sw_batch_size = config['training']["sw_batch_size"]

            val_outputs = sliding_window_inference(
                val_data["image"].to(device),
                roi_size,
                sw_batch_size,
                model
            )

            plt.figure("check", (18, 6))

            # Image
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(
                val_data["image"][0, 0, :, :, 40],
                cmap="gray"
            )

            # Label
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(
                val_data["label"][0, 0, :, :, 40]
            )

            # Output
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(
                torch.argmax(val_outputs, dim=1)
                .detach()
                .cpu()[0, :, :, 40]
            )

            # Save instead of show
            plt.savefig(
                os.path.join(
                    "/storage/projects/vinkle/ez_compass_imaging/code/monai_unet_test/curves_fcd",
                    f"validation_{i}.png"
                ),
                bbox_inches="tight"
            )

            plt.close()

            if i == 2:
                break



    val_org_transform = val_org_transforms()
    val_org_ds=Dataset(data=val_files, transform=val_org_transform)
    val_org_loader=DataLoader(val_org_ds, batch_size=1, num_workers=config['data']["num_workers"])

    model.load_state_dict(
        torch.load(
            os.path.join(config['paths']['checkpoint_dir'], "best_metric_model.pth"),
            weights_only=True
        )
    )
    model.eval()
    post_transform = post_transforms(val_org_transform)
    with torch.no_grad():
        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            roi_size = config['training']["roi_size"]
            sw_batch_size = config['training']["sw_batch_size"]
            val_data["pred"] = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model
            )
            val_data = [post_transform(item) for item in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            dice_metric(y_pred=val_outputs, y=val_labels)
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
    print("Metric on original image spacing: ", metric)
            

if __name__ == "__main__":
    train()



# best mean dice: 0.9502 at epoch: 542
# train completed, best_metric: 0.9502 at epoch: 542
# Metric on original image spacing:  0.9606824517250061


# FCD only flair metric

# epoch 600 average loss: 0.0625
# current epoch: 600 current mean dice: 0.0063
# best mean dice: 0.0133 at epoch: 458
# train completed, best_metric: 0.0133 at epoch: 458
# Metric on original image spacing:  0.01254983525723219

