import re
import glob
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism, first
import os
import pandas as pd
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    AddChanneld,
    LoadImaged,
    SpatialCropd,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations,
    SpatialCrop,
    Crop,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
import torch
import numpy as np
from sklearn.model_selection import train_test_split


# Von Clemens geschriebene funktion um die daten rauszufiltern zu denen es kein label gibt
def match_image_with_label(volumes_path, segmentation_path):
    image_dict = []
    for image_path in volumes_path:
        case_name = re.search(r'case_[0-9]{5}.nii', image_path)

        for label_path in segmentation_path:
            if label_path.find(case_name.group()) != -1:
                image_dict.append({"vol": image_path, "seg": label_path})
                break

    return image_dict

def dice_metric(predicted, target):
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value

def match_box(data, images):
    for image in images:
        box = data[data['case'] == image['vol']]
        value = box["y1"] < box["y2"]
        if value.bool:

            box_start = box[["y1", "x1", "z1"]]
            box_start = box_start.to_numpy().astype(int)
            image["start"] = box_start

            box_end = box[["y2", "x2", "z2"]]
            box_end = box_end.to_numpy().astype(int)
            image["end"] = box_end
        else:
            box_start = box[["y2", "x2", "z1"]]
            box_start = box_start.to_numpy().astype(int)
            image["start"] = box_start

            box_end = box[["y1", "x1", "z2"]]
            box_end = box_end.to_numpy().astype(int)
            image["end"] = box_end

    return images


class cropBox(object):
    def __call__(self, img):
        try:
            trans = SpatialCropd(keys=["vol", "seg"], roi_start=img['start'][0], roi_end=img['end'][0])
        except:
            trans = SpatialCropd(keys=["vol", "seg"], roi_start=img['start'], roi_end=img['end'])
        image = trans(img)
        return image

set_determinism(seed=0)

in_dir = 'C:\\Users\\ChiaraFreistetter\\Desktop\\fh\\inno-organ_segmentation\\data\\KiTS-20210922T123706Z-001'

volumes = sorted(glob.glob(in_dir + "\\images\\**\\*.nii"))
segmentation = sorted(glob.glob(in_dir + "\\labels\\**\\*.nii"))
# "\\labelsTr\\*.nii.gz"

# to make sure we every image has a label
all_files = match_image_with_label(volumes, segmentation)

###################################################################################################################
# Bounding box
###################################################################################################################

# get the bounding boxes
df = pd.read_csv("bounding_box_points.csv")
df['z1'] = 0
df["z2"] = 512

all_files = match_box(df, all_files)

###################################################################################################################
#Train-Test Split and Transforms
###################################################################################################################
train_files, validation_files = train_test_split(all_files, test_size=0.2, random_state=42, shuffle=True)

a_max = 200
a_min = -200
spatial_size = [128, 128, 64]
pixdim = (1.5, 1.5, 1.0)

# calculate on gpu -> warning not campatible with every gpu!!!!!
device = torch.device("cuda:0")


#Mostly same but instead of crop foreground now cropping for bounding box
validation_transforms = Compose(
    [
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstd(keys=["vol", "seg"]),
        Orientationd(keys=["vol", "seg"], axcodes="PLS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        cropBox(),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),
        ToTensord(keys=["vol", "seg"]),
    ]
        )

train_transforms = Compose(
    [
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstd(keys=["vol", "seg"]),
        Orientationd(keys=["vol", "seg"], axcodes="PLS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        cropBox(),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),
        ToTensord(keys=["vol", "seg"]),
    ]
        )

#load data
validation_ds = Dataset(data=validation_files, transform=validation_transforms)
validation_loader = DataLoader(validation_ds, batch_size=1)

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1)

###################################################################################################################
# Model, Training and Eval
###################################################################################################################
model_dir = 'C:\\Users\\ChiaraFreistetter\\Desktop\\fh\\inno-organ_segmentation\\results'

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    kernel_size=3,
).to(device)

model.load_state_dict(torch.load(
    os.path.join(model_dir, "C:\\Users\\ChiaraFreistetter\\Desktop\\fh\\inno-organ_segmentation\\results\\best_metric_model_BBox.pth")))
model.eval()

# hier wird der optimization alg. sowie die metric als variable gesetzt
loss = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optim = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)
# die brauchen wir dann weiter unten
max_epochs = 65
validation_interval = 5

# training part
# ein paar leere varaiblen zum später verwenden
best_metric = -1
best_metric_epoch = -1
save_loss_train = []
save_loss_validation = []
save_metric_train = []
save_metric_validation = []


for epoch in range(max_epochs):
    # das ganze wird schön in der konsole ausgegeben damit man den vortschritt sieht
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    train_epoch_loss = 0
    train_step = 0
    epoch_metric_train = 0

    # jede datei
    # erst hier beginnt die schleife bei der dann durch jede datei durchgegangen wird
    for batch_data in train_loader:
        train_step += 1
        volume = batch_data["vol"]
        # wenn das label nicht gleich 0 ist
        label = batch_data["seg"]
        label = label != 0
        volume, label = (volume.to(device, non_blocking=True), label.to(device, non_blocking=True))
        optim.zero_grad()
        outputs = model(volume)
        # für jedes image wird der Dice ausgerechnet damit wir wissen wie gut unser model an dem file performed hat
        train_loss = loss(outputs, label)
        # die beiden lines haben dann was mit dem optimization alg zu tun,
        # was genau passiert weiß ich nicht aber ich glaube es passt das model an?
        train_loss.backward()
        optim.step()

        # gesamt-loss einer epoche
        train_epoch_loss += train_loss.item()
        # wieder ne schöne ausgabe zum ansehen
        print(
            f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
            f"Train_loss: {train_loss.item():.4f}")
        # gesamt metric der epoche
        train_metric = dice_metric(outputs, label)
        epoch_metric_train += train_metric
        print(f'Train_dice: {train_metric:.4f}')
    # hier endet die schleife für jede datei in dem ordner --> das ende der epoche quasi
    print('-' * 20)

    # quasi der mittelwert des loss, der wird dann gespeichert
    train_epoch_loss /= train_step
    print(f'Epoch_loss: {train_epoch_loss:.4f}')
    save_loss_train.append(train_epoch_loss)
    np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)

    # wie bei loss jetzt auch mit metric
    epoch_metric_train /= train_step
    print(f'Epoch_metric: {epoch_metric_train:.4f}')
    save_metric_train.append(epoch_metric_train)
    np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

    # validation interval gibt an nach wie vielen epochen man mit
    # gespeicherten model vergeleicht und bei besserung dann abspeichert
    if (epoch + 1) % validation_interval == 0:

        model.eval()
        with torch.no_grad():
            validation_epoch_loss = 0
            validation_metric = 0
            epoch_metric_validation = 0
            validation_step = 0

            for validation_data in validation_loader:
                validation_step += 1

                # wie bie training oben
                validation_volume = validation_data["vol"]
                validation_label = validation_data["seg"]
                validation_label = validation_label != 0
                validation_volume, validation_label = (validation_volume.to(device, non_blocking=True), validation_label.to(device, non_blocking=True))

                validation_outputs = model(validation_volume)

                validation_loss = loss(outputs, validation_label)
                validation_epoch_loss += validation_loss.item()
                validation_metric = dice_metric(validation_outputs, validation_label)
                epoch_metric_validation += validation_metric

            validation_epoch_loss /= validation_step
            print(f'validation_loss_epoch: {validation_epoch_loss:.4f}')
            save_loss_validation.append(validation_epoch_loss)
            np.save(os.path.join(model_dir, 'loss_validation.npy'), save_loss_validation)

            epoch_metric_validation /= validation_step
            print(f'validation_dice_epoch: {epoch_metric_validation:.4f}')
            save_metric_validation.append(epoch_metric_validation)
            np.save(os.path.join(model_dir, 'metric_validation.npy'), save_metric_validation)

            # hier fast das gleich wie in oberen zeilen,
            # dann wird geschaut ob die neue metric besser ist als die alte und wenn ja wird es abgespeichert
            if epoch_metric_validation > best_metric:
                best_metric = epoch_metric_validation
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    model_dir, "best_metric_model_BBox.pth"))

            # dann nur noch ein paar ausgaben
            print(
                f"current epoch: {epoch + 1} current mean dice: {validation_metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
# wenn alle oben definierten epochen durchlaufen sind wird nochmal die beste metric die erreicht wurde ausgegeben
print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")
