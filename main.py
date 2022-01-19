import re
import glob
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
import os
from monai.transforms import (
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
import torch
import numpy as np


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
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value


set_determinism(seed=0)

# in_dir = 'C:\\Users\\ChiaraFreistetter\\Desktop\\fh\\inno-organ_segmentation\\data\\KiTS-20210922T123706Z-001'
in_dir = 'C:\\Users\\ChiaraFreistetter\\Desktop\\fh\\inno-organ_segmentation\\data\\testRun1'

# path to data
path_train_volumes = sorted(glob.glob(in_dir + "\\imagesTr\\**\\*.nii"))
path_train_segmentation = sorted(glob.glob(in_dir + "\\labelsTr\\**\\*.nii"))
# "\\labelsTr\\*.nii.gz" funktioniert auch --> datensparender aebr nopch unser wegen umgang mit daten

path_validation_volumes = sorted(glob.glob(in_dir + "\\imagesTest\\**\\*.nii"))
path_validation_segmentation = sorted(glob.glob(in_dir + "\\labelsTest\\**\\*.nii"))

train_files = match_image_with_label(path_train_volumes, path_train_segmentation)
validation_files = match_image_with_label(path_validation_volumes, path_validation_segmentation)

# print(train_files)
# print(validation_files)
# print(train_files)

a_max = 200
a_min = -200
spatial_size = [128, 128, 64]
pixdim = (1.5, 1.5, 1.0)

# training set
train_transforms = Compose(
    [
        LoadImaged(keys=["vol", "seg"]),
        AddChanneld(keys=["vol", "seg"]),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["vol", "seg"], source_key="vol"),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),
        ToTensord(keys=["vol", "seg"]),
    ]
)
# for comparing
validation_transforms = Compose(
    [
        LoadImaged(keys=["vol", "seg"]),
        AddChanneld(keys=["vol", "seg"]),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),
        ToTensord(keys=["vol", "seg"]),

    ]
)

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1)

validation_ds = Dataset(data=validation_files, transform=validation_transforms)
validation_loader = DataLoader(validation_ds, batch_size=1)

# data_dir = 'D:/Youtube/Organ and Tumor Segmentation/datasets/Task03_Liver/Data_Train_Test'
model_dir = 'C:\\Users\\ChiaraFreistetter\\Desktop\\fh\\inno-organ_segmentation\\results'

device = torch.device("cuda:0")
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

loss = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optim = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)
max_epochs = 3
validation_interval = 1

# training part
best_metric = -1
best_metric_epoch = -1
save_loss_train = []
save_loss_validation = []
save_metric_train = []
save_metric_validation = []

# eine epoche
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    train_epoch_loss = 0
    train_step = 0
    epoch_metric_train = 0
    # jede datei
    for batch_data in train_loader:
        train_step += 1

        volume = batch_data["vol"]
        # wenn das label nicht gleich 0 nist
        label = batch_data["seg"]
        label = label != 0
        volume, label = (volume.to(device), label.to(device))

        optim.zero_grad()
        outputs = model(volume)

        train_loss = loss(outputs, label)

        train_loss.backward()
        optim.step()

        # gesamt-loss einer epoche
        train_epoch_loss += train_loss.item()
        print(
            f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
            f"Train_loss: {train_loss.item():.4f}")
        # gesamt metric der epoche
        train_metric = dice_metric(outputs, label)
        epoch_metric_train += train_metric
        print(f'Train_dice: {train_metric:.4f}')

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
                validation_volume, validation_label = (validation_volume.to(device), validation_label.to(device),)

                # hier ist anders als oben, waas macht die funktion?
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

            # bis hier auch eig wie oben nur hald das mit validation und nicht traindata gemacht wird
            # dann wird geschaut ob die nue metric besser ist als die alte und wenn ja wird es abgespeichert
            if epoch_metric_validation > best_metric:
                best_metric = epoch_metric_validation
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    model_dir, "best_metric_model.pth"))

            # dann nur noch ein paar ausgaben
            print(
                f"current epoch: {epoch + 1} current mean dice: {validation_metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")
