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
from sklearn.model_selection import train_test_split

'''
    Hier ist ein video das ein wenig über monai erklärt (dauert ca 38min):
    https://www.youtube.com/watch?v=Ih-4xzRJYO0&ab_channel=ProjectMONAI

    Das im video verwendete notebook:
    https://github.com/Project-MONAI/MONAIBootcamp2021/blob/main/day1/1.%20Getting%20Started%20with%20MONAI.ipynb
'''


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


# Hier ne website zu Dice und DiceLoss:
# https://pycad.co/the-difference-between-dice-and-dice-loss/
def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    # achtung weil wir hier den Dice LOSS berechnet haben und den Dice wollen
    # drehen wir das ganze nocheinmal um indem wir nochmal 1 - diceloss rechnen
    # zb Dice = 0,7  DiceLoss = 1 - 0,7 = 0,3 dann wieder zurück zum Dice = 1 - 0,3 = 0,7
    value = 1 - dice_value(predicted, target).item()
    return value


set_determinism(seed=0)

# in_dir = 'C:\\Users\\ChiaraFreistetter\\Desktop\\fh\\inno-organ_segmentation\\data\\KiTS-20210922T123706Z-001'
in_dir = 'C:\\Users\\ChiaraFreistetter\\Desktop\\fh\\inno-organ_segmentation\\data\\testRun1'

# path to data
# glob hilft uns die unterschiedlich benannten files zu selecten
path_train_volumes = sorted(glob.glob(in_dir + "\\imagesTr\\**\\*.nii"))
path_train_segmentation = sorted(glob.glob(in_dir + "\\labelsTr\\**\\*.nii"))
# "\\labelsTr\\*.nii.gz" funktioniert auch --> datensparender aber noch unsicher wegen umgang mit daten

path_validation_volumes = sorted(glob.glob(in_dir + "\\imagesTest\\**\\*.nii"))
path_validation_segmentation = sorted(glob.glob(in_dir + "\\labelsTest\\**\\*.nii"))
# hier rufen wir dann Clemens funktion auf damit wir uns nicht darum scheren müssen ob alle images label haben
all_files = match_image_with_label(volumes, segmentation)

# train test split funktion von sklearn (eigentlich train validation split, test fehlt)
train_files, validation_files = train_test_split(all_files, test_size=0.33, random_state=42)


# print(train_files)
# print(validation_files)
# print(train_files)

# das sind werte die einfach vom bsp code sind, ob und wie man die berechnet weiß ich nicht
a_max = 200
a_min = -200
spatial_size = [128, 128, 64]
pixdim = (1.5, 1.5, 1.0)

# training set
# transforms sind die filter die wir pro image anwenden um sie leichter erkenntba für das model zu machen
# (wenn man die abgiebt erkennen wir meistens nix brauchbares)
train_transforms = Compose(
    [
        LoadImaged(keys=["vol", "seg"]),
        # monai funktion die auch medical files laden kann, enfern auch non medical datein
        AddChanneld(keys=["vol", "seg"]),  # ich glaube das added farbchannels die man bei img braucht
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),  # sorry weiß nicht was das macht
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        # richtet das bild random neu aus (damit das model auch auf eve gedrehte img angewendet werden kann)
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        # hier leider auch keinen plan
        CropForegroundd(keys=["vol", "seg"], source_key="vol"),  # ebenso wie hier (sorry XD)
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),  # in einheitliche größe bringen
        ToTensord(keys=["vol", "seg"]),  # am schluss in einen tensor umwandeln
    ]
)

# Infos zu pytorch tensor :
# https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html

# for comparing
# das gleiche wird auch für die valitdation daten definiert
validation_transforms = Compose(
    [
        LoadImaged(keys=["vol", "seg"]),
        AddChanneld(keys=["vol", "seg"]),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
        Resized(keys=["vol", "seg"], spatial_size=spatial_size),
        # ToTensord(keys=["vol", "seg"]),

    ]
)
# daten für die beiden sets werden geladen
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1)

validation_ds = Dataset(data=validation_files, transform=validation_transforms)
validation_loader = DataLoader(validation_ds, batch_size=1)

# data_dir = 'D:/Youtube/Organ and Tumor Segmentation/datasets/Task03_Liver/Data_Train_Test'
model_dir = 'C:\\Users\\ChiaraFreistetter\\Desktop\\fh\\inno-organ_segmentation\\results'
# damit sagen wir das wir auf dem gpu rechnen wollen ( achtung nicht mit allen grafikkarten kompatibel!!!)
device = torch.device("cuda:0")
# hier wird das UNet model definiert, auch hier bin ich nicht sicher was die einzelnen sachen genau aussagen
'''
 Layer aber warum 2 ouput channel? eve wegen dem label?
 down/upsampeling by factor 2 (stride 2)
 sind die convolutions pro layer
 dimentions hab ich zu spatial_dims geändert weils die neuere version ist
'''

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# hier wird der optimization alg. sowie die metric als variable gesetzt
loss = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optim = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)
# die brauchen wir dann weiter unten
max_epochs = 1
validation_interval = 1

# training part
# ein paar leere varaiblen zum später verwenden
best_metric = -1
best_metric_epoch = -1
save_loss_train = []
save_loss_validation = []
save_metric_train = []
save_metric_validation = []

# eine epoche
# eine epoche ist wie oft wir durch ALLE daten durchgeben, also im moment alles was in unseren ordner ist
# wenn wir die epoche höher setzten dann geht das ganze mehrmals durch die gleichen files
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
        volume, label = (volume.to(device), label.to(device))

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
                validation_volume, validation_label = (validation_volume.to(device), validation_label.to(device),)

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
                    model_dir, "best_metric_model.pth"))

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

