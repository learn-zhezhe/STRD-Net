import os

import torch
import torchvision


# Read all the images required for train and val
def read_rs_images(rs_dir, is_train=True):
    # ----------------------------------------------------------------------------------------#
    #   Specify the .txt file containing the names of the data to be used for train and val
    # ----------------------------------------------------------------------------------------#
    txt_fname = os.path.join(rs_dir,'train.txt' if is_train else 'val.txt')

    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features_1, features_2, labels = [], [], []
    for i, fname in enumerate(images):
        features_1.append(torchvision.io.read_image(os.path.join(rs_dir, 'img_irrg', f'{fname}.jpg')))
        features_2.append(torchvision.io.read_image(os.path.join(rs_dir, 'img_ndvi', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(rs_dir, 'png', f'{fname}.png'), mode))

    return features_1, features_2, labels


# ------------------------------------------------------------------#
# RS_COLORMAP is the colormap corresponding to the labels
# rs_dir is the dataset directory
# ------------------------------------------------------------------#
RS_COLORMAP = [[0, 255, 255], [0, 255, 0],  [255, 255, 255]]
rs_dir = 'dataset/data'


def rs_colormap2label():
    """
    Establish a mapping from label RGB colors to class indices
    """
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(RS_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                       colormap[2]] = i
    return colormap2label

class RSDataset(torch.utils.data.Dataset):
    """
    Custom dataset reading
    """
    def __init__(self, is_train, crop_size, rs_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features_1, features_2, labels = read_rs_images(rs_dir, is_train=is_train)
        self.features_1 = [
            self.normalize_image(feature)
            for feature in self.filter(features_1)]
        self.features_2 = [
            self.normalize_image(feature)
            for feature in self.filter(features_2)]
        self.labels = self.filter(labels)
        self.colormap2label = rs_colormap2label()
        print('read ' + str(len(self.features_1)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float())

    def filter(self, imgs):
        return [
            img for img in imgs if (img.shape[1] >= self.crop_size[0] and
                                    img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature_1, feature_2, label = rs_rand_crop(self.features_1[idx], self.features_2[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature_1, feature_2, rs_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features_1)

def rs_rand_crop(feature_1, feature_2, label, height, width):
    """
    Randomly crop remote sensing images
    """
    rect = torchvision.transforms.RandomCrop.get_params(
        feature_1, (height, width))
    feature_1 = torchvision.transforms.functional.crop(feature_1, *rect)
    feature_2 = torchvision.transforms.functional.crop(feature_2, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature_1, feature_2, label

def rs_label_indices(colormap, colormap2label):
    """
    Map the RGB values in the RS labels to their corresponding class indices
    """
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 +
           colormap[:, :, 2])
    return colormap2label[idx]

def load_data_rs(batch_size, crop_size):
    """
    Load remote sensing image data
    """
    train_iter = torch.utils.data.DataLoader(
        RSDataset(True, crop_size, rs_dir), batch_size, shuffle=True,
        drop_last=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(
        RSDataset(False, crop_size, rs_dir), batch_size, drop_last=True,
        num_workers=0)
    return train_iter, test_iter