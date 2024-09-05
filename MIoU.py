import os

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Dataset classification labels
RS_COLORMAP = [[0, 255, 255], [0, 255, 0],  [255, 255, 255]]

# Randomly crop remote sensing images
def rs_rand_crop(feature_1, feature_2, label, height, width):
    rect = torchvision.transforms.RandomCrop.get_params(
        feature_1, (height, width))
    feature_1 = torchvision.transforms.functional.crop(feature_1, *rect)
    feature_2 = torchvision.transforms.functional.crop(feature_2, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature_1, feature_2, label

# Map the RGB values in the RS labels to their corresponding class indices
def rs_label_indices(colormap, colormap2label):
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 +
           colormap[:, :, 2])
    return colormap2label[idx]

# Define a function to establish a mapping from label RGB colors to class indices
def rs_colormap2label():
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(RS_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                       colormap[2]] = i
    return colormap2label

# Read all images required for training and validation
def read_rs_images(rs_dir, is_train=False):
    # ------------------------------------------------------------------#
    #   Specify the dataset to be used for evaluation
    # ------------------------------------------------------------------#
    txt_fname = os.path.join(rs_dir,'test.txt' if is_train else 'test.txt')

    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features_1, features_2, labels = [], [], []
    for i, fname in enumerate(images):
        # ---------------------------------------------------------------------------#
        #   Specify the folder address of the dataset corresponding to the .txt file
        # ---------------------------------------------------------------------------#
        features_1.append(
            torchvision.io.read_image(
                os.path.join(rs_dir, 'img_irrg', f'{fname}.jpg')))
        features_2.append(
            torchvision.io.read_image(
                os.path.join(rs_dir, 'img_ndvi', f'{fname}.jpg')))
        labels.append(
            torchvision.io.read_image(
                os.path.join(rs_dir, 'png', f'{fname}.png'),
                mode))

    return features_1, features_2, labels

# Custom data loading class for data reading
class RSDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, rs_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features_1, features_2, labels = read_rs_images(rs_dir, is_train=False)
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
        feature_1, feature_2, label = rs_rand_crop(self.features_1[idx], self.features_2[idx],
                                                   self.labels[idx], *self.crop_size)
        return (feature_1, feature_2, rs_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features_1)

# Load remote sensing image data
def load_data_rs(batch_size, crop_size):

    train_iter = torch.utils.data.DataLoader(
        RSDataset(True, crop_size, rs_dir), batch_size, shuffle=True,
        drop_last=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(
        RSDataset(False, crop_size, rs_dir), batch_size, drop_last=True,
        num_workers=0)
    return train_iter, test_iter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, crop_size = 6, (256, 256)
train_iter, test_iter = load_data_rs(batch_size, crop_size)

def predict(img_1, img_2):
    X_1 = test_iter.dataset.normalize_image(img_1).unsqueeze(0)
    X_2 = test_iter.dataset.normalize_image(img_2).unsqueeze(0)
    X_1 = X_1.to(device)
    X_2 = X_2.to(device)
    # ------------------------------------------------------------------#
    #   Select the trained weights of the corresponding model,
    #   with the model and weights saved in the 'module_data' directory
    # ------------------------------------------------------------------#
    module = torch.load("save_model/strdnet/", map_location=device)
    pred = module(X_1, X_2).argmax(dim=1)
    pred_img = pred.reshape(pred.shape[1], pred.shape[2])
    # pred_np = pred.squeeze().cpu().numpy()S
    return pred_img

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

rs_dir = 'dataset/data'
crop_rect = (0, 0, 256, 256)
test_images_1, test_images_2, test_labels = read_rs_images(rs_dir, False)
n = len(test_images_1)

def inference(num_classes):

    # Initialize performance metrics
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    tn = np.zeros(num_classes)

    for i in tqdm(range(n)):
        # Obtain the predicted values
        X_1 = torchvision.transforms.functional.crop(test_images_1[i], *crop_rect)
        X_2 = torchvision.transforms.functional.crop(test_images_2[i], *crop_rect)
        pred = predict(X_1, X_2).cpu()
        # Obtain the labels
        colormap2label = rs_colormap2label()
        label = torchvision.transforms.functional.crop(test_labels[i], *crop_rect)
        label_indices = rs_label_indices(label, colormap2label)
        label = label_indices
        label = np.array(label)
        pred = np.array(pred)

        # Initialize performance metrics
        for cat in range(num_classes):
            tp[cat] += ((pred == cat) & (label == cat)).sum()
            fp[cat] += ((pred == cat) & (label != cat)).sum()
            fn[cat] += ((pred != cat) & (label == cat)).sum()
            tn[cat] += ((pred != cat) & (label != cat)).sum()

    # Calculate performance metrics
    iou = np.divide(tp, (tp + fp + fn))
    pre = np.divide(tp, (tp + fp))
    recall = np.divide(tp, (tp + fn))
    f1 = np.divide(2 * pre * recall, (pre + recall))
    acc = np.divide((tp + tn).sum(), (tp + fn + fp + tn).sum())


    # Print performance metrics
    print('---------------------------------------------------')
    print('IOU:  ', iou)
    print('mIOU: ', iou.mean())
    print('pre:  ', pre)
    print('recall:  ', recall)
    print('F1', f1)
    print('Ave.F1', f1.mean())
    print('Acc', acc)
    print('---------------------------------------------------')

    return "Testing Finished!"

if __name__ == '__main__':
    num_classes = 3
    inference(num_classes)