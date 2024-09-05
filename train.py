"""
Image Semantic Segmentation
"""
from dataset.dataset import *
from utils.train_function import *
from model.STRD_Net import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ------------------------------------------------------------------------------------#
#   Model training options: UNet, FCN, DeepLabv3plus, PSPNet, STRA_Net
#   The location for saving the trained model is reconfigured in utils.train_function
# ------------------------------------------------------------------------------------#
net = STRD_Net()
# ----------------------------------------------------------------------#
#   Specify hyperparameters, training epochs, learning rate, and weights
# ----------------------------------------------------------------------#
num_epochs = 2
lr = 8e-5
wd = 1e-5
# ------------------------------------------------------------------#
#   Specify whether to use GPU for training
# ------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------------------------#
#   Define batch_size and image size
# ------------------------------------------------------------------#
batch_size = 2
crop_size = (256, 256)

# Read the dataset
train_iter, test_iter = load_data_rs(batch_size, crop_size)
print(train_iter,test_iter)

# Define the optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
# Start training
train(net, train_iter, test_iter, dice_loss, focal_loss, optimizer, num_epochs, device)