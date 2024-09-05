import os

from utils.Time import *
from utils.Loss import *
from torch.utils.tensorboard import SummaryWriter


size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)


# Create two SummaryWriter instances
train_writer = SummaryWriter("logs_strdnet/train")
test_writer = SummaryWriter("logs_strdnet/test")


# Calculate the number of pixels that are predicted correctly
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


class Accumulator:
    """
    Define the Accumulator class to sum up n variables
    """
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Calculate the model's accuracy on the dataset using GPU
def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)

    with torch.no_grad():
        for X1, X2, y in data_iter:
            if isinstance(X1, list):
                X1 = [x.to(device) for x in X1]
            else:
                X1 = X1.to(device)
            if isinstance(X2, list):
                X2 = [x.to(device) for x in X2]
            else:
                X2 = X2.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X1, X2), y), size(y))
            time.sleep(0.01)
    return metric[0] / metric[1]


# Call the FocalLoss function
focal_loss = FocalLoss(alpha=0.25, gamma=2, reduction='mean')

# Call the DiceLoss function
dice_loss = DiceLoss(n_classes=3)

def train_batch(net, X1, X2, y, dice_loss, focal_loss, optimizer, device):
    X1 = X1.to(device)
    X2 = X2.to(device)
    y = y.to(device)
    time.sleep(0.01)
    net.train()
    optimizer.zero_grad()
    pred = net(X1, X2)
    # Define the hybrid loss
    # loss_ce = ce_loss(pred, y)
    loss_focal = focal_loss(pred, y)
    loss_dice = dice_loss(pred, y, softmax=True)
    loss = 0.2 * loss_focal + 0.8 * loss_dice
    loss.sum().backward()
    optimizer.step()
    train_loss_sum = loss.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train(net, train_iter, test_iter, dice_loss, focal_loss, optimizer, num_epochs, device):

    timer = Timer()
    net = net.to(device)

    # Define the learning rate cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-5)

    for epoch in range(num_epochs):
        # Four dimensions: training loss, training accuracy, number of instances, number of features
        metric = Accumulator(4)
        # for i, (features_1, features_2, labels) in tqdm(enumerate(train_iter), desc="Training"):
        for i, (features_1, features_2, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch(
                net, features_1, features_2, labels, dice_loss, focal_loss, optimizer, device)
            metric.add(l, acc, labels.shape[0], labels.numel())
            optimizer.step()
            time.sleep(0.01)
            timer.stop()
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # Update the learning rate lr
        scheduler.step()
        # Record the learning rate, loss, and accuracy during the training process
        train_writer.add_scalar("train_loss", metric[0] / metric[2], epoch)
        train_writer.add_scalar("lr_2", optimizer.param_groups[0]['lr'], epoch)
        train_writer.add_scalar("acc", metric[1] / metric[3], epoch)
        # Record the accuracy during the testing process
        test_writer.add_scalar("acc", test_acc, epoch)

        # Save the model
        module = net
        folder_path = 'save_model/strdnet'
        os.makedirs(folder_path, exist_ok=True)
        file_name = 'va_11011_strdnet_{}.pth'.format(epoch + 1)
        file_path = folder_path + '/' +file_name
        torch.save(module, file_path)
        print(f"The model for epoch {epoch+1} has been saved")
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
              f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(device)}')
    print(f'time {timer.sum()}sec')

    train_writer.close()
    test_writer.close()