import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

#from tiatoolbox import rcParam
from tiatoolbox.tools.tissuemask import MorphologicalMasker
from tiatoolbox.models.dataset.classification import WSIPatchDataset
from tiatoolbox.models.dataset.wsiclassification import WSILabelPatchDataset

import os
import pandas as pd

#mpl.rcParams['figure.dpi'] = 300 # for high resolution figure in notebook

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if device=='cuda:0': ON_GPU = True

wsi_dir = 'V:\Comparative-Computational\Pancreas\scans'
excel_dir = 'V:\Comparative-Computational\Pancreas\scans\Training_dataset_list_formatiert.xlsx'

class_dict={0: 'SS', 1: 'FFPE'}

def get_data(wsi_dir, excel_dir, class_dict):
    class_dict = {v: k for k, v in class_dict.items()}
    df=pd.read_excel(excel_dir, sheet_name="Slides").sort_values(by="ImageID")
    img_idx = list(df["ImageID"])
    paths=[]
    labels=[]
    for i, img_id in enumerate(img_idx):
        path = wsi_dir+os.sep+str(img_id)+".svs"
        label = class_dict[df.iloc[i]["Label"]]
        if os.path.isfile(path):
            paths.append(path)
            labels.append(label)
    return paths, labels

wsi_paths, labels = get_data(wsi_dir, excel_dir, class_dict)

res = 5
un = 'power'
patch_input_shape=[256, 256]
stride_shape=[240, 240]

train_ds = WSILabelPatchDataset(
    img_paths=wsi_paths[:35],
    img_labels=labels[:35],
    class_dict=class_dict,
    mode="wsi",
    save_dir="C:\\Users\\ge63kug\\source\\repos\\tiatoolbox\\HE_SS\\out_data",
    patch_input_shape=patch_input_shape,
    stride_shape=stride_shape,
    #preproc_func=None,
    auto_get_mask=True,
    resolution=res,
    units=un,
    thres=0.5,
    #ignore_background=False,
    #auto_mask_method="morphological",
    #kernel_size = 128,
    #min_region_size = 1000,
)

test_ds = WSILabelPatchDataset(
    img_paths=wsi_paths[35:],
    img_labels=labels[35:],
    class_dict=class_dict,
    mode="wsi",
    save_dir="C:\\Users\\ge63kug\\source\\repos\\tiatoolbox\\HE_SS\\out_data",
    patch_input_shape=patch_input_shape,
    stride_shape=stride_shape,
    #preproc_func=None,
    auto_get_mask=True,
    resolution=res,
    units=un,
    thres=0.5,
    #ignore_background=False,
    #auto_mask_method="morphological",
    #kernel_size = 128,
    #min_region_size = 1000,
)


# imports
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from datetime import datetime

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE=4
EPOCHS=2

# run name for tensorboard (to distinguish different runs)
# to access tensorboard, in cmd run: tensorboard --logdir=runs
# then open in browser: localhost:6006
timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
run_name = 'function_test'+timestamp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# dataloaders
trainloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=0)


testloader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=0)

# constant for classes
classes = ('SS', 'FFPE')

# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 7, 3)
        self.conv2 = nn.Conv2d(8, 16, 5, 1)
        
        self.fc1 = nn.Linear(16 * 19 * 19, 120)
        self.fc2 = nn.Linear(120, 2)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 19 * 19)
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, criterion, epoch, steps_per_epoch=5):
  # Switch model to training mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
  model.train()

  train_loss = 0
  train_total = 0
  train_correct = 0
  running_loss = 0
  step = len(train_loader)//steps_per_epoch

  # We loop over the data iterator, and feed the inputs to the network and adjust the weights.
  for batch_idx, (data, target) in enumerate(trainloader, start=0):
    
    # Load the input features and labels from the training dataset
    data, target = data.to(device), target.to(device)
    
    # Reset the gradients to 0 for all learnable weight parameters
    optimizer.zero_grad()
    
    # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-9 in this case)
    output = model(data)
    
    # Define our loss function, and compute the loss
    loss = criterion(output, target)
    train_loss += loss.item()
    running_loss += loss.item()

    scores, predictions = torch.max(output.data, 1)
    train_total += target.size(0)
    train_correct += int(sum(predictions == target))
            
    # Reset the gradients to 0 for all learnable weight parameters
    optimizer.zero_grad()

    # Backward pass: compute the gradients of the loss w.r.t. the model"s parameters
    loss.backward()
    
    # Update the neural network weights
    optimizer.step()
    
    if batch_idx % step == step - 1:
        running_writer.add_scalar('Loss', running_loss/(step*BATCH_SIZE), epoch*steps_per_epoch+batch_idx//step)
        running_loss=0

  acc = round((train_correct / train_total) * 100, 2)
  print("Epoch [{}], Loss: {:.5f}, Accuracy: {}".format(epoch, train_loss/train_total, acc), end="")
  train_writer.add_scalar('Loss', train_loss/train_total, (epoch+1)*steps_per_epoch-1)
  train_writer.add_scalar('Accuracy', acc, (epoch+1)*steps_per_epoch-1)
  train_writer.add_figure('Predictions vs. Actuals', plot_classes_preds(model, data, target),  global_step=(epoch+1)*steps_per_epoch-1)

def test(model, device, test_loader, criterion, epoch, steps_per_epoch=5):
  # Switch model to evaluation mode. This is necessary for layers like dropout, batchnorm etc which behave differently in training and evaluation mode
  model.eval()

  test_loss = 0
  test_total = 0
  test_correct = 0

  example_images = []
  with torch.no_grad():
      for data, target in test_loader:
          # Load the input features and labels from the test dataset
          data, target = data.to(device), target.to(device)
          
          # Make predictions: Pass image data from test dataset, make predictions about class image belongs to (0-9 in this case)
          output = model(data)
          
          # Compute the loss sum up batch loss
          test_loss += criterion(output, target).item()
          
          scores, predictions = torch.max(output.data, 1)
          test_total += target.size(0)
          test_correct += int(sum(predictions == target))

  acc = round((test_correct / test_total) * 100, 2)
  print(" Test_loss: {:.5f}, Test_accuracy: {}".format(test_loss/test_total, acc))
  test_writer.add_scalar('Loss', test_loss/test_total, (epoch+1)*steps_per_epoch-1)
  test_writer.add_scalar('Accuracy', acc, (epoch+1)*steps_per_epoch-1)

# helper functions

def matplotlib_imshow(img, one_channel=False):
    '''
    helper function to show an image
    '''
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def labeled_random_batch(data_loader):
    # get some random training images
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    # plot the images in the batch, along with true labels
    col_num=8
    row_num=(BATCH_SIZE-1)//col_num+1
    fig = plt.figure(figsize=(col_num*3, row_num*3))
    for idx in np.arange(BATCH_SIZE):
        ax = fig.add_subplot(row_num, col_num, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(classes[labels[idx]])

    # write to tensorboard
    running_writer.add_figure('labeled_batch', fig)

def add_embedding(data_set, n):
    # select random images and their target indices
    images, labels = select_n_random(data_set.data, data_set.targets)

    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]

    size=torch.tensor(images.shape[1:]).prod()

    # log embeddings
    features = images.view(-1, size)
    running_writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
    running_writer.close()

def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    col_num=8
    row_num=(BATCH_SIZE-1)//col_num+1
    fig = plt.figure(figsize=(col_num*3, row_num*3))
    for idx in np.arange(BATCH_SIZE):
        ax = fig.add_subplot(row_num, col_num, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    running_writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    running_writer.close()

def create_pr_curves(net, data_loader):
    # 1. gets the probability predictions in a test_size x num_classes Tensor
    # 2. gets the preds in a test_size Tensor
    # takes ~10 seconds to run
    class_probs = []
    class_label = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            output = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]

            class_probs.append(class_probs_batch)
            class_label.append(labels)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_label = torch.cat(class_label)

    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_label)

def create_confusion_matrix(net, data_loader):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for inputs, labels in data_loader:
        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth


    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()


#initialize network, loss function and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# initialize SummmaryWriters, default `log_dir` is "runs" - added child-dirs for train, test and others
train_writer = SummaryWriter('runs/'+run_name+'/train')
test_writer = SummaryWriter('runs/'+run_name+'/test')
running_writer = SummaryWriter('runs/'+run_name+'/running')

# show one batch to check if dataset is correctly initialized
labeled_random_batch(trainloader)

# create projector thing
add_embedding(trainset, 100)

#training loop
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    train(net, device, trainloader, optimizer, criterion, epoch)
    test(net, device, testloader, criterion, epoch)

print('Finished Training')

create_pr_curves(net, testloader)
running_writer.add_figure("Confusion matrix", create_confusion_matrix(net, testloader))
