# Hot Dog Recognition
# We'll use ResNet to classify whether an image is of hot dog or not. We'll also leverage transfer learning to acomplish this task,l
# the model will be fine-tuned on ResNet model pre-trained on ImageNet dataset.

# Fine Tunning
# 1. Pre-train a neural network model,
# 2. Create a new neural network model, i.e., the target model. This replicates
# all model designs and their parameters, except the output layer.
# 3. Add an output layer whose output size is the number of tartget dataset categories
# to the target model, and randomly initialize the model parameters of this layer.
# 4. Train the target model on a target dataset.Train the output layer from scratch, 
# while the parameters of al remaining layers are fine-tuned based on the parameters of the source model.

from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os

# Obtaining the Dataset
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

# instances to read all the image files in the training dataset and testing dataset, respectively
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# During Training: crop a random area with random size and random aspect ration from the image
# then scale the area to an input with a height and width of 224 pixels.

# During Testing: scale the height and width of image to 256 pixels
# then crop the center area with height and width of 224 pixels to use as the input
# normalize the values of the three RGB color channels
# the average of all values of the channel is subtractred from each value and then the result is divided by the 
# standrard deviation of all values of the channel to produce the output.

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize
])

# Defining and Initializing the Model

pretrained_net = torchvision.models.resnet18(pretrained=True)

# target model
# the number of outputs is equal to the number of categories in the target dataset
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)

# Fine Tune the Model

# uses fine tuning to be called multiple times
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs), batch_size= batch_size,
        shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform= test_augs), 
        batch_size=batch_size)
    
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
            if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x}, {'params': net.fc.parameters(), 'lr': learning_rate * 10}], lr= learning_rate, weight_decay = 0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    
    d2l.train(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

train_fine_tuning(finetune_net, 5e-5)


