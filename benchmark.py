import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
# from torch.autograd import Variable

from torchvision import transforms
from torchvision.datasets import FashionMNIST

# from octconv-old import OctConv2d, OctReLU, OctMaxPool2d
from networks import OctCNN, NormalCNN


use_cuda = torch.cuda.is_available()
batch_size = 512
num_epochs = 30
learning_rate = 1e-4

# Prepare datasets.
transform = transforms.Compose([transforms.ToTensor(),
																transforms.Normalize((.1307,), (.3081,))])
train_data = FashionMNIST(root='./data', download=True,
													train=True, transform=transform)
test_data = FashionMNIST(root='./data', download=False,
												 train=False, transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
											  	shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
												 shuffle=False)

# Construct network model and prepare for training.
model = OctCNN(alpha=0.2) # --- USING OCTCONV
# model = NormalCNN() # --- NOT USING OCTCONV
print("Total parameters: {}".format(
	sum(p.numel() for p in model.parameters()) ))
print("Trainable parameters: {}".format(
	sum(p.numel() for p in model.parameters() if p.requires_grad) ))
print("Convolution parameters: {}".format(
	sum([p.numel() for p in model.parameters() if p.requires_grad][:-4]) ))

if use_cuda:
	model = model.cuda()
	model = nn.DataParallel(model)
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop.
for epoch in range(num_epochs):
	for batch, (images, labels) in enumerate(train_loader):
		if use_cuda:
			images, labels = images.cuda(), labels.cuda()

		optimizer.zero_grad()
		outputs = model(images)
		loss = nn.CrossEntropyLoss()(outputs, labels)
		loss.backward()
		optimizer.step()

		if batch % 20 == 0:
			correct = 0
			total = 0
			for images, labels in test_loader:
				if use_cuda:
					images, labels = images.cuda(), labels.cuda()
				outputs = model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted==labels).sum().detach()
			accuracy = 100 * (correct.item() / total)

			print("Epoch {:2}/{}, Batch {:3} --- Loss={:8.5}, Acc={:5.3}%".format(
				epoch, num_epochs, batch, loss.item(), accuracy))

