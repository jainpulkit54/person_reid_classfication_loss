import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from PIL import Image
from networks import *
from datasets import *

writer = SummaryWriter('logs_market1501_classification')
images_path = '../market1501_train/'
os.makedirs('checkpoints_market1501_classification', exist_ok = True)
n_classes = 4
n_samples = 4
batch_size = n_classes * n_samples
dataset = ImageFolder(folder_path = images_path)
val_dataset = ImageFolder_val(folder_path = images_path, images_name = dataset.val_images_name, targets = dataset.val_targets, subfolder_name = dataset.val_subfolder_name)
mySampler = SequentialSampler(dataset)
myBatchSampler = myBatchSampler(mySampler, dataset, n_classes = n_classes, n_samples = n_samples)
train_loader = DataLoader(dataset, shuffle = False, num_workers = 0, batch_sampler = myBatchSampler)
val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = True, num_workers = 0)

no_of_training_batches = len(train_loader)/batch_size
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 100

featureNet = FeatureNet()
optimizer = optim.Adam(featureNet.parameters(), lr = 1e-3, betas = (0.9, 0.999))
criterion = nn.CrossEntropyLoss(reduction = 'mean')
softmax_activation = nn.Softmax(dim = 1)

def run_epoch(data_loader, model, optimizer, epoch_count = 0):

	model.to(device)
	model.train()

	running_loss = 0.0
	print("Training.....")

	for batch_id, (imgs, labels) in tqdm(enumerate(data_loader), total = len(data_loader)):

		iter_count = epoch_count * len(data_loader) + batch_id
		imgs = imgs.to(device)
		labels = labels.to(device)
		features = model.get_features(imgs)
		batch_loss = criterion(features, labels)
		optimizer.zero_grad()

		batch_loss.backward()
		optimizer.step()
		running_loss = running_loss + batch_loss.item()
		
		# Adding the logs in Tensorboard
		writer.add_scalar('Classification Loss on Training Set', batch_loss.item(), iter_count)
	
	return running_loss

def predict_train_accuracy(data_loader, model, checkpoint_path, epoch_count):
		
	checkpoint = torch.load(checkpoint_path)
	model_parameters = checkpoint['state_dict']
	model.load_state_dict(model_parameters)
	model.to(device)
	model.eval()

	correctly_predicted_samples = 0
	total_samples = 0
	print("Evaluating on Train Dataset.....")

	for batch_id, (imgs, labels) in tqdm(enumerate(data_loader), total = len(data_loader)):

		imgs = imgs.to(device)
		
		with torch.no_grad():
			features = model.get_features(imgs).cpu()
		
		probabilities = softmax_activation(features)
		predicted_classes = torch.argmax(probabilities, dim = 1)
		correct = torch.sum(torch.eq(predicted_classes, labels).type(torch.int32))
		correctly_predicted_samples += correct
		total_samples += len(labels)

	accuracy = (correctly_predicted_samples/total_samples) * 100
	# Adding the logs in Tensorboard
	writer.add_scalar('Classification Accuracy on Training Set', accuracy.item(), epoch_count)

	return accuracy.item()

def predict_val_accuracy(data_loader, model, checkpoint_path, epoch_count):
	
	checkpoint = torch.load(checkpoint_path)
	model_parameters = checkpoint['state_dict']
	model.load_state_dict(model_parameters)
	model.to(device)
	model.eval()

	correctly_predicted_samples = 0
	total_samples = 0
	print("Evaluating on Validation Dataset.....")

	for batch_id, (imgs, labels) in tqdm(enumerate(data_loader), total = len(data_loader)):

		imgs = imgs.to(device)

		with torch.no_grad():
			features = model.get_features(imgs).cpu()

		probabilities = softmax_activation(features)
		predicted_classes = torch.argmax(probabilities, dim = 1)
		correct = torch.sum(torch.eq(predicted_classes, labels).type(torch.int32))
		correctly_predicted_samples += correct
		total_samples += len(labels)

	accuracy = (correctly_predicted_samples/total_samples) * 100
	# Adding the logs in Tensorboard
	writer.add_scalar('Classification Accuracy on Validation Set', accuracy.item(), epoch_count)

	return accuracy.item()

def fit(train_loader, model, optimizer, n_epochs):

	print('Training Started\n')
	
	for epoch in range(n_epochs):
		
		loss = run_epoch(train_loader, model, optimizer, epoch_count = epoch)
		loss = loss/no_of_training_batches

		print('Training Loss after epoch ' + str(epoch + 1) + ' is:', loss)
		torch.save({'state_dict': model.cpu().state_dict()}, 'checkpoints_market1501_classification/model_epoch_' + str(epoch + 1) + '.pth')

		checkpoint_path = 'checkpoints_market1501_classification/model_epoch_' + str(epoch + 1) + '.pth'
		training_accuracy = predict_train_accuracy(train_loader, model, checkpoint_path, epoch)
		print('Training Classification Accuracy after epoch ' + str(epoch + 1) + ' is:', training_accuracy)
		
		val_accuracy = predict_val_accuracy(val_loader, model, checkpoint_path, epoch)
		print('Validation Classification Accuracy after epoch ' + str(epoch + 1) + ' is:', val_accuracy)

fit(train_loader, featureNet, optimizer = optimizer, n_epochs = epochs)