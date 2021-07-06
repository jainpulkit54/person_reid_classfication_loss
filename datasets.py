import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms

# Use this for folder wise images

class ImageFolder(data.Dataset):

	def __init__(self, folder_path):
		self.folder_path = folder_path
		self.targets_name = sorted(os.listdir(self.folder_path))
		# subfolder_name = []
		# images_name = []
		# targets = []
		train_subfolder_name = []
		val_subfolder_name = []
		train_images_name = []
		val_images_name = []
		train_targets = []
		val_targets = []
		
		for index, folder_name in enumerate(self.targets_name):
			path = os.listdir(self.folder_path + folder_name + '/')
			n_images = len(path)
			half = int(n_images/2)
			train_subfolder_name.extend([(folder_name + '/')] * half)
			val_subfolder_name.extend([(folder_name + '/')] * (n_images - half))
			train_images_name.extend(path[0: half])
			val_images_name.extend(path[half: n_images])
			train_targets.extend([index]*half)
			val_targets.extend([index]*(n_images - half))
			
			# subfolder_name.extend([(folder_name + '/')]*n_images)
			# images_name.extend(path)
			# targets.extend([index]*n_images)
		
		# self.subfolder_name = subfolder_name
		# self.images_name = images_name
		# self.targets = targets
		# self.total_samples = len(self.targets)

		self.train_subfolder_name = train_subfolder_name
		self.val_subfolder_name = val_subfolder_name
		self.train_images_name = train_images_name
		self.val_images_name = val_images_name
		self.train_targets = train_targets
		self.val_targets = val_targets
		self.train_total_samples = len(self.train_targets)
		self.val_total_samples = len(self.val_targets)
		self.totensor = transforms.ToTensor()
		self.horizontal_flip = transforms.RandomHorizontalFlip(p = 0.5)
		self.resize = transforms.Resize(size = (256, 128))

	def __getitem__(self, index):
		
		img = Image.open(self.folder_path + self.train_subfolder_name[index] + self.train_images_name[index]).convert('RGB')
		img = self.resize(img)
		img = self.horizontal_flip(img)
		img = self.totensor(img)
		target = self.train_targets[index]
		
		return img, target

	def __len__(self):
		
		return self.train_total_samples

class ImageFolder_val(data.Dataset):

	def __init__(self, folder_path, images_name, targets, subfolder_name):
		
		self.resize = transforms.Resize(size = (256, 128))
		self.totensor = transforms.ToTensor()
		self.folder_path = folder_path
		self.images_name = images_name
		self.subfolder_name = subfolder_name
		self.targets = targets

	def __getitem__(self, index):
		
		img = Image.open(self.folder_path + self.subfolder_name[index] + self.images_name[index]).convert('RGB')
		img = self.resize(img)
		img = self.totensor(img)
		target = self.targets[index]

		return img, target

	def __len__(self):

		return len(self.images_name)

class myBatchSampler(data.BatchSampler):

	def __init__(self, sampler, train_dataset, n_classes, n_samples):
		self.sampler = sampler
		self.train_dataset = train_dataset
		self.n_classes = n_classes
		self.n_samples = n_samples
		self.batch_size = self.n_classes * self.n_samples
		self.total_samples = len(self.sampler)
		self.targets = self.train_dataset.train_targets
		self.targets = np.array(self.targets)
		self.labels_set = set(self.targets)
		self.labels_to_indices = {label: np.where(self.targets == label)[0] for label in self.labels_set}
		self.classwise_used_label_to_indices = {label: 0 for label in self.labels_set}

	def __iter__(self):
		
		self.count = 0
		while(self.count + self.batch_size < self.total_samples):
			self.count = self.count + self.batch_size
			labels_chosen = np.random.choice(list(self.labels_set), self.n_classes, replace = False)
			batch_images_indices = []
			for label in labels_chosen:
				indices = self.labels_to_indices[label][self.classwise_used_label_to_indices[label]:
				(self.classwise_used_label_to_indices[label] + self.n_samples)]
				if len(indices) < self.n_samples:
					indices_to_add = self.n_samples - len(indices)
					indices = list(indices)
					for i in range(indices_to_add):
						indices.append(indices[i])
				batch_images_indices.extend(indices)
				self.classwise_used_label_to_indices[label] += self.n_samples
				if self.classwise_used_label_to_indices[label] + self.n_samples > len(self.labels_to_indices[label]):
					np.random.shuffle(self.labels_to_indices[label])
					self.classwise_used_label_to_indices[label] = 0

			yield batch_images_indices

	def __len__(self):

		return self.total_samples // self.batch_size
