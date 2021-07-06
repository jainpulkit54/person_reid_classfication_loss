import os
import h5py
import torch
import numpy as np
from networks import *
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

os.makedirs('classification_loss_mars_lr', exist_ok = True)

query_images = '/home/pulkit/Datasets/market1501/query/'
gallery_images = '/home/pulkit/Datasets/market1501/bounding_box_test/'
hf_query = h5py.File('classification_loss_mars_lr/query_embeddings.h5', 'w')
hf_gallery = h5py.File('classification_loss_mars_lr/gallery_embeddings.h5', 'w')

images_ = sorted(os.listdir(gallery_images)) # Contains the image names present in the gallery
queries_ = sorted(os.listdir(query_images)) # Contains the image names present in the query set

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = FeatureNet()
checkpoint = torch.load('checkpoints_mars_classification/model_epoch_6.pth')
model_parameters = checkpoint['state_dict']
model.load_state_dict(model_parameters)
model.to(device)
model.eval()

resize = transforms.Resize(size = (256, 128))
totensor = transforms.ToTensor()

def extract_embeddings(dataset_path, dataset):

	print('Extracting Embeddings:')

	for batch_id, image_name in tqdm(enumerate(dataset), total = len(dataset)):
		
		img = Image.open(dataset_path + image_name).convert('RGB')
		img = resize(img)
		img = totensor(img)
		img = img.unsqueeze(0)
		img = img.to(device)
		
		with torch.no_grad():
			embeddings = model.get_embeddings(img)
		
		embeddings = embeddings.cpu().numpy()

		if batch_id == 0:
			embeddings_array = embeddings
		else:
			embeddings_array = np.append(embeddings_array, embeddings, axis = 0)

	return embeddings_array

embeddings_query = extract_embeddings(query_images, queries_)
embeddings_gallery = extract_embeddings(gallery_images, images_)

print(embeddings_gallery.shape)
print(embeddings_query.shape)

hf_query.create_dataset('emb', data = embeddings_query)
hf_gallery.create_dataset('emb', data = embeddings_gallery)

hf_query.close()
hf_gallery.close()