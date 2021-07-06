import os
import numpy as np
import torch
from PIL import Image

images_path = '/home/pulkit/Datasets/market1501/bounding_box_train/'
folder = 'market1501_train/'

images_names = sorted(os.listdir(images_path))
images_names = images_names[:-1]

for name in images_names:
	distractor_id = name[0:2]
	if distractor_id == '-1':
		print('Distractor Found')
		pass
	else:
		person_id = name[0:4]
		int_id = int(person_id)
		if int_id == 0:
			pass
		else:
			os.makedirs(folder + person_id, exist_ok = True)
			img = Image.open(images_path + name).convert('RGB')
			img.save(folder + person_id + '/' + name)