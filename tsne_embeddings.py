import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class_num = np.load('images_class.npy')
image_embeddings = np.load('images_embeddings.npy')
image_embeddings = np.transpose(image_embeddings, (0,2,1))
image_embeddings = np.squeeze(image_embeddings)

tsne = TSNE(n_components = 3)
x = tsne.fit_transform(image_embeddings)
np.random.seed(0)

def plot_embeddings(embeddings, targets):

	fig = plt.figure()
	ax = Axes3D(fig)
	legend = []
	colors = np.random.rand(751, 3)
	for i in range(751):

		if i >= 80 and i <= 120:

			legend.append(str(i))
			inds = np.where(targets == i)[0]
			x = embeddings[inds, 0]
			y = embeddings[inds, 1]
			z = embeddings[inds, 2]
			ax.scatter(x, y, z, alpha = 1, color = colors[i, :])

	plt.legend(legend)
	plt.savefig('market_1501_test_set_embeddings.png')
	plt.show()

plot_embeddings(x, class_num)