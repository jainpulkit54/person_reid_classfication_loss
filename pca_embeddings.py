import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class_num = np.load('mnist_train_set_embeddings/images_class.npy')
image_embeddings = np.load('mnist_train_set_embeddings/images_embeddings.npy')

pca = PCA(n_components = 3)
x = pca.fit_transform(image_embeddings)
np.random.seed(0)

def plot_embeddings(embeddings, targets):

	fig = plt.figure()
	ax = Axes3D(fig)
	legend = []
	colors = np.random.rand(10, 3)
	for i in range(10):

		legend.append(str(i))
		inds = np.where(targets == i)[0]
		x = embeddings[inds, 0]
		y = embeddings[inds, 1]
		z = embeddings[inds, 2]
		ax.scatter(x, y, z, alpha = 1, color = colors[i, :])

	plt.legend(legend)
	plt.savefig('mnist_train_set_embeddings_pca.png')
	plt.show()

plot_embeddings(x, class_num)