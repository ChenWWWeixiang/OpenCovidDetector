import numpy as np
from sklearn.manifold import TSNE

x = np.array([[0, 0, 0, 1, 2], [0, 1, 1, 3, 5], [1, 0, 1, 7, 2], [1, 1, 1, 10, 22]])

ts = TSNE(n_components=2)

ts.fit_transform(x)

print(ts.embedding_)
