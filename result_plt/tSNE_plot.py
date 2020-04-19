import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

x = np.load('../deep_f/X.npy')
c=np.load('../deep_f/Y.npy')

#ts = TSNE(n_components=2)

#ts.fit_transform(x)
plt.figure(figsize=(10,10))
#y = ts.fit_transform(x)
y=np.load('../deep_f/T_X.npy')
#np.save('../deep_f/T_X.npy',y)
for i in range(3):
    plt.scatter(y[c==i, 0], y[c==i, 1])
plt.title('t-SNE Curve', fontsize=14)
plt.legend(['normal','CAP','COVID19'])
plt.savefig('jpgs/tSNE.jpg')
plt.show()
