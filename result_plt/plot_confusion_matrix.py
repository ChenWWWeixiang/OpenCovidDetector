import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
data=np.load('../key_result/test.npy')
y_pred=np.array(data[:,1:-1],np.float)
y_pred=np.argmax(y_pred,1)
y_true = np.array(data[:,-1],np.uint8)
sns.set()
f,ax=plt.subplots()


C2= confusion_matrix(y_true, y_pred,labels=[0,1,2,3],normalize=True)
print(C2) #
sns.heatmap(C2,annot=True,ax=ax,vmax=73,cmap='Blues',fmt="d")

ax.set_title('Confusion Matrix')
ax.set_xlabel('Prediction')
ax.set_ylabel('Ground Truth')
plt.xticks((0.5,1.5,2.5,3.5),['Healthy','CAP','Influenza','COVID-19'])
plt.yticks((0.5,1.5,2.5,3.5),['Healthy','CAP','Influenza','COVID-19'])
plt.savefig('jpgs/confusionmatrix.jpg')
plt.show()
