import os
import numpy as np
from sklearn.cluster import KMeans
from Bio.Cluster import kcluster
from Bio.Cluster import clustercentroids
from sklearn.metrics import silhouette_score
import seaborn as sb
import matplotlib.pyplot as plt
l=70
inpath='/mnt/data9/mp_NCPs/mp_analysis-x/gallary'
outpath='/mnt/data9/mp_NCPs/mp_analysis-x/cluster'
os.makedirs(outpath,exist_ok=True)
X=[]
Name=[]
mod='kmeans'
for item in os.listdir(inpath):
    data=np.load(os.path.join(inpath,item)).reshape((50*l))
    X.append(data)
    Name.append(item.split('.npy')[0])
if mod=='kmeans':
    #estimator = KMeans(n_clusters=6)

    #estimator.fit(X)
    #y_kmeans = estimator.predict(X)
    #centroids = estimator.cluster_centers_
    coef=[]
    x=range(3,20)
    for clusters in x:
        centroids, error, nfound = kcluster(X, clusters, dist='u', npass=100)
        silhouette_avg = silhouette_score(X, centroids, metric='cosine')
        coef.append(silhouette_avg)
    k=np.argmax(coef)+3
    print(k)
    centroids, error, nfound = kcluster(X, k, dist='u', npass=100)
    C=[]
    X=np.array(X)
    for i in range(k):
        C.append(X[centroids==i,:].mean(0))
    centroids=np.array(C)

else:
    AGE=[]
    AGE_M=[]
    AGE_F=[]
    ages_genders=open('mp_ages_genders.txt','r').readlines()
    namelist=[da.split('\t')[0].split('ill')[-1].replace('/','_') for da in ages_genders]
    idx=np.array([namelist.index(a) for a in Name])
    idx_data=np.array(ages_genders)[idx]
    CLS=[]
    for i in idx_data:
        sex_a=i.split('\t')[-1][:-1]=='M'
        age_a=int(i.split('\t')[-2])//20-1
        #print()
        cls=sex_a*3+age_a
        #print(age_a,cls)
        AGE.append(int(i.split('\t')[-2]))
        if sex_a==1:
            AGE_M.append(int(i.split('\t')[-2]))
        else:
            AGE_F.append(int(i.split('\t')[-2]))
        CLS.append(cls)
    plt.figure(1)
    plt.hist(AGE)
    plt.figure(2)
    plt.hist(AGE_M)
    plt.figure(3)
    plt.hist(AGE_F)
    plt.show()
    CLS=np.array(CLS)
    X=np.array(X)
    centroids=np.zeros((np.max(CLS),X.shape[1]))
    for i in range(np.max(CLS)):
        centroids[i]=X[CLS==i].mean(0)
centroids=[centroids[i,:].reshape((50,l))for i in range(centroids.shape[0])]
for i,c in enumerate(centroids):
    plt.figure()
    sb.heatmap(c, vmin=0, vmax=0.05, cmap='jet')
    plt.xlabel('Days after First Period')
    plt.ylabel('Nomalized Position along Z direction')
    plt.title('Heatmap for Typical Lesion Progress Patterns')
    plt.savefig(outpath+'/'+str(i) + '.jpg')
    plt.close()
    np.save(outpath+'/cluster'+str(i)+'.npy',c)

a=1