import numpy as np
data=np.load('../key_result/ab_number_count.npy')
rate=np.array(data[:,1],np.float)
name=np.array(data[:,0]).tolist()
age=name

a=1