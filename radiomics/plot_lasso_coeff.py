from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from sklearn.model_selection import cross_val_score,train_test_split,KFold
import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np
def rmse_cv(model):
    rmse=-cross_val_score(model, df, cls, scoring="neg_mean_squared_error", cv = 10)
    return(rmse)
df = pd.read_csv("R_wichfake_features.csv",error_bad_lines=False)
cls=df.pop('label')
id=df.pop('id')
kf = KFold(10,random_state=0,shuffle=False)
Lambdas=np.logspace(-3.5,-0.5,200)
#´æ·ÅÆ«»Ø¹éÏµÊý
lasso_cofficients=[]
scores=[]
MS_S=[]
load=True
if not load:
    for Lambda in Lambdas:
        lasso=Lasso(alpha=Lambda)
        lasso.fit(df._values, cls._values)
        lasso_cofficients.append(lasso.coef_)
        x=rmse_cv(lasso)
        MS_S.append(np.mean(x))
        scores.append(x)
else:
    lasso_cofficients=np.load('coff.npy')
    scores=np.load('score.npy')
    MS_S=scores.mean(1)
    a=1
#scores.append(r)
plt.style.use('ggplot')
plt.plot(Lambdas,lasso_cofficients)
plt.xscale('log')
plt.xlabel('Log(alpha)')
plt.ylabel('Cofficients')
plt.title('Coffs of different alpha')
plt.savefig('Coff_test.jpg')

np.save('coff.npy',lasso_cofficients)
np.save('score.npy',scores)
np.save('MS_S.npy',MS_S)