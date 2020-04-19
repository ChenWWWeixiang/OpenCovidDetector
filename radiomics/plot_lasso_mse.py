import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from sklearn.model_selection import cross_val_score,train_test_split,KFold
import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input_csv", help="input file's name", type=str,
                    default='new_r_features.csv')

args = parser.parse_args()
df = pd.read_csv(args.input_csv,error_bad_lines=False)
cls=df.pop('label').astype(int)

id=df.pop('id')
M=df.max()._values
m=df.min()._values
#np.save('M.npy',M)
#np.save('m.npy',m)
df=(df - df.min()) / (df.max() - df.min())

#Lambdas=np.logspace(-7,0,100)
model = LassoCV(cv=10).fit(df._values, cls._values)

# Display results
m_log_alphas = -np.log10(model.alphas_)

plt.figure()
#ymin, ymax = 2300, 3800
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate')
print(model.alpha_)
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Alpha and MSEs for LASSO')
plt.axis('tight')
#plt.ylim(ymin, ymax)
plt.savefig('Lasso_alphas_choose.jpg')

lasso_cofficients=[]
for Lambda in model.alphas_:
    lasso = Lasso(alpha=Lambda)
    lasso.fit(df._values, cls._values)
    lasso_cofficients.append(lasso.coef_)
plt.figure()
plt.style.use('ggplot')
print(model.alpha_)
plt.plot(m_log_alphas,lasso_cofficients)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate')
#plt.xscale('log')
plt.xlabel('-Log(alpha)')
plt.ylabel('Cofficients')
plt.title('Coffs of different alpha')
plt.savefig('Coff_test.jpg')


lut = dict(zip(cls.unique(), "rbg"))
row_colors = cls.map(lut)

sns.clustermap(df, row_colors=row_colors,standard_scale=1,figsize=(20, 20))
plt.title('Cluster Heatmap of Features Before LASSO',x=10,y=1,fontsize=20)
plt.savefig("BeforeCHM.jpg")
corr = df.corr()
f,ax= plt.subplots(figsize = (15, 15),nrows=1)
# cubehelix mapÑÕÉ«/home/cwx/extra/MVI-R
sns.heatmap(corr, ax = ax, vmax=1, vmin=-1)
ax.set_title('Corr matrix before LASSO',fontsize=20)
f.savefig('corr_heatmap_before.jpg', bbox_inches='tight')
#plt.show()


model = Lasso(alpha=model.alpha_)
model.fit(df._values, cls._values)

coef = pd.Series(model.coef_, index = df.columns)
#coef[coef.abs()<3e-3]=0
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
#print(rmse_cv(model).mean())

after_df = df.iloc[:,(coef!=0)._values]
# matplotlib colormap
f,ax= plt.subplots(figsize = (7, 7),nrows=1)
sns.heatmap(after_df.corr(), ax = ax, vmax=1, vmin=-1,)
ax.set_title('Corr matrix after LASSO',fontsize=20)
f.savefig('corr_heatmap_after.jpg', bbox_inches='tight')
#plt.show()
sns.clustermap(after_df, row_colors=row_colors,standard_scale=1,figsize=(10, 20))
plt.title('Cluster Heatmap of Features After LASSO',x=10,y=1,fontsize=20)
plt.savefig("After_CHM.jpg")



#saving=np.save('coefs.npy',np.stack([name,co],-1))
#matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
plt.figure(figsize=(18,6))
imp_coef = pd.concat([coef.sort_values().head(5),
                     coef.sort_values().tail(5)])
imp_coef.plot(kind = "barh",fontsize=7)
print(imp_coef.keys())
#print(imp_coef.values)
plt.axis('tight')
plt.title("Coefficients in the Lasso Model",fontsize=20)
plt.savefig("coffs.jpg", bbox_inches='tight')
joblib.dump(model, "train_model.m")