#import libraries
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#import data
cc_data = pd.read_csv('CC GENERAL.csv')

#####EDA

#look at distributions
cc_data.describe()

#look at formatting
cc_data.head()

#look for null values and at dtypes
cc_data.info()

#look at null values for minimum_payments
cc_data[cc_data.MINIMUM_PAYMENTS.isna()]

#look at null values for credit_limit
cc_data[cc_data.CREDIT_LIMIT.isna()]

#look at CUST_ID tail
cc_data.tail()

#look at data 3 std from mean
cc_data[np.abs(stats.zscore(cc_data.loc[:, (cc_data.columns != 'CUST_ID')])) >= 3]

#histograms of all columns except cust_id
for i in cc_data.loc[:, cc_data.columns != 'CUST_ID']:
    plt.hist(cc_data[i], edgecolor='black')
    plt.xticks()
    plt.xlabel(i)
    plt.ylabel('number of people')
    plt.show()
    
#heat map to find extreme positive and negative correlations
plt.figure(figsize=(16, 6))
sns.heatmap(cc_data.loc[:, (cc_data.columns != 'CUST_ID')].corr(), annot=True)
plt.title('Correlation Heatmap for Numerical Variables', fontdict={'fontsize':12}, pad=12);

#pairplot to show plots against each variable
sns.pairplot(cc_data.loc[:, (cc_data.columns != 'CUST_ID')])
plt.show()

####impute

#use simpleimputer to impute using the median
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
cc_data['MINIMUM_PAYMENTS'] = imputer.fit_transform(cc_data['MINIMUM_PAYMENTS'].values.reshape(-1,1))
cc_data['CREDIT_LIMIT'] = imputer.fit_transform(cc_data['CREDIT_LIMIT'].values.reshape(-1,1))

####PCA

#select all data except CUST_ID
cc_data_for_PCA = cc_data.loc[:, (cc_data.columns != 'CUST_ID')]

#standardize
cc_data_for_PCA_scaled = (cc_data_for_PCA - cc_data_for_PCA.mean(axis=0)) / cc_data_for_PCA.std(axis=0)

from sklearn.decomposition import PCA

#create principal components (2 axes based on elbow method below)
pca = PCA(2)
cc_data_pca = pca.fit_transform(cc_data_for_PCA_scaled)

#convert to dataframe
component_names = [f"PC{i+1}" for i in range(cc_data_pca.shape[1])]
cc_data_pca = pd.DataFrame(cc_data_pca, columns=component_names)

#plot data using principal components
sns.scatterplot(x=cc_data_pca.loc[:,'PC1'],y=cc_data_pca.loc[:,'PC2'])
plt.show()

#determine loadings
loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=cc_data.loc[:, (cc_data.columns != 'CUST_ID')].columns,  # and the rows are the original features
)
loadings

#PC1 is characterized by how much the card owner purchases using card
#PC2 is characterized by how much the card owner takes cash advances using card

#determine % explained variance and use % cumulative variance for elbow method to determine number of PCs

def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs

plot_variance(pca);

#about 50% of the variance is explained by these principal components
#2 PCA chosen based on the elbow method

#####cluster analysis

#standardize for kmeans
from sklearn.preprocessing import StandardScaler

kmeans_columns = ['PURCHASES','ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE']
cc_data_kmeans = cc_data.loc[:, kmeans_columns]

standardize = StandardScaler()
cc_data_kmeans = standardize.fit_transform(cc_data_kmeans)

#generate clusters to determine different clusters to advertise to

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state = 1)
cc_data["Ad_Groups"] = kmeans.fit_predict(cc_data_kmeans)

cc_data["Ad_Groups"] = cc_data["Ad_Groups"].astype("category")

#plot variables against clusters

for i in ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']:
    sns.violinplot(x ='Ad_Groups',y=i,data=cc_data)
    plt.show()
    
# Cluster characteristics and what to advertise

# Cluster 0: mostly installment purchases and some oneoff purchases
# high priced item advertisements broken into installments such as infomercial items, furniture, etc.
# Cluster 1: small item purchases
# local advertisements for low priced items such as best buy, restaurants, etc. or items off of amazon
# Cluster 2: mostly oneoff purchases and some installment purchases
# high priced item advertisements such as electronics, furniture, household appliances, etc.
# Cluster 3: cash advances
# loan advertisements

#####preparing data for classification

#create X and y variables
X = cc_data.loc[:,kmeans_columns]
y = cc_data['Ad_Groups']

#import libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#create dummy variables for y
y = pd.get_dummies(y)

# train/test split with stratify making sure classes are evenlly represented across splits
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.75, random_state=1)

#define scaler
scaler=MinMaxScaler()

#apply preprocessing to split data with scaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#####pickle
import pickle

outfile = open('scaler.pkl', 'wb')
pickle.dump(scaler,outfile)
outfile.close()

#####KNN and Random Forest

#import machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from numpy import mean, std

#Baseline

#KNeighborsClassifier with five-fold cross-validation
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train,y_train,cv=5)
print(mean(cv), '+/-', std(cv))

#random forest classifier with five-fold cross validation
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train,y_train,cv=5)
print(mean(cv), '+/-', std(cv))

#hyperparameter tuning

from sklearn.model_selection import GridSearchCV

#performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: {} +/- {}'.format(str(classifier.best_score_),str(classifier.cv_results_['std_test_score'][classifier.best_index_])))
    print('Best Parameters: ' + str(classifier.best_params_))
    
#KNeighborsClassifier tuning with five-fold cross-validation
knn = KNeighborsClassifier()
param_grid = {
              'n_neighbors' : [9],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'p' : [1,2]
             }
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = False, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train,y_train)
clf_performance(best_clf_knn,'KNN')

#random forest tuning with five-fold cross-validation
rf = RandomForestClassifier(random_state = 1)
param_grid =  {
                'n_estimators': [93], 
                'bootstrap': [True,False], #bagging (T) vs. pasting (F)
                #'max_depth': np.arange(1,10,2),
                'max_features': ['auto','sqrt'],
                'min_samples_leaf': [1],
                'min_samples_split': [2]
              }
clf_rf_rnd = GridSearchCV(rf, param_grid = param_grid, cv = 5, n_jobs = -1)
best_clf_rf_rnd = clf_rf_rnd.fit(X_train,y_train)
clf_performance(best_clf_rf_rnd,'Random Forest')

#####final model

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(random_state = 1, bootstrap= False, max_features= 'auto', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 93)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

#assess accuracy
print('RandomForestClassifier test accuracy: {}'.format(accuracy_score(y_test, y_pred)))

#####pickle

outfile = open('random_forest_model.pkl', 'wb')
pickle.dump(rf,outfile)
outfile.close()