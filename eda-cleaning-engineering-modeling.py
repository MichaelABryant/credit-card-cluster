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

#####feature engineering

cc_data['AVG_PURCHASE_TRX_PRICE'] = cc_data.loc[:,'PURCHASES']/cc_data.loc[:,'PURCHASES_TRX']

cc_data['AVG_PURCHASE_TRX_PRICE'] = cc_data.AVG_PURCHASE_TRX_PRICE.replace(np.NaN, 0)
cc_data['AVG_PURCHASE_TRX_PRICE'] = cc_data.AVG_PURCHASE_TRX_PRICE.replace(np.inf, 0)

cc_data['BALANCE_TO_CREDIT_LIMIT'] = cc_data.loc[:,'BALANCE']/cc_data.loc[:,'CREDIT_LIMIT']

cc_data['AVG_CASH_ADVANCE_TRX_AMOUNT'] = cc_data.loc[:,'CASH_ADVANCE']/cc_data.loc[:,'CASH_ADVANCE_TRX']

cc_data['AVG_CASH_ADVANCE_TRX_AMOUNT'] = cc_data.AVG_CASH_ADVANCE_TRX_AMOUNT.replace(np.NaN, 0)
cc_data['AVG_CASH_ADVANCE_TRX_AMOUNT'] = cc_data.AVG_CASH_ADVANCE_TRX_AMOUNT.replace(np.inf, 0)

#####cluster analysis

##purchases

#standardize for kmeans
from sklearn.preprocessing import StandardScaler

kmeans_columns1 = ['AVG_PURCHASE_TRX_PRICE', 'ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']
cc_data_kmeans1 = cc_data.loc[:, kmeans_columns1]

standardize = StandardScaler()
cc_data_kmeans1 = standardize.fit_transform(cc_data_kmeans1)

#silhouette and elbow method

from sklearn.cluster import KMeans

kmeans_models = [KMeans(n_clusters=k, random_state=1).fit(cc_data_kmeans1) for k in range (1, 10)]
innertia = [model.inertia_ for model in kmeans_models]

plt.plot(range(1, 10), innertia)
plt.title('Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#generate clusters to determine different clusters to advertise to

kmeans = KMeans(n_clusters=4, random_state = 1)
cc_data["Ad_Groups_Purchases"] = kmeans.fit_predict(cc_data_kmeans1)

cc_data["Ad_Groups_Purchases"] = cc_data["Ad_Groups_Purchases"].astype("category")

#plot variables against clusters

for i in ['BALANCE', 'BALANCE_FREQUENCY', 'BALANCE_TO_CREDIT_LIMIT','AVG_PURCHASE_TRX_PRICE', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']:
    sns.stripplot(x ='Ad_Groups_Purchases',y=i,data=cc_data)
    plt.show()
    

##cash advance

#standardize for kmeans
kmeans_columns2 = ['BALANCE', 'AVG_CASH_ADVANCE_TRX_AMOUNT']
cc_data_kmeans2 = cc_data.loc[:, kmeans_columns2]

standardize = StandardScaler()
cc_data_kmeans2 = standardize.fit_transform(cc_data_kmeans2)

#silhouette and elbow method
kmeans_models = [KMeans(n_clusters=k, random_state=1).fit(cc_data_kmeans2) for k in range (1, 10)]
innertia = [model.inertia_ for model in kmeans_models]

plt.plot(range(1, 10), innertia)
plt.title('Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#generate clusters to determine different clusters to advertise to
kmeans = KMeans(n_clusters=4, random_state = 1)
cc_data["Ad_Groups_Cash_Advance"] = kmeans.fit_predict(cc_data_kmeans2)

cc_data["Ad_Groups_Cash_Advance"] = cc_data["Ad_Groups_Cash_Advance"].astype("category")


#plot variables against clusters
for i in ['BALANCE', 'BALANCE_FREQUENCY', 'BALANCE_TO_CREDIT_LIMIT', 'AVG_PURCHASE_TRX_PRICE', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'AVG_CASH_ADVANCE_TRX_AMOUNT','CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']:
    sns.stripplot(x ='Ad_Groups_Cash_Advance',y=i,data=cc_data)
    plt.show()


#####preparing data for classification

#create X and y variables
X1 = cc_data.loc[:,kmeans_columns1]
y1 = cc_data['Ad_Groups_Purchases']

X2 = cc_data.loc[:,kmeans_columns2]
y2 = cc_data['Ad_Groups_Cash_Advance']

#import libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


##puchases
#create dummy variables for y
y1 = pd.get_dummies(y1)

# train/test split with stratify making sure classes are evenlly represented across splits
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, stratify=y1, train_size=0.75, random_state=1)

#define scaler
scaler_purchases=MinMaxScaler()

#apply preprocessing to split data with scaler
X_train1 = scaler_purchases.fit_transform(X_train1)
X_test1 = scaler_purchases.transform(X_test1)

#####pickle
import pickle

outfile = open('scaler_purchases.pkl', 'wb')
pickle.dump(scaler_purchases,outfile)
outfile.close()



##cash
#create dummy variables for y
y2 = pd.get_dummies(y2)

# train/test split with stratify making sure classes are evenlly represented across splits
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, stratify=y2, train_size=0.75, random_state=1)

#define scaler
scaler_cash=MinMaxScaler()

#apply preprocessing to split data with scaler
X_train2 = scaler_cash.fit_transform(X_train2)
X_test2 = scaler_cash.transform(X_test2)



#####pickle
import pickle

outfile = open('scaler_cash.pkl', 'wb')
pickle.dump(scaler_cash,outfile)
outfile.close()


#####purchases clusters ML


#import machine learning libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from numpy import mean, std

#Baseline

#KNeighborsClassifier with five-fold cross-validation
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train1,y_train1,cv=5)
print(mean(cv), '+/-', std(cv))

#random forest classifier with five-fold cross validation
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train1,y_train1,cv=5)
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
              'n_neighbors' : np.arange(5,12,1),
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'p' : [1,2]
             }
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = False, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train1,y_train1)
clf_performance(best_clf_knn,'KNN')

#RandomForest tuning with five-fold cross-validation
rf = RandomForestClassifier(random_state = 1)
param_grid =  {
                'n_estimators': np.arange(5,10,1), 
                'bootstrap': [True,False], #bagging (T) vs. pasting (F)
                #'max_depth': [1],
                'max_features': ['auto','sqrt'],
                'min_samples_leaf': np.arange(1,5,1),
                'min_samples_split': np.arange(1,5,1)
              }
clf_rf_rnd = GridSearchCV(rf, param_grid = param_grid, cv = 5, n_jobs = -1)
best_clf_rf_rnd = clf_rf_rnd.fit(X_train1,y_train1)
clf_performance(best_clf_rf_rnd,'Random Forest')

#####final model

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(random_state = 1, bootstrap= False, max_features= 'auto', min_samples_leaf= 3, min_samples_split= 2, n_estimators= 7)
rf.fit(X_train1,y_train1)
y_pred1 = rf.predict(X_test1)

#assess accuracy
print('RandomForestClassifier test accuracy: {}'.format(accuracy_score(y_test1, y_pred1)))

#####pickle

outfile = open('random_forest_model.pkl', 'wb')
pickle.dump(rf,outfile)
outfile.close()


#####cash clusters ML

##baseline

#KNeighborsClassifier with five-fold cross-validation
knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train2,y_train2,cv=5)
print(mean(cv), '+/-', std(cv))

#random forest classifier with five-fold cross validation
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train2,y_train2,cv=5)
print(mean(cv), '+/-', std(cv))

##hyperparameter tuning

#KNeighborsClassifier tuning with five-fold cross-validation
knn = KNeighborsClassifier()
param_grid = {
              'n_neighbors' : np.arange(5,12,1),
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'p' : [1,2]
             }
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = False, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train2,y_train2)
clf_performance(best_clf_knn,'KNN')

#RandomForest tuning with five-fold cross-validation
rf = RandomForestClassifier(random_state = 1)
param_grid =  {
                'n_estimators': np.arange(9,15,1), 
                'bootstrap': [True,False], #bagging (T) vs. pasting (F)
                #'max_depth': [1],
                'max_features': ['auto','sqrt'],
                'min_samples_leaf': np.arange(1,5,1),
                'min_samples_split': np.arange(4,10,1)
              }
clf_rf_rnd = GridSearchCV(rf, param_grid = param_grid, cv = 5, n_jobs = -1)
best_clf_rf_rnd = clf_rf_rnd.fit(X_train2,y_train2)
clf_performance(best_clf_rf_rnd,'Random Forest')

##final model

knn = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 5, p= 1, weights= 'distance')
knn.fit(X_train2,y_train2)
y_pred2 = knn.predict(X_test2)

#assess accuracy
print('KNN test accuracy: {}'.format(accuracy_score(y_test2, y_pred2)))

outfile = open('knn_model.pkl', 'wb')
pickle.dump(knn,outfile)
outfile.close()
