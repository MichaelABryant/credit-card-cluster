# Project 4: Credit Card Users Cluster Analysis (Python)

This repository is for the analysis, clustering, and modeling done with a credit card history dataset. Below you will find an overview of the data, code, and results. The goal was to create a project where I perform an exploratory data analysis (EDA) including a principal component analysis (PCA), cluster analaysis, feature engineer, and apply machine learning algorithms to predict clusters based on credit card usage.

### Code Used 

**Python Version:** 3.7.10 <br />
**Packages:** pandas, numpy, scipy, sklearn, matplotlib, seaborn, flask, statsmodels, shap, eli5, pickle<br />
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  

## Heart Disease Dataset

The dataset was gathered from [Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci). The dataset contains 14 variables and 303 patient records.

### Variables

`age`, `sex`,`cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

## Files

### eda-engineering.py

This file contains the EDA and feature engineering. The EDA is performed using descriptive statistics, histograms to determine distributions, and a correlation heatmap using the Pearson correlation coefficient. A feature is engineered by creating a predictor based on risk factors. Other feature engineering includes the creation of dummy variables for categorical variables with pandas get_dummies and numerical features are scaled using MinMaxScaler. The scaler is pickled after fitting for use with productionization.

### modeling.py

This file contains the modeling where I hyperparameter tune: GaussianNB, LogisticRegression, DecisionTreeClassifier, kNeighborsClassifier, RandomForestClassifier, SVC (support vector classifier), XGBClassifier, StackingClassifier, (hard) VotingClassifier, (soft) VotingClassifier, BaggingClassifier, (pasting with) BaggingClassifier, and AdaBoostClassifier. Since the computational needs are low from having 303 records with 12 features, I used 13 ML algorithms and ensemble methods. The models are hyperparameter tuned with GridSearchCV based on accuracy and the best models are judged based on accuracy, sensitivity, specificity, precision, and AUC metrics. This file also contains code to derive the feature importance from the best models using shap and eli5 package.

### final-model.py

This file contains the best model (SVC) and it is pickled for use with productionization.

## Results

### EDA

I looked at the distributions of the data and the correlations between variables. Below are some of the highlights:

<div align="center">
  
<figure>
<img src="images/cor-heatmap.jpg"><br/>
  <figcaption>Figure 1: Correlation heatmap for numerical variables using Pearson correlation coefficient</figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/thalach-age-target.jpg"><br/>
  <figcaption>Figure 2: Scatter plot with linear regression lines showing maximum heart rate decreases at a greater rate with age for those with heart disease.</figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/oldpeak-target.jpg"><br/>
  <figcaption>Figure 3: Violin plot showing lower oldpeak values for those with heart disease.</figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/risk-factors-target.jpg"><br/>
  <figcaption>Figure 4: Bar plot showing minimal relationship between number of risk factors and heart disease.</figcaption>
</figure>
<br/><br/>
  
</div>

### Feature Engineering

I feature engineered using the dataset for future modeling. I made the following changes:
* Created dummy variables for `sex`,`cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`

### Model Building

First, I split the data into train and tests sets with a test set size of 25%.

I then hyperparameter tuned 13 different models with five-fold cross-validation and evaluated them using accuracy.

The models I used were GaussianNB, LogisticRegression, DecisionTreeClassifier, kNeighborsClassifier, RandomForestClassifier, SVC (support vector classifier), XGBClassifier, StackingClassifier, (hard) VotingClassifier, (soft) VotingClassifier, BaggingClassifier, (pasting with) BaggingClassifier, and AdaBoostClassifier.

### Model Performance

For this application it's important to minimize false negatives (i.e., people who have heart disease but were predicted not to). I looked at accuracy, precision, specificity, sensitivity, AUC, and MCC to choose the best model. For these reasons, the best model was:
<br/><br/>
**Hard VotingClassifier**
* Accuracy: 0.8553
* Sensitivity: 0.8717
* Specificity: 0.8367
* Precision: 0.8571
* MCC: 0.7084

The hard voting classifier included GaussianNB, LogisticRegression, and SVC.

### Feature Importance

According to the permutation importance for the hard voting classifier in Figure 5, the most important features, in order, were `slope`, `thalach`, `oldpeak`, `ca`, and `cp`. `fbs` and `chol` had no impact on the model which is against conventional wisdom that diabetes and high cholesterol increase the risk for heart disease. Although, this group of people are not representative of the general population, because the common trait of these patients is that they have all experienced angina. `thalach` ranks highly in feature importance, as shown in Figure 2, there is a definite difference between the heart disease groups. Interestingly, maximum heart rate converges for the two groups, so `thalach` is probably more useful to determine heart disease in younger patients than older. `slope` is the most important feature to the hard voting classifier which is determined from the ST segment of the electrocardiogram. Looking at Figure 6, the SVC model does not prioritize this feature as much so it must be heavily weighted from either naive Bayes or logistic regression.

<div align="center">
  
<figure>
<img src="images/perm-import-hv.JPG"><br/>
  <figcaption>Figure 5: Permutation importance for the hard voting classifier.</figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/shap-svc.jpg"><br/>
  <figcaption>Figure 6: SHAP summary plot of the feature importance for the SVC model.</figcaption>
</figure>
<br/><br/>
  
</div>

The decision tree graphic, shown in Figure 7, is included because it is easy to derive insights from this model and it scored similarly in accuracy to the best models. `thal` appears to be one of the more important features as it also appears in the permutation importance for the hard voting classifier. Having a reversible defect, as determined by the thalium stress test, may signify heart disease whereas having normal results (i.e., the best outcome) is a good indicator that you don't have heart disease. A thalium stress test indicated a fixed defect is the most heavily weighted feature for the hard voting classifier to classify records as having no heart disease.

<div align="center">

<figure>
<img src="images/dt.jpg"><br/>
  <figcaption>Figure 7: Graphical representation of a decision tree.</figcaption>
</figure>
<br/><br/>
  
</div>

## Productionization

I built a [Heroku web app](https://predict-heart-diseases.herokuapp.com/) with a front end which takes patient medical information input from the user and outputs a heart disease prediction using the hard voting classifier model. More information on this can be found at it's [repo](https://github.com/MichaelBryantDS/heart-disease-pred-app).

<div align="center">
  
<figure>
<img src="images/front-end.JPG"><br/>
  <figcaption>Figure 8: Front end of application using the hard voting classifier model.</figcaption>
</figure>
<br/><br/>
  
</div>

## Resources

1. [Kaggle: Credit Card Dataset](https://www.kaggle.com/arjunbhasin2013/ccdata)
