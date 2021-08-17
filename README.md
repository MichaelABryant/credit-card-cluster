# Project 4: Credit Card Users Cluster Analysis (Python/HTML/Heroku)

This repository is for the analysis, clustering, and modeling done with a credit card history dataset. Below you will find an overview of the data, code, and results. The goal was to create a project where I perform an exploratory data analysis (EDA) including a principal component analysis (PCA), cluster analaysis, feature engineering, apply machine learning algorithms to predict clusters based on credit card usage, and create a [deployed application with a front end](https://ad-advisor.herokuapp.com/) to productionize the best performing model. The repo for the app can be found [here](https://github.com/MichaelBryantDS/credit-card-cluster-app).

### Code Used 

**Python Version:** 3.7.10 <br />
**Packages:** pandas, numpy, scipy, sklearn, matplotlib, seaborn, flask, shap, eli5, pickle<br />
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  

## Heart Disease Dataset

The dataset was gathered from [Kaggle](https://www.kaggle.com/arjunbhasin2013/ccdata). The dataset contains 18 variables and 8950 patient records.

### Variables

`CUST_ID`, `BALANCE`, `BALANCE_FREQUENCY`, `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `CASH_ADVANCE`, `PURCHASES_FREQUENCY`, `ONEOFF_PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX`, `PURCHASES_TRX`, `CREDIT_LIMIT`, `PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT`, `TENURE`

## Files

### eda-cleaning-engineering-modeling.py

This file contains the EDA, data cleaning, feature engineering, clustering, and modeling. The EDA is performed using descriptive statistics, histograms to determine distributions, and a correlation heatmap using the Pearson correlation coefficient. Features are engineered based on the principal component analysis (PCA) results. Other feature engineering includes the creation of clusters and numerical features are scaled using MinMaxScaler. Two clusters are created based on the prinipal components and models are created to predict classification. The scalers and models are pickled after fitting for use with productionization.

## Results

### EDA

I looked at the distributions of the data and the correlations between variables. Below are some of the highlights:

<div align="center">
  
<figure>
<img src="images/corr-heatmap.jpg"><br/>
  <figcaption></figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/pca-explained-variance.jpg"><br/>
  <figcaption></figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/pca-weights.jpg"><br/>
  <figcaption></figcaption>
</figure>
<br/><br/>
  
</div>

<div align="center">
  
<figure>
<img src="images/pca-results.jpg"><br/>
  <figcaption></figcaption>
</figure>
<br/><br/>
  
</div>

### Data Cleaning

### Feature Engineering

I feature engineered using the dataset for modeling. I made the following changes:

* Created variables `AVG_PURCHASE_TRX_PRICE`, `BALANCE_TO_CREDIT_LIMIT`, and `AVG_CASH_ADVANCE_TRX_AMOUNT`
* Created two sets of four clusters:
  * The first set is based on `AVG_PURCHASE_TRX_PRICE`, `ONEOFF_PURCHASES`, and `INSTALLMENTS_PURCHASES`
  * The second set is based on `BALANCE` and `AVG_CASH_ADVANCE_TRX_AMOUNT`

### Model Building

### Model Performance

## Productionization

## Resources

1. [Kaggle: Credit Card Dataset](https://www.kaggle.com/arjunbhasin2013/ccdata)
