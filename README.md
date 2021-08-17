# Project 4: Credit Card Users Cluster Analysis (Python)

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

## Results

### EDA

I looked at the distributions of the data and the correlations between variables. Below are some of the highlights:

<div align="center">
  
<figure>
<img src=""><br/>
  <figcaption></figcaption>
</figure>
<br/><br/>
  
</div>

### Feature Engineering

### Model Building

### Model Performance

## Productionization

## Resources

1. [Kaggle: Credit Card Dataset](https://www.kaggle.com/arjunbhasin2013/ccdata)
