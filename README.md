# Bankruptcy Prediction witn Machine Learning 


## Objective
The project aims to analyze data from the Taiwan Economic Journal spanning the years 1999 to 2009. The dataset provides insights into the performance and stability of companies, with a specific focus on bankruptcy classification based on the regulations of the Taiwan Stock Exchange. The primary goal is to develop predictive models for effectively classifying companies as bankrupt or not.

## Data Processing
The original dataset contained 6819 rows and 96 columns, with 220 companies classified as bankrupt and 6599 as non-bankrupt. Data cleaning techniques were applied, revealing that the dataset was already in good condition from the beginning. While maintaining the same number of rows, we removed some columns, reducing their count to 93.


### Feature Reduction
The dataset was simplified using the Kbest feature selection method. Additionally, steps were taken to eliminate collinearity between features, ensuring the robustness of the model. This resulted in a streamlined dataset with 18 remaining features. 


### Addressing Data Imbalance
Given the uneven distribution of bankrupt and non-bankrupt companies, the data was balanced through upscaling techniques to enhance the models' ability to accurately classify both classes. 

All the applied methods for feature reduction and data scaling can be found in the file [functions.py](https://github.com/Beata2307/Bankruptcy_Prediction/blob/main/functions_PR_7.py).

## Models Used
In the project three different classification models were used:

1. Logistic Regression
2. XGBoost
3. HistGradientBoostingClassifier

### Authors
Beata & Karina
