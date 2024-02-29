# functions to used for data processing and runnign scripts
import imblearn
import pandas as pd

from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

def data_resampling(scaling, X_train, y_train):
    
    """Data resampling:
    Parameters:
        - scaling type: 'up' or 'down' if you want to upscale or downscale data
        - X_training atributes set and y training target values
    Returns:
        - upscaled or downscaled training set 
    """
    
    train = pd.concat([X_train, y_train], axis=1)
    
    no_0 = train[train['Bankrupt'] == 0]
    yes_1 = train[train['Bankrupt'] == 1]
    
    if scaling == 'down':
        no_0_dw = resample(no_0, replace=False, n_samples=len(yes_1), random_state=0)          
        train_resampled = pd.concat([yes_1, no_0_dw], axis=0)
        
    elif scaling == 'up':       
        yes_1_up = resample(yes_1, replace=True, n_samples = len(no_0), random_state=0)       
        train_resampled = pd.concat([no_0, yes_1_up], axis=0)
        
    else:
        print("Something went wrong. Use 'up' or 'down' as parameters to chose type of data sampling/scaling?")
    
    
    X_resampled = train_resampled.drop('Bankrupt', axis=1).copy()
    y_resampled = train_resampled['Bankrupt'].copy()
    
    return X_resampled, y_resampled


def kbest_features_selection(X, y, scaler = MinMaxScaler(), func=f_classif, k_atr=30):
    """Feature selection usin Kbest method:
    Parameters:
        - X - atributes (not scaled, original data) and y - target values
        - scaler for atributes 
        - func - scoring function used to evaluate the importance of each feature
        - k - number of top features to select
    
    Returns:
        - reduced set of atributes (not scaled columns ! original values)
        """
    
    scaler = scaler
    
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    kbest = SelectKBest(func, k=k_atr)
    
    X_scaled_kbest = kbest.fit_transform(X_scaled, y)
    
    scores = kbest.scores_
    columns = kbest.feature_names_in_
    
    df_score_columns = pd.DataFrame({'scores' : scores, 'columns' : columns}).sort_values(by='scores', ascending = False).head(k_atr)
    
    X_reduced = X[list(df_score_columns['columns'])]
        
    return X_reduced



def remove_collinearity(X_set, threshold=0.95):
    
    correlation_matrix = X_set.corr()    
    
    col_i = []
    col_j = []

    for j in range(len(correlation_matrix.columns)): 
        for i in range(j):
            if (abs(correlation_matrix.iloc[j,i]) > threshold) and (correlation_matrix.columns[j] != correlation_matrix.columns[i]):
                col_j.append(correlation_matrix.columns[j])
                col_i.append(correlation_matrix.columns[i])


    cols_to_remove = list(set(col_j))

    X_set_reduced = X_set.drop(cols_to_remove, axis=1).copy()
    
    return X_set_reduced


def split_scale(X, y, scaler = MinMaxScaler(), test_size=0.2, random_state=None):
    """Function that splits data into train and test sets and scales the features.
    
    Required Parameters:
        - X - data frame with features
        - y - target variable
    Optiona Parameters:
        - scaler - how to scale features, by default MinMaxScaler() is used
        - test_size - by default 20% of data
        - random_state - by default None
        
    Returns: training and test data sets
        X_train, X_test, y_train, y_test    
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = scaler
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test



def model_score(y_test, y_pred):
    
    scores = {
        'precission' : ['{:.3f}'.format(precision_score(y_test, y_pred))],
        'accuracy' : ['{:.3f}'.format(accuracy_score(y_test, y_pred))],
        'recall' : ['{:.3f}'.format(recall_score(y_test, y_pred))],
        'f1_score' : ['{:.3f}'.format(f1_score(y_test, y_pred))]
    }
    
    scores_summary = pd.DataFrame(scores)
        
    return scores_summary