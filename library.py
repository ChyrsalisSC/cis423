import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class MappingTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, mapping_column, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=True):  
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'RenamingTransformer.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    X_ = pd.get_dummies(X_,
                               prefix=self.target_column,   
                               prefix_sep='_',    
                               columns=[self.target_column],
                               dummy_na = self.dummy_na,   
                               drop_first=  self.drop_first  
                               )
    #assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    #print("works")
    return X_

  def fit(self, X, y = None): 
    print("Warning: OHETransformer.fit does nothing.")
    return X
 
  def fit_transform(self, X, y= None): #was stuck for like 2 hours becuase i had (self,x) not (self,X,y=None)
    result = self.transform(X)
    return result
  
  
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  #fill in rest below
  def transform(self, X):
    Table = X.copy()
    #get columbs for keeping

    if self.action == 'drop':
      Table = Table.drop(columns = self.column_list)
    else:
      Table = Table[self.column_list]
    return Table
  
  def fit(self, X, y = None): 
    print("Warning:DropColumnsTransformer.fit does nothing.")
    return X
    

  def fit_transform(self, X, y= None):
    result = self.transform(X)
    return result

  
class Sigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):  
    self.target_column = target_column


  def fit(self, X, y = None):
    print("Warning: Sigma3Transformer(.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'Sigma3Transformer(.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    m = X_[self.target_column].mean()
    #compute std of column - look for method
    sigma = X_[self.target_column].std()
    minb, maxb =  (m - 3*sigma, m+ 3*sigma)
    X_[self.target_column] = X_[self.target_column].clip(lower=minb, upper=maxb)
    return X_

 
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class TukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']
    self.target_column = target_column
    self.fence = fence

  def fit(self, X, y = None):
    print("Warning: TukeyTransformer(.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'TukeyTransformer(.transform expected Dataframe but got {type(X)} instead.'

    X_ = X.copy()
    #now add on outer fences
    q1 = X_[  self.target_column ].quantile(0.25)
    q3 = X_[  self.target_column ].quantile(0.75)
    iqr = q3-q1
    if (self.fence == 'inner'):
        inner_low = q1-1.5*iqr  #??????? .5 works i guess?
        inner_high = q3+1.5*iqr
        X_[  self.target_column ] =  X_[self.target_column].clip(lower=inner_low, upper=inner_high)
    else:
      
        outer_low = q1-3*iqr
        outer_high = q3+3*iqr
        X_[  self.target_column ] =  X_[self.target_column].clip(lower=outer_low, upper=outer_high)
  
    return X_

 
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
class PearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshold):
    self.threshold = threshold


  def fit(self, X, y = None):
    print("Warning: PearsonTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
   
    X_ = X.copy()

    X_ = X_.corr(method='pearson')
    masked_df = X_.abs() > self.threshold
    upper_mask = np.triu(masked_df , 1).astype(bool)

    columns = np.transpose(upper_mask.nonzero()) 
    correlated_columns = []
   
    [correlated_columns.append(masked_df.columns[item[1]]) for item in columns if masked_df.columns[item[1]] not in correlated_columns]
    

    new_df = masked_df.drop(columns=correlated_columns)

    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class MinMaxTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass
  #fill in rest below
  def transform(self, X):
    Table = X.copy()
    for col in X:
      mi = Table[col].min()
      mx = Table[col].max()
      denom = (mx - mi)
      Table[col] -= mi
      Table[col] /=denom

    return Table
  
  def fit(self, X, y = None): 
    print("Warning:DropColumnsTransformer.fit does nothing.")
    return X
    

  def fit_transform(self, X, y= None):
    result = self.transform(X)
    return result
