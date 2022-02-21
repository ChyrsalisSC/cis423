import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

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
    
def halving_search(model, grid, x_train, y_train, factor=3, scoring='roc_auc'):

    halving_cv = HalvingGridSearchCV(
      model, grid,  #our model and the parameter combos we want to try
      scoring = scoring,  #could alternatively choose f1, accuracy or others
      n_jobs=-1,
      min_resources="exhaust",
      factor = factor,  #a typical place to start so triple samples and take top 3rd of combos on each iteration
      cv=5, random_state=1234,
      refit=True  #remembers the best combo and gives us back that model already trained and ready for testing
    )
    grid_result = halving_cv.fit(x_train ,  y_train)
    return grid_result
  
def threshold_results(thresh_list, actuals, predicted):
  result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accuracy'])
  for t in thresh_list:
    yhat = [1 if v >=t else 0 for v in predicted]
    #note: where TP=0, the Precision and Recall both become 0
    precision = precision_score(actuals, yhat, zero_division=0)
    recall = recall_score(actuals, yhat, zero_division=0)
    f1 = f1_score(actuals, yhat)
    accuracy = accuracy_score(actuals, yhat)
    result_df.loc[len(result_df)] = {'threshold':t, 'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy}
  return result_df
  

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


class KNNTransformer(BaseEstimator, TransformerMixin):
  def __init__(self,n_neighbors=5, weights="uniform", add_indicator=False):
    self.n_neighbors = n_neighbors
    self.weights=weights 
    self.add_indicator=add_indicator

  def fit(self, X, y = None):
    print("Warning: KNNTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'KNNTransformer.transform expected Dataframe but got {type(X)} instead.'
    X_ = X.copy()
    cols = X_.columns
    imputer = KNNImputer(n_neighbors =self.n_neighbors, weights = self.weights, add_indicator=False)
    imputed_data = imputer.fit_transform(X_)
    result = pd.DataFrame(imputed_data)
    result.columns = cols 
    return result
    
  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
def find_random_state(df, labels, n=200):
  var = []  #collect test_error/train_error where error based on F1 score

  #2 minutes
  for i in range(1,n):
      train_X, test_X, train_y, test_y = train_test_split(df, labels, test_size=0.2, shuffle=True,
                                                      random_state=i, stratify=labels)
      model.fit(train_X, train_y)  #train model
      train_pred = model.predict(train_X)  #predict against training set
      test_pred = model.predict(test_X)    #predict against test set
      train_error = f1_score(train_y, train_pred)  #how bad did we do with prediction on training data?
      test_error = f1_score(test_y, test_pred)     #how bad did we do with prediction on test data?
      error_ratio = test_error/train_error        #take the ratio
      var.append(error_ratio)

  rs_value = sum(var)/len(var)
  rs_value  #1.0024032354009096

  idx = np.array(abs(var - rs_value)).argmin()  #find the index of the smallest value
  return idx


titanic_transformer = Pipeline(steps=[
    ('drop', DropColumnsTransformer(['Age', 'Gender', 'Class', 'Joined', 'Married',  'Fare'], 'keep')),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', MappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe', OHETransformer(target_column='Joined')),
    ('age', TukeyTransformer(target_column='Age', fence='outer')), #from chapter 4
    ('fare', TukeyTransformer(target_column='Fare', fence='outer')), #from chapter 4
    ('minmax', MinMaxTransformer()),  #from chapter 5
    ('imputer', KNNTransformer())  #from chapter 6
    ], verbose=True)

customer_transformer = Pipeline(steps=[
    ('id', DropColumnsTransformer(column_list=['ID'])),
    ('os', OHETransformer(target_column='OS')),
    ('isp', OHETransformer(target_column='ISP')),
    ('level', MappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('gender', MappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('time spent', TukeyTransformer('Time Spent', 'inner')),
    ('minmax', MinMaxTransformer()),
    ('imputer', KNNTransformer())
    ], verbose=True)


def dataset_setup(feature_table, labels, the_transformer, rs=1234, ts=.2):
    X_train, X_test, y_train, y_test = train_test_split(feature_table, labels, test_size=ts, shuffle=True,
                                                    random_state=rs, stratify=labels)

    X_train_transformed = the_transformer.fit_transform(X_train)
    X_test_transformed = the_transformer.fit_transform(X_test)

    x_trained_numpy = X_train_transformed.to_numpy()
    x_test_numpy = X_test_transformed.to_numpy()
    y_train_numpy = np.array(y_train)
    y_test_numpy = np.array(y_test)
 
    return  x_trained_numpy,  y_train_numpy,  x_test_numpy , y_test_numpy

def titanic_setup(titanic_table, transformer=titanic_transformer, rs=88, ts=.2):
    x_trained_numpy, y_train_numpy, x_test_numpy, y_test_numpy = dataset_setup(titanic_table.drop(columns='Survived'),
                                                                           titanic_table['Survived'].to_list(),
                                                                           titanic_transformer, rs, ts)
    return x_trained_numpy, y_train_numpy, x_test_numpy, y_test_numpy
  
def customer_setup(customer_table, transformer=customer_transformer, rs=107, ts=.2):
    x_trained_numpy, y_train_numpy, x_test_numpy, y_test_numpy = dataset_setup(customer_table.drop(columns='Rating'),
                                                                           customer_table['Rating'].to_list(),
                                                                           customer_transformer,rs, ts)
    return x_trained_numpy, y_train_numpy, x_test_numpy, y_test_numpy
