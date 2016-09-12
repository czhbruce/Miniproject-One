import simplejson as json
import ujson
import scipy
import dill
import re
import sklearn as sk
import pandas as pd
import numpy as np
from sklearn import neighbors,datasets,cross_validation,grid_search,linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
#import matplotlib.pylab as plt
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

class ColumnSelectTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    Returns the first k columns of a feature array
    """
    def __init__(self, colname1, colname2):
        self.long = colname1
        self.lat = colname2
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.loc[:,(self.long,self.lat)]


class Estimator(sk.base.BaseEstimator, sk.base.RegressorMixin):
    """
    A shell estimator that combines a transformer and regressor into a single object.
    """
    def __init__(self, transformer, model):
        if transformer != 'NoTrans':
            self.transformer = transformer
        self.model = model
        pass

    def fit(self, X, y):
        X_trans = self.transformer.fit(X, y).transform(X)
        self.model.fit(X_trans, y)
        return self
    
    def score(self, X, y):
        X_test = self.transformer.transform(X)
        return self.model.score(X_test, y)

    def predict(self, X):
        X_test = self.transformer.transform(X)
        return self.model.predict(X_test)

def compute_error(clf, X, y):
    cv = cross_validation.ShuffleSplit(len(y),n_iter=20,test_size=0.2,random_state=42)
    return -cross_validation.cross_val_score(clf, X, y, cv=cv, scoring='mean_squared_error').mean()
"""
class Modssel_city():
    def __init__(self, cityname):
        self.cityname = cityname
        dir = './df.csv'    
        df = pd.DataFrame()
        df = pd.read_csv(dir)
        df_by_city = df.groupby('city')['stars'].mean()
"""

def main():
    """
    ##################read file##################
    fname = './yelp_train_academic_dataset_business.json'
    with open(fname) as f:
        data = ujson.loads(ujson.dumps(f.readlines()))
    f.close()
    dir = './df.csv'
    
    
    #########Convert json into dataframe and then output to file#########
    df = pd.DataFrame()
    df_i = []
    count = 0
    for line in data:
        row = ujson.loads(line)
        df1 = pd.DataFrame([(row.values())], columns=row.keys())
        df_i.append(df1)
        #df = df.append(df1,ignore_index=True)
        count += 1
        print count/37938.0
    df = pd.concat(df_i, axis=0,ignore_index=True)
    """
    """
    #########Better way of converting data##########
    dir = './df_clean.csv'        
    with open('yelp_train_academic_dataset_business.json', 'rb') as f:
        data = f.readlines()
    #print data[:20]
    data = map(lambda x: x.rstrip(), data)
    data_json_str = '[' + ','.join(data) + ']'
    df_clean = pd.read_json(data_json_str)
    df_clean.to_csv(dir)
    """
    
    
    #########read dataframe from well-formatted csv##########
    dir = './df.csv'    
    df = pd.DataFrame()
    df = pd.read_csv(dir)
    """
    ###########Q1-City Model##############
    city_unique = df['city'].unique()
    df_by_city = df.groupby('city')['stars'].mean()
    def Model_city(cityname):
        return df_by_city[cityname]
    dill.dump(Model_city, open("./ANS/Model_city.dill", 'w')) 
    """
    ############Q2-lat_long_model###############37938
    selector = ColumnSelectTransformer('longitude', 'latitude')
    x = selector.transform(df)
    y = np.asarray(df['stars'], dtype="float")
    """
    cv = cross_validation.ShuffleSplit(len(y),n_iter=20,test_size=0.1,random_state=34)
    param_grid = {"n_neighbors": range(26,34,2)}
    nearest_neighbors_cv = grid_search.GridSearchCV(neighbors.KNeighborsClassifier(),
                                    param_grid=param_grid, cv=cv, scoring = 'accuracy')
    nearest_neighbors_cv.fit(x, y)
    cv_accuracy = pd.DataFrame.from_records(
        [(score.parameters['n_neighbors'],
          score.mean_validation_score)
         for score in nearest_neighbors_cv.grid_scores_],columns=['n_neighbors','accuracy'])
    plt.plot(cv_accuracy.n_neighbors, cv_accuracy.accuracy)
    plt.xlabel('n_neighbors')
    plt.ylabel('accuracy')
    plt.show() #The optimized # of neighbors is 32
    """
    neighbor = sk.neighbors.KNeighborsRegressor(n_neighbors=32)
    neighbor.fit(x,y)
    #print neighbor.predict(x)
    dill.dump(neighbor, open("./ANS/Model_lat_long.dill", 'w')) 
    ###########Q3-Category Model##############
    vectorizer = TfidfVectorizer(min_df=1)
    x_cate = df['categories']
    x_cate_list = [','.join(re.findall(r"u'([^']+)",row)) for row in x_cate]
    tfidf = vectorizer.fit_transform(x_cate_list)
    x_cate_arr = tfidf.toarray()
    #categories = re.findall(r"u'([^']+)",l) #get rid of u'
    #print x_cat_arr
    clf = linear_model.LinearRegression()
    MSE = compute_error(clf, tfidf, y)
    print MSE
    """
    # Lasso Regression with cross validation
    np.random.seed(42)
    alphas = np.logspace(-8., -1., 20)
    print alphas
    lasso_models = pd.DataFrame(
    [(alpha,
      "Lasso with alpha = %f" % alpha,
      compute_error(linear_model.LassoLars(alpha=alpha), x_cate_arr, y)) for alpha in alphas],
    columns=['alpha', 'Model', 'MSE'])
    #lasso_models.plot(x='alpha', y='MSE', logx=True, title='MSE')
    print lasso_models
    """
    
    
    
        
    
    #columns = ['city','review_count','name','neighborhoods','tye','business_id', 'full_address',
        #'hours','state','longitude','stars','latitude','attributes','open','categories']
    #df1_t = df1.transpose()
    #print list(df1_t.columns.values)
    #df.joined = test1
    #get column names### print list(df1.columns.values)
    #for restaurant in data:
    #    res = restaurant.strip()
    ############
    
if __name__ == "__main__":
    main()