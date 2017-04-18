#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    

#Explortoray Data Analysis
import pandas as pd
df=pd.DataFrame.from_dict(data_dict, orient='index')

df.head()
df.info()

financial_features=['salary', 'deferral_payments', 'total_payments', 'loan_advances', 
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 

email_features= ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi']
POI_label= ['poi']
email_numeric_features=['to_messages', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi']

# Correct dataframe data types 
df[financial_features]=df[financial_features].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df[email_numeric_features]=df[email_numeric_features].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df=df.fillna(value=0)
df.info()

#Plot distribution 
import matplotlib.pyplot as plt

for variable in financial_features:
    df[[variable,'poi']].boxplot(by='poi',figsize=(5,8))

  
for variable in email_numeric_features:
    df[[variable,'poi']].boxplot(by='poi',figsize=(5,8))


#SelectKBest
cols = [col for col in df.columns if col not in['poi','email_address']]
X = df[cols]
y=df['poi']

from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k='all').fit(X,y)
x_new=selector.transform(X)
scores=selector.scores_
order=scores.argsort()
rank=order.argsort().tolist()

ordered_features=[]
for i in range(18,-1,-1):
    ordered_features.append(cols[rank.index(i)])
print ordered_features
    

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_stock_value','bonus','salary','exercised_stock_options',
'total_payments','fraction_from_poi','fraction_to_poi','shared_receipt_with_poi']


### Task 2: Remove outliers
data_dict.pop('TOTAL')

### Task 3: Create new feature(s)
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if poi_messages!="NaN" and all_messages!= "NaN":
        fraction=poi_messages/(1.0*all_messages)

    return fraction
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

#SVM classifier
#from sklearn.svm import SVC
#estimators = [('reduce_dim', PCA()), ('svm', SVC())]
#pipe = Pipeline(estimators)
#params = dict(reduce_dim__n_components=[2,3, 5, 7], svm__C=[0.1, 1, 10, 100])
#clf = GridSearchCV(pipe, param_grid=params)

#DecisionTree classifier
from sklearn import tree
estimators=[('reduce_dim', PCA()), ("scale", MinMaxScaler(feature_range=(0, 1))),('tree', tree.DecisionTreeClassifier())]
pipe = Pipeline(estimators)
params = dict(reduce_dim__n_components=[2,3,4,5,6], tree__min_samples_split=[2,4,5,6,7])
clf = GridSearchCV(pipe, param_grid=params)

#KMeans 
#from sklearn.cluster import KMeans
#estimators=[('reduce_dim', PCA()), ("scale", MinMaxScaler(feature_range=(0, 1))),('kmeans', KMeans(n_clusters=2))]
#pipe = Pipeline(estimators)
#params = dict(reduce_dim__n_components=[2,3,4,5,6], kmeans__max_iter=[300,400,500],kmeans__n_init=[10,20,30])
#clf = GridSearchCV(pipe, param_grid=params)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
    

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf.fit(features_train, labels_train)
best_clf=clf.best_estimator_
print best_clf
pred=clf.predict(features_test)
from sklearn.metrics import accuracy_score
print "accuracy is ", accuracy_score(labels_test, pred)
from sklearn.metrics import precision_score
print "precision is ", precision_score(labels_test, pred)
from sklearn.metrics import recall_score 
print "recall is ", recall_score(labels_test, pred)
from sklearn.metrics import f1_score
print "F1 score is ", f1_score(labels_test, pred)

# Test new features vs old features
features_list_test= ['poi','total_stock_value','bonus','salary','exercised_stock_options',
'total_payments','from_messages', 'from_this_person_to_poi','shared_receipt_with_poi']
my_dataset_test = data_dict
data = featureFormat(my_dataset_test, features_list_test, sort_keys = True)
labels, features = targetFeatureSplit(data)

clf.fit(features_train, labels_train)
pred=clf.predict(features_test)
print "accuracy is ", accuracy_score(labels_test, pred)
print "precision is ", precision_score(labels_test, pred)
print "recall is ", recall_score(labels_test, pred)
print "F1 score is ", f1_score(labels_test, pred)



dump_classifier_and_data(best_clf, my_dataset, features_list)
