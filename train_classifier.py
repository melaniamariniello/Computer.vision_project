import pickle

#these are 3 libraries I use to train the classifier
from sklearn.ensemble import RandomForestClassifier	
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import numpy 
from numpy import float64

#import data from data.pickle and read them as rb -> data_dict = data dictionary
data_dict = pickle.load(open('./data.pickle', 'rb'))

#print(data_dict.keys())
#print(data_dict)

#data = numpy.asarray(data_dict['data'],dtype=object)		#convert to array (the classifier I use works in this way) (data and labels are lists so I need them as np arrays)
#labels = numpy.asarray(data_dict['labels'],dtype=object)	#convert to array

#now I split these data in a training set and a test set (to train and test performance)
#print(labels)

#x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
#from data array -> x_train and x_test
#from labels array -> y_trains and y_test
#the training sets train the model by using cross validation
#test_size is the size of the test set -> 0.2 = 20% of data are used for test
#shuffle=true -> I'm shuffling the data (I always have to do it)
#the objects will have the same proportions (stratify=labels)

#USE RANDOM FOREST CLASSIFIER
#model = RandomForestClassifier()	#I create the model using this classifier (simple algorithm)

#model.fit(x_train, y_train)	#I fit the model using these test data

#y_predict = model.predict(x_test)	#I make the prediction after I have trained the classifier
#score = accuracy_score(y_predict, y_test) #I need to see how the classifier performs


#USE DECISION TREE CLASSIFIER
#from sklearn.tree import DecisionTreeClassifier
#data = numpy.asarray(data_dict['data'])		
#labels = numpy.asarray(data_dict['labels'])
#x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state = 0)
#dtree_model = DecisionTreeClassifier(max_depth = 2).fit(x_train, y_train)
#dtree_predictions = dtree_model.predict(x_test)

#score = accuracy_score(dtree_predictions, y_test) #I need to see how the classifier performs
# training a linear SVM classifier
data = numpy.asarray(data_dict['data'],dtype=float64)		
labels = numpy.asarray(data_dict['labels'], dtype=object)

x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state = 0)

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(x_train, y_train)
svm_predictions = svm_model_linear.predict(x_test)
score= accuracy_score(svm_predictions, y_test) #I need to see how the classifier performs
model = svm_model_linear

print('{}% of samples were classified correctly !'.format(score * 100)) #based on the % I see the perfomance of the classifier -> 100% = perfect classifier

#GET REPORT
print((classification_report(y_test,svm_predictions, zero_division=1)))

#print((classification_report(y_test_2,svm_predictions_2, zero_division=1)))

#Now I save the model
f = open('model.p', 'wb') #model.p is the name of the model (I will have a model.p file which is the saved model)
pickle.dump({'model': model}, f)
f.close()