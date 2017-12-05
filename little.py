# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree,metrics
import numpy as np 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.externals import joblib
import time
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


info = open('../Info','a')

# Load dataset
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

mytestname = ['Lat','Lng','speed','signal','cellid','pci','tac','mnc','Real_label']
#mydatasets = pandas.read_csv('BatteryAfterParse_0214_0212',sep=',',names=mytestname)
myvalidname = ['date','Lat','Lng','speed','signal','cellid','pci','tac','mnc','Real_label']

#mydatasets = pandas.read_csv('../BatteryAfterParse_0214_0212',sep=',',names=mytestname)

#mydatasets = pandas.read_csv('../BatteryAfterParse_0906_total',sep=',',names=mytestname)
mydatasets = pandas.read_csv('../total_train',sep=',',names=mytestname)
mydatasets = pandas.read_csv('../train_hsr',sep=',',names=mytestname)
mydatasets = pandas.read_csv('../train_hsr_mnc',sep=',',names=mytestname)
#mydatasets = pandas.read_csv('../total_mnc',sep=',',names=mytestname)
mydatasets = pandas.read_csv('../train_hsr_more',sep=',',names=mytestname)
mydatasets = pandas.read_csv('../train_hsr_more_clear',sep=',',names=mytestname)

TRAIN = 'train_hsr_more_clear'
TRAIN = 'train_hsr_more_clear'
TRAIN = 'train_sub_more_clear'
TRAIN = 'train_tra_more_clear'
TRAIN = 'train_hw_more_clear'
#TRAIN = 'total_train_mnc_clear'
TRAIN = 'total_train_gps'
mydatasets = pandas.read_csv('../'+TRAIN,sep=',',names=mytestname)

#mydatasets = pandas.read_csv('../total_train',sep=',',names=mytestname)
#myvalid = pandas.read_csv('../BatteryAfterParse_119_0004',sep=',',names=myvalidname)
#myvalid = pandas.read_csv('../BatteryAfterParse_jcc',sep=',',names=myvalidname)

#myvalid = pandas.read_csv('../BatteryAfterParse_0906_test',sep=',',names=myvalidname)

#myvalid = pandas.read_csv('../highway_test',sep=',',names=myvalidname)
#myvalid = pandas.read_csv('../20171115',sep=',',names=myvalidname)
# myvalid = pandas.read_csv('../BatteryAfterParse_119_0005',sep=',',names=myvalidname)
myvalid = pandas.read_csv('../BatteryAfterParse_119_0005_mnc',sep=',',names=myvalidname)
#myvalid = pandas.read_csv('../BatteryAfterParse_118_0004',sep=',',names=myvalidname)
#myvalid = pandas.read_csv('../jcc_111529',sep=',',names=myvalidname)
#myvalid = pandas.read_csv('../total_test',sep=',',names=myvalidname)

TEST = 'BatteryAfterParse_119_0005_mnc'
TEST = 'total_test_new'
#TEST = 'total_test_cell'
#TEST = '1115_165'
#TEST = 'BatteryAfterParse_118_0004'
TEST = 'BatteryAfterParse'
myvalid = pandas.read_csv('../' + TEST,sep=',',names=myvalidname)

info.write('train:\t'+TRAIN+'\n')
info.write('test:\t'+TEST+'\n')

# mydatasets = mydatasets.drop('signal',1)
# myvalid = myvalid.drop('signal',1)

# mydatasets = mydatasets.drop('mnc',1)
# myvalid = myvalid.drop('mnc',1)


#myvalid = pandas.read_csv('../test',sep=',',names=myvalidname)
#dataset = pandas.read_csv(url, names=names)

mydatasets = mydatasets.drop_duplicates()
myvalid = myvalid.drop_duplicates()
#print(mydatasets[(mydatasets['Real_label'] == 0) & (mydatasets['speed'] > 30)])

#print(mydatasets)

#dataset = pandas.read_csv(url)

#print(dataset)

# shape
#print(dataset.shape)
#print(mydatasets.shape)

# head
#print(dataset.head(150))

# descriptions
#print(dataset.describe())
#print(mydatasets.describe())

# class distribution
#print(dataset.groupby('class').size())

# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# mydatasets.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# histograms
# dataset.hist()
# plt.show()

# scatter plot matrix
# scatter_matrix(dataset)
# plt.show()


# Split-out validation dataset
# array = dataset.values
# X = array[:,0:4]
# Y = array[:,4]
# validation_size = 0.20
# seed = 7
# X_train, X_validation, Y_train, Y_validation = \
# model_selection.train_test_split \
# (X, Y, test_size=validation_size, random_state=seed)

# mydatasets['signal'] = (mydatasets['signal']/10).astype(int)
# myvalid['signal'] = (myvalid['signal']/10).astype(int)


# print myvalid['signal']

#-----------------------------------------
myarray = mydatasets.values
myvalidarray = myvalid.values
#originalarray = myvalid[:,1:-1]

X = myarray[:,0:-1]
#X = myarray[:,0:3]
#X = myarray[:,3:-1]
Y = myarray[:,-1]

print(X)

validation_size = 0.20
seed = 7
# X_train, X_validation, Y_train, Y_validation = \
# model_selection.train_test_split \
# (X, Y, test_size=validation_size, random_state=seed)
X_train = X
Y_train = Y

X_validation = myvalidarray[:,1:-1]
#X_validation = myvalidarray[:,1:4]
#X_validation = myvalidarray[:,4:-1]
Y_validation = myvalidarray[:,-1]
Y_validation = np.array(Y_validation, dtype=float)
originalarray = myvalidarray[:,0:-1]



#print(Y_validation)
#print(X_validation)
#print(originalarray)


#print(X)

#print (X_train.size)

#Test options and evaluation metric
seed = 7
scoring = 'accuracy'


models = []
#models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
#models.append(('XGBT', xgb()))
#models.append(('GBT', GradientBoostingClassifier()))

#models.appedd(('RF',))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
#print (models)
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	info.write(msg+'\n')
print('\n')
info.write('\n')
# Compare Algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

def svc(X_train, Y_train,X_validation):
	clf = SVC()
	clf.fit(X_train, Y_train)
	predictions = clf.predict(X_validation)
	return predictions

def linearsvc(X_train, Y_train,X_validation):
	clf = LinearSVC()
	clf.fit(X_train, Y_train)
	predictions = clf.predict(X_validation)
	return predictions
# Make predictions on validation dataset
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# predictions = knn.predict(X_validation)
def knb(X_train, Y_train,X_validation):
	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	predictions = knn.predict(X_validation)
	return predictions
#print(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

# 
# clf = tree.DecisionTreeClassifier()
# print("done1")
# clf = clf.fit(X_train, Y_train)
# print("done2")
# predictions = clf.predict(X_validation)
# print(predictions)


def cart(X_train, Y_train,X_validation):
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, Y_train)
	predictions = clf.predict(X_validation)
	return predictions

#
# clf = GaussianNB()
# print("done1")
# clf = clf.fit(X_train, Y_train)
# print("done2")
# predictions = clf.predict(X_validation)
def gaussianNB(X_train, Y_train,X_validation):
	clf = GaussianNB()
	clf = clf.fit(X_train, Y_train)
	predictions = clf.predict(X_validation)
	return predictions


def decisionTree(X_train, Y_train,X_validation,Y_validation):
	clf = tree.DecisionTreeClassifier()
	#clf.fit(X_train[:,[0,1]], Y_train)
	clf.fit(X_train, Y_train)
	tree.export_graphviz(clf,out_file='tree1.dot')
	#plot_feature(X_train,Y_train,clf)

	#plot_roc(X_validation,Y_validation,clf)

	#return clf.predict(X_validation[:,[0,1]])
	return clf.predict(X_validation)

def randomForest(X_train, Y_train,X_validation,n_estimators):
	clf=RandomForestClassifier(n_estimators=n_estimators,criterion='entropy',max_depth=5)
	clf.fit(X_train,Y_train)
	#plot_roc(X_train,Y_train,clf)
	return clf.predict(X_validation)

def su(X,Y,clf):
	x_min,x_max=X[:,0].min(),X[:,0].max()+1
	y_min,y_max=X[:,1].min(),X[:,1].max()+1
	xx, yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
	Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
	Z = Z.reshape(xx.shape)

	plt.plot()
	plt.contourf(xx,yy,Z,alpha=0.4,cmap=plt.cm.RdYlBu)
	plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.brg)
	plt.show()

def plot_roc(X,Y,clf):
	probas_=clf.predict(X)
	from sklearn.metrics import roc_curve
	from sklearn.metrics import auc
	#print(probas_)

	fpr,tpr, threshold = roc_curve(Y,probas_)
	roc_auc = auc(fpr,tpr)
	plt.plot(fpr,tpr,label='DT-AUC:%.2f'%(auc(fpr,tpr)))	
	plt.plot([0,1],[0,1],'k--')
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.0])
	plt.xlabel('False Positive Rate',fontsize = 20)
	plt.ylabel('True Positive Rate', fontsize = 20)
	plt.legend(loc='lower right',fontsize = 20)
	plt.show()
#------------------------------

#predictions = randomForest(X_train, Y_train, X_validation,100)
predictions = decisionTree(X_train, Y_train, X_validation,Y_validation)

clf1 = tree.DecisionTreeClassifier()
tStart = time.time()
clf1.fit(X_train, Y_train)
tEnd = time.time()
print ('DT, time = %f s'%(tEnd-tStart))
info.write('DT, time = %f s\n'%(tEnd-tStart))
joblib.dump(clf1,'test_DT_hsr.pkl')
#clf1 = joblib.load('test_DT_all.pkl')

# clf2 = LogisticRegression()
# tStart = time.time()
# clf2.fit(X_train, Y_train)
# tEnd = time.time()
# joblib.dump(clf2,'test_LR_hsr.pkl')
#clf2 = joblib.load('test_LR_all.pkl')

clf3 = KNeighborsClassifier()
tStart = time.time()
clf3.fit(X_train, Y_train)
tEnd = time.time()
print ('KNN, time = %f s'%(tEnd-tStart))
info.write('KNN, time = %f s\n'%(tEnd-tStart))
joblib.dump(clf3,'test_KNN_hsr.pkl')
#clf3 = joblib.load('test_KNN_all.pkl')

clf4=RandomForestClassifier(n_estimators=200,criterion='entropy',max_depth=5)
tStart = time.time()
clf4.fit(X_train,Y_train)
tEnd = time.time()
print ('RF_100, time = %f s'%(tEnd-tStart))
info.write('RF_100, time = %f s\n'%(tEnd-tStart))
joblib.dump(clf4,'test_RF_100_5_hsr.pkl')
#clf4 = joblib.load('test_RF_100_5_all.pkl')

# clf2 = RandomForestClassifier(criterion='entropy')

# param_grid = {"n_estimators":[50,100],
# 				"max_depth":[3,4]}

# grid = GridSearchCV(estimator=clf2,param_grid=param_grid,cv=1)
# grid.fit(X_train,Y_train)
# print(grid)
# print(grid.best_params_)
# #print(grid.best_estimater_.alpha)

clf5=RandomForestClassifier(n_estimators=1000,criterion='entropy',max_depth=5)
tStart = time.time()
clf5.fit(X_train,Y_train)
tEnd = time.time()
print ('RF_1000, time = %f s'%(tEnd-tStart))
info.write('RF_1000, time = %f s\n'%(tEnd-tStart))
#joblib.dump(clf5,'total_RF_1000_5.pkl')
joblib.dump(clf5,'hsr_cell_RF_1000_5.pkl')
#clf5 = joblib.load('total_cell_RF_1000_5.pkl') #5
#clf5 = joblib.load('total_gps_RF_1000_5.pkl') #4
#clf5 = joblib.load('total_RF_1000_5.pkl') #6
#clf5 = joblib.load('hsr_RF_1000_5.pkl') #3-1
#clf5 = joblib.load('sub_RF_1000_5.pkl') #3-2
#clf5 = joblib.load('tra_RF_1000_5.pkl') #3-3
#clf5 = joblib.load('hw_RF_1000_5.pkl') #3-4
#clf5 = joblib.load('hw_gps_RF_1000_5.pkl') #1-4
#clf5 = joblib.load('tra_gps_RF_1000_5.pkl') #1-3
#clf5 = joblib.load('sub_gps_RF_1000_5.pkl') #1-2
#clf5 = joblib.load('hsr_gps_RF_1000_5.pkl') #1-1


#clf5 = joblib.load('test_RF_1000_5_all_b.pkl')
#clf5 = joblib.load('test_RF_1000_5_hsr.pkl')


# clf6 = SVC(probability=True)
# tStart = time.time()
# clf6.fit(X_train,Y_train)
# tEnd = time.time()
# print ('SCV, time = %f s'%(tEnd-tStart))
# joblib.dump(clf6,'test_SVM_hsr_b.pkl')
#clf6 = joblib.load('test_SVM_all.pkl')
#clf6 = joblib.load('test_SVM_hsr.pkl')
#clf6 = joblib.load('test_SVM_hsr_b.pkl')

# clf7 = LinearSVC()
# clf7.fit(X_train,Y_train)
# joblib.dump(clf7,'LSVM.pkl')
#clf7 = joblib.load('LSVM.pkl')

clf8 = LinearDiscriminantAnalysis()
tStart = time.time()
clf8.fit(X_train,Y_train)
tEnd = time.time()
print ('LDA, time = %f s'%(tEnd-tStart))
info.write('LDA, time = %f s\n'%(tEnd-tStart))

# clf9 = DecisionTreeClassifier()
# tStart = time.time()
# clf9.fit(X_train,Y_train)
# tEnd = time.time()
# print ('CART, time = %f s'%(tEnd-tStart))

clf10 = xgb.XGBClassifier()
tStart = time.time()
clf10.fit(X_train,Y_train)
tEnd = time.time()
print ('xgb, time = %f s'%(tEnd-tStart))
info.write('xgb, time = %f s\n'%(tEnd-tStart))

clf11 = GradientBoostingClassifier()
tStart = time.time()
clf11.fit(X_train,Y_train)
tEnd = time.time()
print ('gbdt, time = %f s'%(tEnd-tStart))
info.write('gbdt, time = %f s\n'%(tEnd-tStart))


# plt.figure(figsize=[14,6])
# #for clf, title in zip([clf1,clf2,clf3,clf4,clf5,clf6],['DT','LR','KNN','RF(100)','RF(1000)','SVM']):
# for clf, title in zip([clf1,clf2,clf3,clf4,clf5],['DT','LR','KNN','RF(100)','RF(1000)']):
# 	tStart = time.time()
# 	probas_=clf.predict_proba(X_validation)
# 	tEnd = time.time()
# 	#probas_ = clf.fit(X_train,Y_train).predict_proba(X_validation)
# 	fpr, tpr, threshold = roc_curve(Y_validation,probas_[:,1],pos_label =1)
# 	plt.plot(fpr,tpr,label='%s-AUC:%.2f'%(title,auc(fpr,tpr)))

# 	predictions = clf.predict(X_validation)
# 	print(title)
# 	print ('time = %f s'%(tEnd-tStart))
# 	print(accuracy_score(Y_validation, predictions))
# 	print(confusion_matrix(Y_validation, predictions))
# 	print(classification_report(Y_validation, predictions))


# plt.plot([0,1],[0,1],'k--')
# plt.xlim([0.0,1.0])
# plt.ylim([0.0,1.0])
# plt.xlabel('False Positive Rate',fontsize = 20)
# plt.ylabel('True Positive Rate', fontsize = 20)
# plt.legend(loc='lower right',fontsize = 20)
# plt.show()

#print (X_validation)





predictions=clf1.predict(X_validation)
proba = clf1.predict_proba(X_validation)

def output(filename, predictions, proba):
	f = open('../output/'+str(filename),'w')
	ocounter,zcounter = 0,0
	for index in range(len(predictions)):

		for temp in range(len(originalarray[index])):
			if temp ==0:
				f.write("\""+str(originalarray[index][temp])+"\""+'\t')
			else:	
			#print (str(X_validation[index][temp])+'\t'),
				f.write(str(originalarray[index][temp])+'\t')
		#print (str(predictions[index])+'\n')
		f.write(str(predictions[index])+'\t'+str(proba[index][0])+'\t'+str(proba[index][1])+'\n')
		if predictions[index] == 1.0:
			ocounter+=1
		else:
			zcounter+=1
	print "ocounter:" + str(ocounter)
	print "zcounter:" + str(zcounter)


for clf, title in zip([clf1,clf3,clf4,clf5,clf8,clf10,clf11],['DT','KNN','RF(100)','RF(1000)','LDA','xgb','dgbt']):
	tStart = time.time()
	probas_=clf.predict_proba(X_validation)
	tEnd = time.time()
	#probas_ = clf.fit(X_train,Y_train).predict_proba(X_validation)
	#fpr, tpr, threshold = roc_curve(Y_validation,probas_[:,1],pos_label =1)
	#plt.plot(fpr,tpr,label='%s-AUC:%.2f'%(title,auc(fpr,tpr)))

	predictions = clf.predict(X_validation)
	
	filename = title + 'RecResult'
	output(filename,predictions,proba)


	print(title)
	print('time = %f s'%(tEnd-tStart))
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
	print('-------------------------------------------------\n\n')
	# info.write(title)
	# info.write('time = %f s\n'%(tEnd-tStart))
	# info.write(accuracy_score(Y_validation, predictions))
	# info.write(confusion_matrix(Y_validation, predictions))
	# info.write(classification_report(Y_validation, predictions))
	# info.write('-------------------------------------------------\n\n')


#print (proba[1][1])

# ocounter,zcounter = 0,0
# for index in range(len(predictions)):

# 	for temp in range(len(originalarray[index])):
# 		if temp ==0:
# 			f.write("\""+str(originalarray[index][temp])+"\""+'\t')
# 		else:	
# 			#print (str(X_validation[index][temp])+'\t'),
# 			f.write(str(originalarray[index][temp])+'\t')
# 	#print (str(predictions[index])+'\n')
# 	f.write(str(predictions[index])+'\t'+str(proba[index][0])+'\t'+str(proba[index][1])+'\n')
# 	if predictions[index] == 1.0:
# 		ocounter+=1
# 	else:
# 		zcounter+=1
# print "ocounter:" + str(ocounter)
# print "zcounter:" + str(zcounter)


def plot_confusion_matrix(cm, title='Confusion matrix',cmap=plt.cm.Blues):
	import numpy as np
	plt.imshow(cm,interpolation='nearest',cmap = cmap)
	plt.title(title)
	plt.colorbar()
	# tick_marks = np.arrange(len(digits.target_names))
	# plt.xticks(tick_marks, digits.target_names, rotation=45)
	# plt.yticks(tick_marks, digits.target_names)
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	#plt.figure()
	plt.show()




# print (Y_validation)
# print (predictions)


# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))


#plot_confusion_matrix(metrics.confusion_matrix(Y_validation, predictions))


info.close()
