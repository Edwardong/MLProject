####################
###   READ  ME   ###
####################
#******************************************
#USE DIFFERENT MODULE(PARAMETERS TUNING or TRAIN&TEST) THEN RUN THIS FILE
#
#******************************************



import time
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
#import fashion_mnist_load as mnist_load
import LoadData
import plt_roc as pr
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV


# Random Forest
def random_forest(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    print('************* Random Forest ************')
	
    model = RandomForestClassifier(n_estimators=250, max_depth=50, max_features = 'auto', criterion='entropy', n_jobs=-1)
    model.fit(train_x, train_y)
    return model




def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid("on")    


if __name__ == '__main__':
    print('reading training and testing data...')
    train_x, train_y, test_x, test_y = LoadData.Load()

    print(train_x.shape,train_y.shape, test_x.shape, test_y.shape)
# '''
# ####################
# #train model & test#
# ####################
# #**************************************************************************************
#     start_time = time.time()


#     model = random_forest(train_x, train_y)

#     print('training took %fs!' % (time.time() - start_time))

#     start_time = time.time()


#     predict = model.predict(test_x)

#     print('predict took %fs!' % (time.time() - start_time))

#     print(classification_report(test_y, predict))

#     precision = metrics.precision_score(test_y, predict,average='macro')
#     recall = metrics.recall_score(test_y, predict,average='macro')
#     print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
#     accuracy = metrics.accuracy_score(test_y, predict)
#     print('accuracy: %.2f%%' % (100 * accuracy))
#     pr.plt_roc(model,test_x,test_y)
# #*************************************************************************************
# '''


############################################
#grid search of n_estimators & max_features#
############################################
#****************************************************************************************************************
#     param_test1 = {'n_estimators':range(25,251,25),'max_features':['auto','log2']}
#     gsearch1 = GridSearchCV(estimator = RandomForestClassifier(max_depth=50,criterion='entropy',n_jobs=-1, 
#         ), 
#                        param_grid = param_test1, scoring='accuracy',cv=5,n_jobs = -1)
#     gsearch1.fit(train_x,train_y)
#     print(gsearch1.cv_results_)
#     print(gsearch1.best_estimator_,gsearch1.best_params_)



#     # Calling Method
#     plt.figure() 
#     plot_grid_search(gsearch1.cv_results_, np.arange(25,251,25), ['auto','log2'], 'N Estimators', 'Max Features')
# #********************************************************************************************************************



# ##############################################
# #grid search of min_samples_split & max_depth#
# ##############################################
# #********************************************************************************************************************
#     param_test2 = {'min_samples_split':range(60,160,20), 'max_depth':range(10,131,20)}
#     gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 150,criterion='entropy',n_jobs=-1, 
#         max_features = 'auto'), 
#                        param_grid = param_test2, scoring='accuracy',cv=5,n_jobs = -1)
#     gsearch2.fit(train_x,train_y)
#     print(gsearch2.cv_results_)
#     print(gsearch2.best_estimator_,gsearch2.best_params_)



#     # Calling Method
#     plt.figure() 
#     plot_grid_search(gsearch2.cv_results_, np.arange(60,160,20), np.arange(10,131,20), 'Min Samples Split', 'Max Depth')
# #********************************************************************************************************************


##################################
#grid search of min_samples_split#
##################################
#********************************************************************************************************************
    # param_test3 = {'min_samples_split':range(2,63,10),'max_features':['auto'], 'max_depth':[50]}
    # gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 150,max_depth=50,criterion='entropy',n_jobs=-1, 
    #     ), 
    #                    param_grid = param_test3, scoring='accuracy',cv=5,n_jobs = -1)
    # gsearch3.fit(train_x,train_y)
    # print(gsearch3.cv_results_)
    # print(gsearch3.best_estimator_,gsearch3.best_params_)



    # # Calling Method
    # plt.figure() 
    # plot_grid_search(gsearch3.cv_results_, np.arange(2,63,10), ['auto'], 'Min Samples Split', 'Max Features')
#********************************************************************************************************************

####################
#train model & test#
####################
#********************************************************************************************************************
    start_time = time.time()


    model = random_forest(train_x, train_y)

    print('training took %fs!' % (time.time() - start_time))

    start_time = time.time()


    predict = model.predict(test_x)

    print('predict took %fs!' % (time.time() - start_time))

    print(classification_report(test_y, predict))

    precision = metrics.precision_score(test_y, predict,average='macro')
    recall = metrics.recall_score(test_y, predict,average='macro')
    print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    accuracy = metrics.accuracy_score(test_y, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))
    pr.plt_roc(model,test_x,test_y)
#********************************************************************************************************************






    

