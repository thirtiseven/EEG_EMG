import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import lightgbm as lgb


data = np.load('vector_auto_regression_data.npy')
labels = np.load('eeg_emg_label.npy')

data = data.reshape(data.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)

#X_train = train_x.values
#X_test = test_x.values
#y_train = train_y.values
#y_test = test_y.values

params = {'num_leaves': 60, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
	'min_data_in_leaf': 30,
	'objective': 'binary', #定义的目标函数
	'max_depth': -1,
	'learning_rate': 0.03,
	"min_sum_hessian_in_leaf": 6,
	"boosting": "gbdt",
	"feature_fraction": 0.9,  #提取的特征比率
	"bagging_freq": 1,
	"bagging_fraction": 0.8,
	"bagging_seed": 11,
	"lambda_l1": 0.1,             #l1正则
	# 'lambda_l2': 0.001,     #l2正则
	"verbosity": -1,
	"nthread": -1,                #线程数量，-1表示全部线程，线程越多，运行的速度越快
	'metric': {'binary_logloss', 'auc'},  ##评价函数选择
	"random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
	# 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
}
print('Training...')
trn_data = lgb.Dataset(X_train, y_train)
val_data = lgb.Dataset(X_test, y_test)
clf = lgb.train(params, 
	trn_data, 
	num_boost_round = 1000,
	valid_sets = [trn_data,val_data])

print('Predicting...')
y_prob = clf.predict(X_test, num_iteration=clf.best_iteration)
y_pred = [int(x+0.5) for x in y_prob]

print(y_pred)

print(y_test)

print("AUC score: {:<8.5f}".format(sklearn.metrics.accuracy_score(y_pred, y_test)))
