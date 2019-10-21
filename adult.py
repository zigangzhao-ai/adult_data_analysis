import pandas as pd
from sklearn.cross_validation import train_test_split 
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('adult.csv',header=None,index_col=False,
                  names=['age','company','weight','education','end-time','marriage','career','family',
                  'race','sex','mony','loss','week-time','location','get'])

#display(data.head)
data_lite = data[['age','company','education','sex','week-time','career','get']]
data_dummies = pd.get_dummies(data_lite)
print('original features:\n',list(data_lite.columns),'\n')
print('virtual features:\n',list(data_dummies.columns),'\n')
print(data_dummies.head())

features = data_dummies.loc[:,'age':'career_ Transport-moving']
X = features.values
y =data_dummies['get_ >50K'].values

print('\n\n\n')
print('result:')

print('features.state:{} labels.state:{}'.format(X.shape,y.shape))
print('\n=============================')

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
go_tree = DecisionTreeClassifier(max_depth=5)
go_tree.fit(X_train,y_train)
print('\n=============')

print('model score:{:.2f}'.format(go_tree.score(X_test,y_test)))
print('===============')

Mr_z = [[37,40,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

data_dec = go_tree.predict(Mr_z)
print('==============')
if data_dec ==1:
    print("go on")
else:
    print("give up")
    
print('==========')