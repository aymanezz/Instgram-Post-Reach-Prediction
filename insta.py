import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_squared_error

df = pd.read_csv('instagram_reach.csv')
#print(df.head())
df = df.drop(['USERNAME','Hashtags','S.No','Caption'],axis = 1)
df['Time since posted'] = df['Time since posted'].str.split(r'\D').str.get(0)
X = df.iloc[:,1:3]
y = df['Likes']
print(X.shape)
print(y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
#print(X)

#create model
model = LinearRegression()
model.fit(X_train,y_train)

model.predict([[300,10]])
predicted = model.predict(X_test)



# save the model to disk
filename = 'regressionmodel.h5'
joblib.dump(model, filename)

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)
#get mean square error
mse = mean_squared_error(y_test,predicted)
print(mse)

