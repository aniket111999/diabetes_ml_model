import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv("diabetes.csv")
df['BMI']=df['BMI'].astype(int)

from sklearn.model_selection import train_test_split
columns=['Insulin', 'Glucose', 'Age','BMI']
x=df[columns]
y=df['Outcome']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
model_ml = RandomForestRegressor(n_estimators=100,random_state=0)

#def score_model(model, x_t=x_train, x_v=x_test, y_t=y_train, y_v=y_test):
    #model.fit(x_t,y_t)
    #pred = model.predict(x_v)
    #return mean_absolute_error(y_v,pred)

#mae = score_model(model)
#print("model mae:",mae)
model_ml.fit(x_train,y_train)

pickle.dump(model_ml,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
