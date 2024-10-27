from DataProcessing import Preprocess
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("./Data/train.csv")
process = Preprocess()

df = process.cleaning(df)


df , label = process.labelEncoding(df)
x = df.drop(columns={'loan_amnt'})
y = df['loan_amnt']
x_train , y_train ,x_test ,y_test = process.datasplit(x,y)

x_train ,x_test ,scaler = process.standardize(x_train,x_test)

model = DecisionTreeRegressor()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

process.evaluation(y_test,y_pred)


artifacts  = {
    'model':model,
    'scaler':scaler,
    'label':label
}
print(df.columns)
with open("./Model/model1.pkl", 'wb') as file:
    pickle.dump(artifacts, file)
    
print("Model pickle saved to model folder")