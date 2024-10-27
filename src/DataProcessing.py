import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler ,LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class Preprocess:

    def dataframe(self,df):
        df = df[df['loan_status']==1]    
        return df
        
    def drop_columns(self,df):
        df = df.drop(columns ={'id','loan_status'})
        return df
        

    def coldatatype(self,df):
        numeric_columns =df.select_dtypes(exclude = 'object').drop(columns='loan_amnt')
        
        self.numeric_columns = numeric_columns.columns
        self.categoric_columns = df.select_dtypes(include = 'object').columns


    def datasplit(self,x,y):
        x_train , x_test ,y_train , y_test = train_test_split(x,y,test_size =0.2,random_state =42) 
        return x_train , y_train ,x_test ,y_test
    
    def labelEncoding(self,df):
        label_encoders = {}

        for column in self.categoric_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
        return df ,label_encoders    
                


    def standardize(self,x_train,x_test):

        scaler = MinMaxScaler() 
        x_train[self.numeric_columns] = scaler.fit_transform(x_train[self.numeric_columns])
        x_test[self.numeric_columns] = scaler.transform(x_test[self.numeric_columns])

        return x_train ,x_test ,scaler
    
    def cleaning(self,df):
        df = self.dataframe(df)
        df = self.drop_columns(df)
        self.coldatatype(df)
        return df


    def evaluation(self,y_test,y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("Mean Absolute Error (MAE):", round(mae, 2))
        print("Mean Squared Error (MSE):", round(mse, 2))
        print("Root Mean Squared Error (RMSE):", round(rmse, 2))
        print("R-squared (R2) Score:", round(r2, 2))


    

        




