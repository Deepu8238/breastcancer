import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle


def cleaned_data():
    data=pd.read_csv(r"data/data.csv")
    print(data.head())
    
    data=data.drop(['Unnamed: 32','id'],axis=1)
    data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
    return data    

def create_model(data):
    X=data.drop(['diagnosis'],axis=1)
    Y=data['diagnosis']
    
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    
    #training the model
    X_train, X_test, Y_train, Y_test=train_test_split(
        X,Y , test_size=0.2, random_state=42
    )    
    model=LogisticRegression()    
    model.fit(X_train,Y_train) 
    
    #testing the model
    Y_pred=model.predict(X_test)
    print("Accuracy of the model:",accuracy_score(Y_test,Y_pred))
    print("Classifction report: ",classification_report(Y_test,Y_pred)) 
    return model,scaler



def main():
   data=cleaned_data() 
   
   model,scaler=create_model(data)
   with open('model/model.pkl','wb') as f:
       pickle.dump(model,f)
       
   with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)   
   
    
if __name__=="__main__":
    main()    
    
    
    