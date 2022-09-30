import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
#Drop null values. 
df_healthcare = pd.read_csv("Healthcare_Dataset.csv", sep=",")
df_healthcare.dropna()
#Replace string variables with numbers to make the module more accurate. 

for i in df_healthcare.columns:
    if(df_healthcare[i].unique().shape[0] ==2):
        df_healthcare[i] = df_healthcare[i].map({df_healthcare[i].unique()[0]:0, df_healthcare[i].unique()[1]:1})  
df_healthcare_final = pd.get_dummies(df_healthcare, columns=['Race','Ethnicity','Region','Age_Bucket','Ntm_Speciality','Ntm_Speciality_Bucket','Risk_Segment_During_Rx','Tscore_Bucket_During_Rx','Change_T_Score','Change_Risk_Segment'])
     
y  = df_healthcare_final['Persistency_Flag']
x = df_healthcare_final.drop(['Persistency_Flag', 'Ptid'], axis =1)


trainx, testx, trainy, testy = train_test_split(x,y, test_size=0.3, random_state=19)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=1000)

logreg.fit(trainx, trainy)

pickle.dump(logreg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))