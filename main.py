import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, metrics


column_names = ['age', 'gender', 'polyuria', 'polydipsia', 'weight_loss', 'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring', 'itching', 'irritability', 'delayed_healing', 'partial_paresis', 'muscle_stiffness', 'alopecia', 'obesity', 'has_diabetes']
raw_data_name = ['data']
df_raw_data = pd.read_csv('diabetes_data.csv', names=raw_data_name)
df_raw_data['data_list'] = df_raw_data['data'].str.split(';')
df_raw_data = pd.DataFrame(df_raw_data['data_list'].tolist())
df_raw_data.drop(index=df_raw_data.index[0], axis=0, inplace=True)
df = pd.DataFrame(columns=column_names)
df = pd.concat([df_raw_data], axis=0)
df.columns = column_names
print(df)


df['gender'] = np.where(
   (df['gender'] == 'Male'), 0, df['gender']
   )

df['gender'] = np.where(
   (df['gender'] == 'Female'), 1, df['gender']
   )


my_model = linear_model.LogisticRegression(max_iter=600)

y = df.values[:,16]
x = df.values[:,0:16]

my_model.fit(x,y)

y_pred = my_model.predict(x)

print(metrics.accuracy_score(y, y_pred))
