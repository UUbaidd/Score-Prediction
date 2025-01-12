#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pickle
save_path = r"D:\dataset_level2.pkl" 
df_matches= pickle.load(open(save_path, 'rb'))



# In[32]:


#process for filling null values in city column
df_matches[df_matches['city'].isnull()]['venue'].value_counts()


# In[33]:


#taking first word of venue as city name
import numpy as np
cities=np.where(df_matches['city'].isnull(),df_matches['venue'].str.split().apply(lambda x:x[0]),df_matches['city'])


# In[34]:


df_matches.drop(columns=['venue'], inplace=True)


# In[35]:


eligible_cities = df_matches['city'].value_counts()[df_matches['city'].value_counts()>600].index.tolist()


# In[36]:


df_matches=df_matches[df_matches['city'].isin(eligible_cities)]


# In[37]:


df_matches['current_Score'] = df_matches.groupby('match_id')['runs'].cumsum()


# In[38]:


df_matches


# In[39]:


df_matches['over'] = df_matches['ball'].apply(lambda x:str(x).split('.')[0])
df_matches['ball_no'] = df_matches['ball'].apply(lambda x:str(x).split('.')[1])


# In[40]:


eligible_cities


# In[41]:


df_matches


# In[42]:


df_matches['balls_bowled'] = (df_matches['over'].astype('int')*6)+ df_matches['ball_no'].astype('int')


# In[43]:


df_matches['balls_left']=120-df_matches['balls_bowled']
df_matches['balls_left']=df_matches['balls_left'].apply(lambda x:0 if x<0 else x)


# In[44]:


df_matches['player_dismissed'] = df_matches['player_dismissed'].apply(lambda x:0 if x=='0' else 1)
df_matches['player_dismissed'] = df_matches['player_dismissed'].astype('int')
df_matches['player_dismissed'] = df_matches.groupby('match_id')['player_dismissed'].cumsum().clip(upper=10)
df_matches['wicket_left'] = 10 - df_matches['player_dismissed']


# In[45]:


df_matches


# In[69]:


last_five = []

# 1. Get unique match IDs
match_ids = df_matches['match_id'].unique()

# 2. Group the DataFrame by 'match_id'
groups = df_matches.groupby('match_id')

# 3. Calculate rolling sum for each match
for id in match_ids:
    group = groups.get_group(id)  # Fetch group for current match_id
    rolling_sum = group['runs'].rolling(window=30).sum()  # Rolling sum on 'runs'
    last_five.extend(rolling_sum.tolist())


# In[70]:


df_matches['last_five']=last_five


# In[71]:


df_matches.info()


# In[72]:


final_df=df_matches.groupby('match_id').sum()['runs'].reset_index().merge(df_matches,on='match_id')


# In[73]:


df_matches['crr']=(df_matches['current_Score']*6)/df_matches['balls_bowled']


# In[74]:


df_matches


# In[75]:


final_df=final_df[['batting_team','bowling_team','city','current_Score','balls_left','wicket_left','crr','last_five','runs_x']]


# In[76]:


final_df


# In[77]:


final_df.dropna(inplace=True)


# In[78]:


final_df.info()


# In[79]:


final_df=final_df.sample(final_df.shape[0])


# In[80]:


X = final_df.drop(columns=['runs_x'])
Y = final_df['runs_x']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=1)


# In[81]:


X_train


# In[82]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
from sklearn.metrics import r2_score , mean_absolute_error


# In[83]:


trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False, drop='first'),['batting_team','bowling_team','city'])
], remainder='passthrough')


# In[84]:


pipe = Pipeline(steps =[
    ('step1',trf),
    ('step2',StandardScaler()),
    ('step3',XGBRegressor(n_estimators=1000,learning_rate=0.2,max_depth=12,random_state=1))
])


# In[85]:


pipe.fit(X_train, Y_train)
y_pred = pipe.predict(X_test)
print(r2_score(Y_test, y_pred))
print(mean_absolute_error(Y_test,y_pred))


# In[63]:


# import joblib

# # Save the pipeline with compression (level 9 for maximum compression)
# joblib.dump(pipe, 'cricket_score_predictor.pkl', compress=9)
# pickle.dump(pipe,open('pipe.pkl1','wb'))


# In[ ]:




