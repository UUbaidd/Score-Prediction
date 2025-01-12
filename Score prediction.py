#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from yaml import safe_load
from tqdm import tqdm


# In[2]:


filenames=[]
for file in os.listdir('Player Rating'):
    filenames.append(os.path.join('Player Rating',file))


# In[3]:


filenames[0:5]


# In[6]:


final_df = pd.DataFrame()
counter = 1

# Assuming 'filenames' is a list of file paths to your YAML files
for file in tqdm(filenames):
    with open(file, 'r') as f:
        df = pd.json_normalize(safe_load(f))  # Load the JSON/YAML data and normalize
        df['match_id'] = counter  # Add the match_id column
        final_df = pd.concat([final_df, df], ignore_index=True)  # Concatenate DataFrames
        counter += 1

print(final_df.head())  


# In[7]:


backup = final_df.copy


# In[25]:


final_df.columns


# In[8]:


final_df


# In[9]:


#droping useless columns
final_df.drop(columns=[
    'meta.data_version',
    'meta.created',
    'meta.revision',
    'info.outcome.bowl_out',
    'info.bowl_out',
    'info.supersubs.South Africa',
    'info.supersubs.New Zealand',
    'info.outcome.eliminator',
    'info.outcome.result',
    'info.outcome.method',
    'info.neutral_venue',
    'info.match_type_number',
    'info.outcome.by.runs',
    'info.outcome.by.wickets'
    
],inplace=True)


# In[10]:


final_df['info.gender'].value_counts()


# In[11]:


final_df['info.overs'].value_counts()


# In[12]:


final_df['info.match_type'].value_counts()


# In[13]:


final_df=final_df[final_df['info.overs']==20]
final_df.drop(columns=['info.overs','info.match_type'],inplace=True)
final_df


# In[14]:


import pickle
save_path = r"D:\dataset_level1.pkl"  

with open(save_path, 'wb') as file:
    pickle.dump(final_df, file)

print(f"File saved successfully at {save_path}")


# In[6]:


import pickle
save_path = r"D:\dataset_level1.pkl" 
matches = pickle.load(open(save_path, 'rb'))
matches.iloc[0]['innings']


# In[7]:


#Extracting insight data from dataset
count = 1
delivery_df = pd.DataFrame()

for index, row in matches.iterrows():
    # Skip specific match IDs
    if count in [75, 108, 150, 180, 268, 360, 443, 458, 584, 748, 982, 1052, 1111, 1226, 1345]:
        count += 1
        continue
    
    count += 1
    
    # Initialize lists
    ball_of_match = []
    batsman = []
    bowler = []
    runs = []
    player_of_dismissed = []
    teams = []
    batting_team = []  # Initialize the batting_team list
    match_id = []
    city = []
    venue = []

    # Process deliveries
    for ball in row['innings'][0]['1st innings']['deliveries']:
        for key in ball.keys():
            try:
                # Extract basic information
                match_id.append(count)
                teams.append(row['info.teams'])  # Correctly access the column `info.teams`
                ball_of_match.append(key)
                batsman.append(ball[key]['batsman'])
                bowler.append(ball[key]['bowler'])
                runs.append(ball[key]['runs']['total'])
                batting_team.append(row['innings'][0]['1st innings']['team'])  # Add batting team
                city.append(row['info.city'])  # Directly access `info.city` column
                venue.append(row['info.venue'])  # Directly access `info.venue` column

                # Handle dismissed player
                if 'wicket' in ball[key]:
                    # Check if `wicket` is a list
                    if isinstance(ball[key]['wicket'], list):
                        # Extract all dismissed players from the list
                        dismissed_players = [
                            dismissal['player_out'] for dismissal in ball[key]['wicket'] if 'player_out' in dismissal
                        ]
                        # Join the names into a single string (comma-separated)
                        player_of_dismissed.append(", ".join(dismissed_players))
                    elif isinstance(ball[key]['wicket'], dict):
                        # Handle single dismissal
                        player_of_dismissed.append(ball[key]['wicket']['player_out'])
                    else:
                        player_of_dismissed.append(None)
                else:
                    player_of_dismissed.append(None)
            except Exception as e:
                print(f"Error processing ball: {ball}, error: {e}")
                continue

    # Create the loop DataFrame
    loop_df = pd.DataFrame({
        'match_id': match_id,
        'teams': teams,
        'ball': ball_of_match,
        'batsman': batsman,
        'bowler': bowler,
        'runs': runs,
        'player_dismissed': player_of_dismissed,
        'batting_team': batting_team,  # Include batting_team in the DataFrame
        'city': city,
        'venue': venue,
    })

    # Concatenate the DataFrame
    delivery_df = pd.concat([delivery_df, loop_df], ignore_index=True)


# In[8]:


delivery_df


# In[13]:


#we have batting team column, now we will create bowling_team column
def bowl(row):
    for team in row['teams']:
        if team != row['batting_team']:
            return team


# In[14]:


delivery_df['bowling_team']=delivery_df.apply(bowl,axis=1)


# In[15]:


delivery_df


# In[16]:


delivery_df.drop(columns=['teams'],inplace=True)


# In[17]:


#Keeping only top 10 teams of recent years, to avoid biasness
teams=[
    'Australia',
    'India',
    'Bangladesh',
    'New Zealand',
    'South Africa',
    'England',
    'Pakistan',
    'Afganistan',
    'Srilanka',
    'West Indies'
]


# In[19]:


delivery_df=delivery_df[delivery_df['batting_team'].isin(teams)]
delivery_Df=delivery_df[delivery_df['bowling_team'].isin(teams)]


# In[20]:


delivery_df


# In[21]:


output = delivery_df[['match_id','batting_team','bowling_team','ball','runs','player_dismissed','city','venue']]


# In[22]:


output


# In[23]:


save_path = r"D:\dataset_level2.pkl"  

with open(save_path, 'wb') as file:
    pickle.dump(output, file)

print(f"File saved successfully at {save_path}")


# In[1]:


import pickle
save_path = r"D:\dataset_level2.pkl" 
df_matches= pickle.load(open(save_path, 'rb'))



# In[2]:


df_matches


# In[3]:


#process for filling null values in city column
df_matches[df_matches['city'].isnull()]['venue'].value_counts()


# In[4]:


#taking first word of venue as city name
import numpy as np
cities=np.where(df_matches['city'].isnull(),df_matches['venue'].str.split().apply(lambda x:x[0]),df_matches['city'])


# In[5]:


df_matches['city'] = cities


# In[6]:


df_matches.drop(columns=['venue'],inplace=True)


# In[7]:


df_matches


# In[8]:


#taking only those cities in which minimum 5 matches has been played
#600 is number of balls
eligible_cities = df_matches['city'].value_counts()[df_matches['city'].value_counts()>600].index.tolist()


# In[9]:


df_matches=df_matches[df_matches['city'].isin(eligible_cities)]


# In[10]:


pip install --upgrade pandas


# In[11]:


df_matches['current_Score'] = df_matches.groupby('match_id')['runs'].cumsum()


# In[12]:


df_matches


# In[13]:


df_matches['over'] = df_matches['ball'].apply(lambda x:str(x).split('.')[0])
df_matches['ball_no'] = df_matches['ball'].apply(lambda x:str(x).split('.')[1])


# In[14]:


df_matches


# In[15]:


df_matches['balls_bowled'] = (df_matches['over'].astype('int')*6)+ df_matches['ball_no'].astype('int')


# In[16]:


df_matches['balls_left']=120-df_matches['balls_bowled']
df_matches['balls_left']=df_matches['balls_left'].apply(lambda x:0 if x<0 else x)


# In[21]:


df_matches['player_dismissed'] = df_matches['player_dismissed'].apply(lambda x:0 if x=='0' else 1)
df_matches['player_dismissed'] = df_matches['player_dismissed'].astype('int')
df_matches['player_dismissed'] = df_matches.groupby('match_id')['player_dismissed'].cumsum()
df_matches['wicket_left'] = 10 - df_matches['player_dismissed']


# In[28]:


import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# View the DataFrame
print(df_matches)


# In[ ]:




