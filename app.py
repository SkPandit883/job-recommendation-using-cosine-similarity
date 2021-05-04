from flask import Flask,request,jsonify
import  pandas as pd
import numpy as np
from array import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route('/')
def hello_world():
   def get_id_from_index(index):
        return df3[df3.index == index]["id"].values[0]
   profile_data={
    'id':[0,1,2,3,4,5,6],
    'name':['santosh','suraj','shuvam','kritish','pujan','samir','a'],
    'location':['saptari','mahendranagar','butwal','butwal','butwal','belbari','b'],
    'skills':['htlm laravel machine learning','flutter html css','nodejs html css reactjs','nodejs laravel css html','aurdino aurdino machine learning','django python  html','html machinelearning'],
    'degree':['bachelor','bachelor','master','slc','+2','master','bachelor']
   }

   df1=pd.DataFrame(profile_data)
   job_skills={
    'id':['test'],
    'name':['a'],
    'location':['a'],
    'skills':['laravel'],
    'degree':['a']
    }

   df2=pd.DataFrame(job_skills)
   df3=df1.append(df2,ignore_index=True)
   last_index=df3.index[-1]
   cv=CountVectorizer()

   profile_count_matrix=cv.fit_transform(df3["skills"])        
   cosine_sim = cosine_similarity(profile_count_matrix,dense_output=False) 

   similar_user =  cosine_sim[last_index]
   similar_user=similar_user.toarray()
   similar_user=similar_user[0]
   user_ids=[]
   i=0
   for user in similar_user:
       user_id=get_id_from_index(i)
       if(user_id!='test'):
            if(user>=0.2):
                user_ids.append(user_id)
       i=i+1
   return jsonify(
      userId=user_ids
   )
  

   

if __name__ == "__main__":
    app.run(debug=True)