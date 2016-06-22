
# coding: utf-8

# In[3]:

import matplotlib 
get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

from __future__ import division
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score


# ## Cleaning and overview of the data. 

# In[4]:

MyFile = r"C:\Users\Bill\Desktop\Tahzoo.com Redesign\GoogleAnalytics\Experiments\labels_categories_actions.tsv"
d1 = pd.read_csv(MyFile, sep="\t")
d1.head()


# I'm going to use the "Apply" Button as a proxy for a job seeker. By doing this I'm saying: "A job seeker is someone who clicks on the "Apply" button at least once. You could take any token from the event Category (SUBMIT, PHONE NOW, or any other DOM item that was sent to Google Analytics).

# In[5]:

d1[d1['eventCategory'].apply(lambda x: "Apply" in x)]


# In[21]:

d1["JobSeeker"] = d1["eventCategory"].apply(lambda x: "Yes" if "Apply" in x else "No")
d1["cleanText"] = d1["eventCategory"].apply(lambda x: x.replace(" ","").replace("::",""))
d1.head()


# In[29]:

df = pd.DataFrame()
list_of_ids = np.unique(d1.eventLabel)
for iter,id in enumerate(list_of_ids):
    allEvents = d1[d1["eventLabel"]==id]["cleanText"].tolist()
    allJobs = d1[d1["eventLabel"]==id]["JobSeeker"].tolist()
    df.loc[iter,"Events"] = " ".join(allEvents)
    if "Yes" in allJobs:
        df.loc[iter,"JobSeeker"] = "Yes"
    else:
        df.loc[iter,"JobSeeker"] = "No"


# In[58]:

pd.concat([df[df["JobSeeker"]=="Yes"].head(10),df[df["JobSeeker"]=="No"].head(10)])


# In[33]:

len(df[df['JobSeeker']=="Yes"])/len(df)


# ## Converting the events to lists of tokens
# In order to predict using the model you have to turn the terms in the text into a 'bag of words' or collection of unigrams. Then the presense or absens of that unigram can be used to predict outcomes. 

# In[73]:

vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(df.Events.tolist())
y = df.JobSeeker.tolist()
X


# In[74]:

X.toarray()


# In[63]:

Counter(X.toarray()[1])


# # Modeling
# I'm just using a Decision Tree Classifier and Logistic Regression as an examples. Further testing of live datasets would show what the best features are. 

# ## Decision Tree Classifier
# Because this model draws a series of decisions around the dataset it will always be 100% accurate. It may have lower performance on newly introduced users. 

# In[118]:

DTclf = DecisionTreeClassifier(random_state=0)
DTclf.fit(X, y)


# In[122]:

DTclf.predict(X)
df["prediction"] = DTclf.predict(X)


# In[123]:

def get_results_matrix(df):  #I'll be using this function over and over again. 
    Right_and_Yes = df[(df["JobSeeker"]==df["prediction"])&(df["JobSeeker"]=="Yes")]
    Right_and_No = df[(df["JobSeeker"]==df["prediction"])&(df["JobSeeker"]=="No")]
    Wrong_and_Yes = df[(df["JobSeeker"]!=df["prediction"])&(df["JobSeeker"]=="Yes")]
    Wrong_and_No = df[(df["JobSeeker"]!=df["prediction"])&(df["JobSeeker"]=="No")]
    
    results1 = pd.DataFrame(index=["Right_and_Yes","Right_and_No","Wrong_and_Yes","Wrong_and_No"], columns=["count","percent"])
    results1.loc["Right_and_Yes",["count","percent"]] = [len(Right_and_Yes),len(Right_and_Yes)/len(df)]
    results1.loc["Right_and_No",["count","percent"]] = [len(Right_and_No),len(Right_and_No)/len(df)]
    results1.loc["Wrong_and_Yes",["count","percent"]] = [len(Wrong_and_Yes),len(Wrong_and_Yes)/len(df)]
    results1.loc["Wrong_and_No",["count","percent"]] = [len(Wrong_and_No),len(Wrong_and_No)/len(df)]
    results1

    results2 = pd.DataFrame(index=["Guess:Right","Guess:Wrong"], columns=["Applied:Yes","Applied:No"])
    results2.loc["Guess:Right","Applied:Yes"] = len(Right_and_Yes)
    results2.loc["Guess:Wrong","Applied:No"] = len(Wrong_and_No)
    results2.loc["Guess:Wrong","Applied:Yes"] = len(Wrong_and_Yes)
    results2.loc["Guess:Right","Applied:No"] = len(Right_and_No)
    results2
    
    print results2["Applied:Yes"].apply(lambda x: x/results2["Applied:Yes"].sum())
    return results1,results2


# In[124]:

results1,results2 = get_results_matrix(df)
results1


# In[125]:

results2


# ## Logistic Regression

# In[126]:

from sklearn.linear_model import LogisticRegression


# In[127]:

LRclf = LogisticRegression(random_state=0)
LRclf.fit(X, y)


# In[128]:

LRclf.predict(X)


# In[129]:

df["prediction"] = LRclf.predict(X)
#df["estimateProbability"] = LRclf.predict_log_proba(X)
pd.concat([df[df["JobSeeker"]=="Yes"].head(10),df[df["JobSeeker"]=="No"].head(10)])


# In[130]:

results1,results2 = get_results_matrix(df)
results1


# In[131]:

results2


# In[144]:

len(LRclf.coef_[0])==len(X.toarray()[1])


# In[148]:

token_values = pd.DataFrame()

token_values["names"] = vectorizer.get_feature_names()
token_values["score"] = LRclf.coef_[0]
token_values.sort("score",ascending=False).head()


# In[ ]:



