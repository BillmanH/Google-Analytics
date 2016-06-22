
# coding: utf-8

# ### loading and cleaning

# In[70]:

import matplotlib 
get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from __future__ import print_function
from time import time
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[2]:

n_samples = 2000
n_features = 10000
n_topics = 10
n_top_words = 20


# In[11]:

MyFile = r"C:\Users\Bill\Desktop\Tahzoo.com Redesign\GoogleAnalytics\Experiments\labels_categories_actions.tsv"
d1 = pd.read_csv(MyFile, sep="\t")
d1.head()


# In[12]:

d1 = d1[d1['eventLabel'].apply(lambda x: "GA1." in x)] 


# Using individual events to infer a ‘topicality’ is similar to the assumptions in TOPIC MODELING: Assuming that an inherent theme in the user's behavior can be derived from the individual actions (.
# I've constructed a corpus as a list of events. So you can see here that this person clicked on "JOIN", "READ MORE" and "BUSINESS OPERATIONS ANALYST AVAILABLE IN:". Those are a "Bag-of-events" lists that will be used to build the Personas. 

# In[88]:

corpus = {}
for i in np.unique(d1['eventLabel']):       #very long cleaning syntax :(
    context = "::".join(d1[d1.eventLabel == i].eventCategory.tolist())
    for word in bad_words:
        context = context.replace(word," ")
    cleaned_context = "".join([word for word in context.split(" ") if len(word)>1]).replace("::"," ")
    corpus[i] = cleaned_context


# In[89]:

len(corpus)


# In[90]:

corpus.keys()[1]


# Because the LDA model is traditionally used to find "topics" in text documents, I'm joining the text like so in order to create a " " separated list of tokens. 

# In[91]:

corpus[corpus.keys()[1]]


# In[92]:

corpus.keys()[35]


# In[93]:

corpus[corpus.keys()[35]]


# Note that I’m using the inner HTML and class of the DOM element. In the current Tahzoo.com site the DOM elements have unspecific or unclear names. Inner HTML was substituted because it gives us some sense of what the user actually clicked on. 
# * Better labels in DOM elements would make better models
# * Other kinds of events (hover, scroll, backup, etc.) would also be useful
# 
# These are not implemented here as this is merely a POC.
# 

# ### This is the part where we build the actual model

# In[94]:

data_samples = corpus.values()
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(data_samples)


# In[95]:

print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
lda.fit(tf)
tf_feature_names = tf_vectorizer.get_feature_names()


# In[96]:

#bunch of functions that I wrote to build the tables for the model:
def get_single_topic(lda, tf_feature_names, n_top_words, topic):
    words = [tf_feature_names[i] for i in lda.components_[topic].argsort()[:-n_top_words - 1:-1]]
    scores = lda.components_[topic][lda.components_[topic].argsort()[:-n_top_words - 1:-1]]
    df = pd.DataFrame(index=words,columns=['topic_{}'.format(topic)],data=scores)
    return df

def get_all_topics(lda, tf_feature_names, n_top_words,n_topics):
    df = pd.DataFrame()
    for topic in range(n_topics):
        tmpdf = get_single_topic(lda, tf_feature_names, n_top_words, topic)
        for item in tmpdf.index:
            df.loc[item,'topic_{}'.format(topic)] = tmpdf.loc[item,'topic_{}'.format(topic)]
    return df

def get_topic_names(lda, tf_feature_names, n_top_words, n_topics):
    themes = pd.Series()
    for topic in range(n_topics):
        theme = " ".join([tf_feature_names[i] for i in lda.components_[topic].argsort()[:-n_top_words - 1:-1]])
        themes.loc['topic_{}'.format(topic)] = theme
    return themes

df = get_all_topics(lda, tf_feature_names, n_top_words,n_topics)


# In[97]:

df.head()


# In[98]:

def score_document(doc_dic,df,lda, tf_feature_names, n_top_words, n_topics,
                    returnDF=True,confidence=.01):
    '''
    gives scores to the origional document, assigning a category to each one. 

    returnDF : By default returns a DataFrame, set to false to return a dict.
    confidence : this is the threashold that the model must meet to match the document to a topic.
    set to .01 to include practically everything, set to .99 to include almost nothing.


    document_scores = score_document(doc_dic,df,lda, tf_feature_names, n_top_words, n_topics)
    '''
    results_dict = {}
    for key in doc_dic.keys():
        document = doc_dic[key]
        words = [tf_feature_names[i] for i in tf.getrow(doc_dic.keys().index(key)).indices]
        scores = df[[word in words for word in df.index]]
        TM_Score = pd.DataFrame()
        TM_Score['docScore'] = scores.sum() 
        TM_Score['globalScore'] = df.sum()
        TM_Score['relevance'] = TM_Score['docScore']/TM_Score['globalScore']
        TM_Score['theme'] = get_topic_names(lda, tf_feature_names, n_top_words, n_topics)
        results = TM_Score['relevance'].fillna(0).to_dict()
        results['document'] = document
        results['top_score'] = TM_Score['relevance'].max()

        if TM_Score['relevance'].max() >= confidence:
            results['top_theme'] = TM_Score['theme'][TM_Score['relevance'].tolist().index(TM_Score['relevance'].max())]
            results['top_topic'] = TM_Score.index[TM_Score['relevance'].tolist().index(TM_Score['relevance'].max())]
        else: 
            results['top_theme'] = 'unassigned'
            results['top_topic'] = 'unassigned'
        results_dict[key] = results
    if returnDF:
        return pd.DataFrame(results_dict).T
    else:
        return results_dict

    
Scored_corpus = score_document(corpus,df,lda, tf_feature_names, n_top_words, n_topics)


# In[67]:

Scored_corpus.head()


# In[73]:

Scored_corpus['top_score'].hist(bins=60)

labels,values = zip(*Counter(Scored_corpus['top_topic'].tolist()).items())

topic_names = [label for label in labels if label != 'unassigned']

indexes = np.arange(len(labels))
width = 1
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5)
plt.xlabel(labels)
plt.xkcd()
plt.show()


# You can see the breakdown of each document in the corpus based on the frequency of each topic (out of order because it's using the order of apperance in "top_topic")

# In[74]:

Scored_corpus['top_score'].describe()


# In[75]:

Scored_corpus[topic_names].median()


# In[78]:

df.to_csv(r'C:\Users\Bill\Desktop\Tahzoo.com Redesign\GoogleAnalytics\Experiments\token_list.csv')


# In[79]:

Scored_corpus.to_csv('C:\Users\Bill\Desktop\Tahzoo.com Redesign\GoogleAnalytics\Experiments\Scored_corpus.csv')


# Building a solid them model involved running this over and over again. 
# * Add terms to the Bad-words list. 
# * Remove or add tokens. 
# * Summarize the text into a single text token
# 
# Repeat this exorcise until the groups are meaningfull. 

# In[81]:

import csv
w = csv.writer(open("C:\Users\Bill\Desktop\Tahzoo.com Redesign\GoogleAnalytics\Experiments\corpus.csv", "w"))
for key, val in corpus.items():
    w.writerow([key, val])
    


# In[ ]:



