
# coding: utf-8

# # Data loading and preparation

# ### Loading required python libraries

# In[1]:


import pandas as pd


# In[2]:


# Loading and checking the training dataset
df_train = pd.read_csv('./../data/training_data.txt', header=None)
df_train.head()


# In[3]:


# Loading and checking the test dataset
df_test = pd.read_csv('./../data/test_data_v0.txt', header=None)
df_test.head()


# In[4]:


# Remove the unnecessary trailing tabs in test dataset 
test = df_test[0].map(str.strip)

test.head()


# In[5]:


# Convert the train dataset to a pandas series
train = df_train[0]

train.head()


# In[6]:


# Spliting the training dataset into response and predictors
y_train = train.map(lambda x: x.split()[0])
x_train = train.map(lambda x: ' '.join(x.split()[1:]))


# In[7]:


# Spliting the test dataset into response and predictors
y_test = test.map(lambda x: x.split()[0])
x_test = test.map(lambda x: ' '.join(x.split()[1:]))


# ## Q1 Analysis

# ### Loading Required python libraries

# In[8]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np


# In[9]:


# Run this if nltk is not configured before
# =========================================
# nltk.download()


# In[10]:


# Define stop words
stop_words = set(stopwords.words('english'))


# In[11]:


def removeStopWords(x):
    """Return only words that are not in stop_words"""
    return [w for w in x if not w in stop_words]


# In[12]:


def getLemma(x):
    """Return the lemma of each word"""
    return [WordNetLemmatizer().lemmatize(w) for w in x]


# In[13]:


# Tokenize each sentence in the training set, remove stop-words and take the lemma
x = x_train.map(word_tokenize).map(removeStopWords).map(getLemma)


# ### Calculate Word Counts

# In[14]:


# Get Unigram Word Counts
unigram_wcounts = x.groupby(y_train).apply(lambda x: [w for rec in x for w in rec]).map(nltk.FreqDist)
unigram_wcounts = pd.DataFrame(list(unigram_wcounts), index=unigram_wcounts.index)
unigram_wcounts


# In[15]:


# Get Bigram Word Counts
bigram_wcounts = x.groupby(y_train).apply(lambda x: [w for rec in x for w in nltk.bigrams(rec)]).map(nltk.FreqDist)
bigram_wcounts = pd.DataFrame(list(bigram_wcounts), index=bigram_wcounts.index)
bigram_wcounts


# In[16]:


# Get Trigram Word Counts
trigram_wcounts = x.groupby(y_train).apply(lambda x: [w for rec in x for w in nltk.trigrams(rec)]).map(nltk.FreqDist)
trigram_wcounts = pd.DataFrame(list(trigram_wcounts), index=trigram_wcounts.index)
trigram_wcounts


# ### Calculate Total Word Counts

# In[17]:


# Unigram total counts
unigram_total_wcount = unigram_wcounts.sum(axis=1)
unigram_total_wcount


# In[18]:


# Bigram total counts
bigram_total_wcount = bigram_wcounts.sum(axis=1)
bigram_total_wcount


# In[19]:


# Trigram total counts
trigram_total_wcount = trigram_wcounts.sum(axis=1)
trigram_total_wcount


# ### Calculate Probabilities

# In[20]:


unigram_probs = unigram_wcounts.div(unigram_total_wcount, axis=0)
unigram_probs


# In[21]:


bigram_probs = bigram_wcounts.div(bigram_total_wcount, axis=0)
bigram_probs


# In[22]:


trigram_probs = trigram_wcounts.div(trigram_total_wcount, axis=0)
trigram_probs


# ### Predictions

# In[23]:


def getUnigramProb(word):
    try:
        pProb = unigram_probs.loc['put', word]
    except:
        pProb = 0
    try:
        tProb = unigram_probs.loc['take', word]
    except:
        tProb = 0
        
    return {
        'pProb': pProb,
        'tProb': tProb,
    }
getUnigramProb('blue')


# In[24]:


def getBigramProb(word):
    try:
        pProb = bigram_probs[word]['put']
    except:
        pProb = 0
    try:
        tProb = bigram_probs[word]['take']
    except:
        tProb = 0
        
    return {
        'pProb': pProb,
        'tProb': tProb,
    }
getBigramProb(('block', 'blue'))


# In[25]:


def getTrigramProb(word):
    try:
        pProb = trigram_probs[word]['put']
    except:
        pProb = 0
    try:
        tProb = trigram_probs[word]['take']
    except:
        tProb = 0
        
    return {
        'pProb': pProb,
        'tProb': tProb,
    }
getTrigramProb(('block', 'circle', 'circle'))


# In[26]:


# Prepare the test set
x2 = x_test.map(word_tokenize).map(removeStopWords).map(getLemma)
x2


# In[27]:


def predict(sent, predType='uni'):
    pProb = 0
    tProb = 0
    
    for w in sent:
        if predType == 'uni':
            p = getUnigramProb(w)
        elif predType == 'bi':
            p = getBigramProb(w)
        else:
            p = getTrigramProb(w)
        pProb += p['pProb']
        tProb += p['tProb']
    
    res = 'put' if pProb > tProb else 'take'
    
    return {
        'prediction': res,
        'pProb': pProb,
        'tProb': tProb
    }


# In[28]:


unigram_prediction = x2.map(predict)
unigram_prediction


# In[29]:


x2_bigram = x2.map(lambda x: list(nltk.bigrams(x)))
x2_bigram


# In[30]:


bigram_prediction = x2_bigram.map(lambda x: predict(x, 'bi'))
bigram_prediction


# In[31]:


x2_trigram = x2.map(lambda x: list(nltk.trigrams(x)))
x2_trigram


# In[32]:


trigram_prediction = x2_trigram.map(lambda x: predict(x, 'tri'))
trigram_prediction


# ### Analysis of the results

# #### Unigram

# In[33]:


unigram_prediction_comparison = unigram_prediction.map(lambda x: x['prediction']) == y_test
unigram_prediction_comparison


# In[34]:


unigram_test_accuracy = unigram_prediction_comparison.sum()/10
unigram_test_accuracy


# In[35]:


unigram_train_accuracy = (x.map(predict).map(lambda x: x['prediction']) == y_train).sum()/len(x)
unigram_train_accuracy


# #### Bigram

# In[36]:


bigram_prediction_comparison = bigram_prediction.map(lambda x: x['prediction']) == y_test
bigram_prediction_comparison


# In[37]:


bigram_test_accuracy = bigram_prediction_comparison.sum()/10
bigram_test_accuracy


# In[38]:


bigram_train_accuracy = (x.map(lambda x: list(nltk.bigrams(x))).map(lambda x: predict(x, 'bi')).map(lambda x: x['prediction']) == y_train).sum()/len(x)
bigram_train_accuracy


# In[39]:


trigram_prediction_comparison = trigram_prediction.map(lambda x: x['prediction']) == y_test
trigram_prediction_comparison


# In[40]:


trigram_test_accuracy = trigram_prediction_comparison.sum()/10
trigram_test_accuracy


# In[41]:


trigram_train_accuracy = (x.map(lambda x: list(nltk.trigrams(x))).map(lambda x: predict(x, 'tri')).map(lambda x: x['prediction']) == y_train).sum()/len(x)
trigram_train_accuracy

