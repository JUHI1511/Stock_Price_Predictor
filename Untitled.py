#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

# imported the file which contains top 25 headlines, stock went up or down(label) and date
data = pd.read_csv('Combined_News_DJIA.csv')
data.head()


# In[2]:


# check for the columns & rows 
data.shape


# In[3]:


plt.style.use('classic')

sns.set()

ax = sns.countplot(x='Label', hue='Label', data=data)


# In[4]:


data.isnull().sum()


# In[5]:


# filling the null values with median 

data['Top23'].fillna(data['Top23'].median,inplace=True)
data['Top24'].fillna(data['Top24'].median,inplace=True)
data['Top25'].fillna(data['Top25'].median,inplace=True)


# In[6]:


# check for NaN values
data.isnull().sum()


# In[7]:


# create_df function convert top 25 news columns into one column of headlines on each day. 
def create_df(dataset):
    
    dataset = dataset.drop(columns=['Date', 'Label'])
    dataset.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
    for col in dataset.columns:
        dataset[col] = dataset[col].str.lower()
        
    headlines = []
    for row in range(0, len(dataset.index)):
        headlines.append(' '.join(str(x) for x in dataset.iloc[row, 0:25]))
        
    df = pd.DataFrame(headlines, columns=['headlines'])
    df['label'] = data.Label
    df['date'] = data.Date
    
    return df


# In[ ]:





# In[8]:


df = create_df(data)
df.head()


# In[9]:


X = df.headlines


# In[10]:


X


# In[11]:


# tokenize into list of words
def tokenize(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


# In[12]:


# example for tokenize test  
# example for tokenize test 
for message in X[:1]:
    print(message)
    print(tokenize(message))


# In[13]:


df.head()


# In[14]:


train = df[df['date'] < '20150101']
test = df[df['date'] > '20141231']


# In[15]:


x_train = train.headlines
y_train = train.label
x_test = test.headlines
y_test = test.label


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[17]:


import pickle
vectorizer_x = TfidfVectorizer()
X_train = vectorizer_x.fit_transform(x_train).toarray()
X_test = vectorizer_x.transform(x_test).toarray()


# In[22]:


pickle.dump( vectorizer_x, open('tranform.pkl', 'wb'))


# In[18]:


LDA = LinearDiscriminantAnalysis()
X_train_new = LDA.fit_transform(X_train,y_train)
X_test_new = LDA.transform(X_test)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier(n_estimators=136,criterion='entropy',random_state=1)
rc.fit(X_train_new,y_train)


# In[20]:


from sklearn.metrics import confusion_matrix,accuracy_score
y_rfc=rc.predict(X_test_new)
print(confusion_matrix(y_test,y_rfc))
print(accuracy_score(y_test,y_rfc))


# In[21]:


filename = 'nlp_model.pkl'
pickle.dump(rc, open(filename, 'wb'))


# In[ ]:




