
# coding: utf-8

# In[6]:


#Importing Libraries
import nltk
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(5)


#Read Wiki_plot and IMDb plot
movies_wi = pd.read_csv('Desktop/movies.csv')

#Printing the number of movies loaded
print("Total number of movies loaded: %s" % (len(movies_wi)))

#Let's display the data
movies_wi.head()


# In[7]:


#Let's combine the wiki_plot and the imdb_plot into one column called plot
movies_wi['plot'] = movies_wi['wiki_plot'].astype(str) + "\n" + movies_wi['imdb_plot'].astype(str)

#Check out the new DataFrame
movies_wi.head()


# In[11]:


# #tokenizing a paragraph into sentences and storing in token_sent
# token_sent = [sent for sent in nltk.sent_tokenize("""Today (May 19, 2016) is the only daughter's wedding. Vito Corleone is the Godfather.""")]

# # tokenizing word in first sentence from token_sent, and saving as token_word
# token_word = [word for word in nltk.word_tokenize(token_sent[0])]

# #Removing tokens that do not contain any letters from token_word
import re #importing re; regular expression
# refined = [word for word in token_word if re.search('[a-zA-Z]', word)]

# #View refined words to observe words after tokenization
# print(refined)


# In[12]:


#Importing the SnowballStemmer to perform stemming
from nltk.stem.snowball import SnowballStemmer
iny_stemmer = SnowballStemmer("english")

#Create a function to perform both stemming and tokenization
def tokenize_stem(your_text):
    #we tokenize by sentence first and then by work
    new_token = [y for x in nltk.sent_tokenize(your_text) for y in nltk.word_tokenize(x)]
    
    # Refine token to remove unnecessary characters
    refined_token = [token for token in new_token if re.search('[a-zA-Z]', token)]
    
    #Stem the refined token
    stem_refined = [iny_stemmer.stem(word) for word in refined_token]
    
    return stem_refined

#Now let's invoke our tokenize_stem function
example = tokenize_stem("Today (May 19, 2016) is his only daughter's wedding.")
print(example)


# In[13]:


#Import TfidfVectorizer to create TF-IDF vectors 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, stop_words='english', use_idf=True, tokenizer=tokenize_stem, ngram_range=(1,3))


# In[14]:


#Applying fit and transform of tfidf_vectorizer with the plot of each movie
#Creating a vector representation of the plot summaries

tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in movies_wi["plot"]])
print(tfidf_matrix.shape)


# In[17]:


# To perform clustering, we import k-means
from sklearn.cluster import KMeans

#Create a KMeans object with 5 clusters and save as km
mykm = KMeans(n_clusters = 5)

#Fit the k-means object with tfidf_matrix
mykm.fit(tfidf_matrix)

our_cluster = mykm.labels_.tolist()

#Create a column cluster to represent the generated cluster for each movie
movies_wi["cluster"] = our_cluster

#Display number of films per cluster 
movies_wi["cluster"].value_counts()


# In[18]:


#Importing cosine_similarity to calculate how similar the movie plots are

from sklearn.metrics.pairwise import cosine_similarity

#Calculate the similarity distance
sim_dist = 1 - cosine_similarity(tfidf_matrix)


# In[20]:


#Importing matplotlib.pyplot to plot graphs
import matplotlib.pyplot as plt

#Set matplotlib to display the output inline
get_ipython().magic('matplotlib inline')

#Importing libraries needed to plot dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram

#Creating mergings matrix
merge_mat = linkage(sim_dist, method = 'complete')

#Plot the dendrogram, using the movie title as label column
dendro = dendrogram(merge_mat, orientation="left", labels = [x for x in movies_wi["title"]], leaf_font_size=27)

#Adjsuting the plot
fig = plt.gcf()
p = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(80, 80)

#Now let's view the plotted dendrogram
plt.show()
