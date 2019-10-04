
# coding: utf-8



#Importing Libraries
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
nltk.download('punkt')
from scipy.cluster.hierarchy import linkage, dendrogram

# Set seed for reproducibility
np.random.seed(5)


#Read Wiki_plot and IMDb plot
movies_wi = pd.read_csv('Desktop/movies.csv', index_col = 0)
#index_col=0 to remove the index column


#Let's display the data
movies_wi.head()


#view the information of the data
movies_wi.info()

#view description of the data
movies_wi.describe()

#Let's combine the wiki_plot and the imdb_plot into one column called plot
movies_wi['plot'] = movies_wi['wiki_plot'].astype(str) + "\n" + movies_wi['imdb_plot'].astype(str)

#Check out the new DataFrame
movies_wi.head()


#create a new column that shows the number of words in each plot
movies_wi["length"] = movies_wi["plot"].apply(len)

#display the top part of the dataset
movies_wi.head()  


#a histogram that shows the distribution of the number of words
movies_wi["length"].plot.hist(bins = 50)

#Description of the length column
movies_wi["length"].describe()


tokenized_sentences = []
for sentence in nltk.sent_tokenize("""Sollozzo kidnaps Hagen to pressure Sonny to accept his deal. 
                                   Michael thwarts a second assassination attempt on his father at the hospital;"""):
    tokenized_sentences.append(sentence)

#tokenize sentence into words
tokenized_words = []
for word in nltk.word_tokenize(tokenized_sentences[0]):
    tokenized_words.append(word)

#remove tokens tha are not in tokenized words
filtered_words = []

for word in tokenized_words:
    if re.search('[a-zA-Z]', word):
        filtered_words.append(word)



#Importing the SnowballStemmer to perform stemming
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

#Create a function to perform both stemming and tokenization
def tokenize_stem(your_text):
    #we tokenize by sentence first and then by work
    new_token = [y for x in nltk.sent_tokenize(your_text) for y in nltk.word_tokenize(x)]
    
    # Refine token to remove unnecessary characters
    refined_token = [token for token in new_token if re.search('[a-zA-Z]', token)]
    
    #Stem the refined token
    stem_refined = [stemmer.stem(word) for word in refined_token]
    
    return stem_refined


#applying the function to the first five rows of the plot column to see
movies_wi['plot'].head(5).apply(tokenize_stem)


# In[24]:


#Import TfidfVectorizer to create TF-IDF vectors 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, stop_words='english', use_idf=True, tokenizer=tokenize_stem, ngram_range=(1,3))


#Applying fit and transform of tfidf_vectorizer with the plot of each movie
#Creating a vector representation of the plot summaries

tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in movies_wi["plot"]])
print(tfidf_matrix.shape)



# To perform clustering, we import k-means
from sklearn.cluster import KMeans

#Create a KMeans object with 5 clusters and save as km
kmeans = KMeans(n_clusters = 5)

#Fit the k-means object with tfidf_matrix
kmeans.fit(tfidf_matrix)

cluster = kmeans.labels_.tolist()

#Create a column cluster to represent the generated cluster for each movie
movies_wi["cluster"] = cluster

#Display number of films per cluster 
movies_wi["cluster"].value_counts()



#Importing cosine_similarity to calculate how similar the movie plots are

from sklearn.metrics.pairwise import cosine_similarity

#Calculate the similarity distance
sim_dist = 1 - cosine_similarity(tfidf_matrix)

#Creating mergings matrix
merged_matrix = linkage(sim_dist, method = 'complete')

#Plot the dendrogram, using the movie title as label column
dendro = dendrogram(merged_matrix, orientation="left", labels = [x for x in movies_wi["title"]], leaf_font_size=27)

#Adjsuting the plot
fig = plt.gcf()
p = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(80, 80)

#Now let's view the plotted dendrogram
plt.show()

#Which movies are most similar?
print('A place in the sun', 'and', "It's a Wonderful life", 'are the most similar movies')

