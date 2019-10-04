#Importing Libraries
import re
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
nltk.download('punkt')

# Set seed for reproducibility
np.random.seed(5)


#Read Wiki_plot and IMDb plot
MOVIES_WI = pd.read_excel('movies.xlsx', index_col=0)
#index_col=0 to remove the index column


#Let's display the data
MOVIES_WI.head()


#view the information of the data
MOVIES_WI.info()

#view description of the data
MOVIES_WI.describe()

#Let's combine the wiki_plot and the imdb_plot into one column called plot
MOVIES_WI['plot'] = MOVIES_WI['wiki_plot'].astype(str) + "\n" + MOVIES_WI['imdb_plot'].astype(str)

#Check out the new DataFrame
MOVIES_WI.head()


#create a new column that shows the number of words in each plot
MOVIES_WI["length"] = MOVIES_WI["plot"].apply(len)

#display the top part of the dataset
MOVIES_WI.head()  


#a histogram that shows the distribution of the number of words
MOVIES_WI["length"].plot.hist(bins=50)

#Description of the length column
MOVIES_WI["length"].describe()


TOKENIZED_SENT = []
for sentence in nltk.sent_tokenize("""Sollozzo kidnaps Hagen to pressure Sonny to accept his deal. 
                                   Michael thwarts a second assassination attempt on his father at 
                                   the hospital;"""):
    TOKENIZED_SENT.append(sentence)

#tokenize sentence into words
TOKENIZED_WORDS = []
for word in nltk.word_tokenize(TOKENIZED_SENT[0]):
    TOKENIZED_WORDS.append(word)

#remove tokens tha are not in tokenized words
FILTERED_WORDS = []

for word in TOKENIZED_WORDS:
    if re.search('[a-zA-Z]', word):
        FILTERED_WORDS.append(word)





STEMMER = SnowballStemmer("english")

#Create a function to perform both stemming and tokenization
def tokenize_stem(your_text):
    #we tokenize by sentence first and then by work
    new_token = [y for x in nltk.sent_tokenize(your_text) for y in nltk.word_tokenize(x)]
    # Refine token to remove unnecessary characters
    refined_token = [token for token in new_token if re.search('[a-zA-Z]', token)]
    #Stem the refined token
    stem_refined = [STEMMER.stem(word)for word in refined_token]
    return stem_refined


#applying the function to the first five rows of the plot column to see
MOVIES_WI['plot'].head(5).apply(tokenize_stem)




TFIDF_VECTORIZER = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, 
                                   stop_words='english', use_idf=True, 
                                   tokenizer=tokenize_stem, ngram_range=(1, 3))

#Applying fit and transform of tfidf_vectorizer with the plot of each movie
#Creating a vector representation of the plot summaries

TFIDF_MATRIX = TFIDF_VECTORIZER.fit_transform([x for x in MOVIES_WI["plot"]])
print(TFIDF_MATRIX.shape)

#instance of kmeans
KM = KMeans(n_clusters=5)

#Fit the k-means object with tfidf_matrix
KM.fit(TFIDF_MATRIX)

CLUSTER = KM.labels_.tolist()

#Create a column cluster to represent the generated cluster for each movie
MOVIES_WI["cluster"] = CLUSTER

#Display number of films per cluster 
MOVIES_WI["cluster"].value_counts()


#Calculate the similarity distance
SIM_DIST = 1 - cosine_similarity(TFIDF_MATRIX)

#Creating mergings matrix
MERGED_MATRIX = linkage(SIM_DIST, method='complete')


#Plot the dendrogram, using the movie title as label column
DENDRO = dendrogram(MERGED_MATRIX, orientation="left", 
                    labels=[x for x in MOVIES_WI["title"]], leaf_font_size=27)

#Adjsuting the plot
FIG = plt.gcf()
PLOTTING = [label.set_color('r') for label in plt.gca().get_xmajorticklabels()]
FIG.set_size_inches(80, 80)

#Now let's view the plotted dendrogram
plt.show()

#Which movies are most similar?
print('A place in the sun', 'and', "It's a Wonderful life", 'are the most similar movies')