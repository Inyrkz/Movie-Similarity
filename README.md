# TASK 5: Movie-Similarity
Find Movie Similarity from Plot Summaries using Kmeans and plotting Dendrograms.
# What is Kmean?

Kmean algorithm is an unsupervised learning algorithm that helps to group similar clusters together in a data.
This task shows how to use machine learning to cluster movie plot based on similarity between the 'wiki plot' in the dataset and the 'imdb plot' in the dataset

METHOD
1. we imported the necessary libraries used for Data Analysis
2. we read in the movie dataframe by using read.xlxs method.
3. we checked the info and describe method on the data.
4. we performed some explanatory data analysis (EDAS)
# Tokenization

Tokenization is a way to split text into tokens. These tokens could be paragraphs, sentences, or individual words. NLTK provides a number of tokenizers in the tokenize module.
5. We then Created a function to perform both stemming and tokenization
6. We imported the Kmeans module from scikitlearn to perform the clustering
7. We performed cosine similarity and then obtained the similarity distance 
7. We plotted the dendrogram from which we obtained the different clusters of movies
8. We printed the movies that were similar


