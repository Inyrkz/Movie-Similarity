# TASK 5: Movie-Similarity
Find Movie Similarity from Plot Summaries using Kmeans and plotting Dendrograms.
# What is Kmean?

Kmean algorithm is an unsupervised learning algorithm that helps to group similar clusters together in a data.
This task shows how to use machine learning to cluster movie plot based on similarity between the 'wiki plot' in the dataset and the 'imdb plot' in the dataset
METHOD
we imported the necessary libraries used for Data Analysis
we read in the movie dataframe by using read.xlxs method.
we checked the info and describe method on the data.

we then moved on to create explanatory data analysis (EDAS)
# Tokenization
Tokenization is a way to split text into tokens. These tokens could be paragraphs, sentences, or individual words. NLTK provides a number of tokenizers in the tokenize module.
	# We then Created a function to perform both stemming and tokenization
at the end we printed the movies that were similar
