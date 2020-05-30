''' IMPORTING THE DATASET '''

# Importing the modules
import numpy as np
import pandas as pd
import nltk

# Reading in IMDb and Wikipedia movie data from the csv file into a pandas dataframe
movies_df = pd.read_csv('datasets/movies.csv')
print("Number of movies loaded: " , (len(movies_df)))

# Displaying the first 5 entries in the dataframe
movies_df.head()

# Combining the wiki_plot and imdb_plot into a single column for better processing
movies_df['plot'] = movies_df['wiki_plot'].astype(str) + "\n" + movies_df['imdb_plot'].astype(str)

# Displaying the first 5 entries in the new dataframe
movies_df.head()


''' METHOD 1 FOR PREPROCESSING '''

''' tokenization '''
# Importing the modules for the tokenization step
from nltk import sent_tokenize, word_tokenize

# Tokenizing by sentence, then by word 
tokens = [ nltk.word_tokenize(text) for text in movies_df['plot'] ]
# Resulting is a list containing tokenized words for each movie plot


''' filtering stopwords and filtering punctuation, digits '''
# Importing the stopwords for english language 
from nltk.corpus import stopwords

# Importing Regular Expression module for filtering out punctuation
import re

filtered_tokens = []

for token in tokens:
    filtered_tokens.append([word for word in token if re.search('[a-zA-Z]', word) and word not in stopwords.words('english')])   

''' stemmimg the words to their roots '''
# Importing the SnowballStemmer to perform stemming
from nltk.stem.snowball import SnowballStemmer

# Creating an English language SnowballStemmer object
stemmer = SnowballStemmer("english")

stems=[]

# Stemming the words to their roots
for token in filtered_tokens:
    stems.append([stemmer.stem(word) for word in token])

''' METHOD 2 FOR PREPROCESSING : combining it all for faster processing '''

# Defining a function to perform both stemming and tokenization
def tokenize_and_stem(text):
    
    # Tokenizing by sentence, then by word
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    # Filtering out raw tokens to remove noise
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    
    # Stemming the filtered tokens
    stems = [stemmer.stem(token) for token in filtered_tokens]
    
    return stems

''' USING TF-IDF VECTORIZATION FOR TRANSFORMING TEXT TO A VECTOR REPRESENTATION FOR IT TO BE PROCESSED BY THE ALGORITHM ''' 
# Importing TfidfVectorizer to create TF-IDF vectors
from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiating the TfidfVectorizer object with stopwords and tokenizer ensuring parameters for efficient processing of text
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, 
                                max_features=200000,
                                min_df=0.2, 
                                stop_words='english',
                                use_idf=True, 
                                tokenizer=tokenize_and_stem,
                                ngram_range=(1,3))


# Fitting and transforming the tfidf_vectorizer with the "plot" of each movie

# This is done to create a vector representation of the plot summaries
tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in movies_df["plot"]])

print(tfidf_matrix.shape)


''' CALCULATING SIMILARITY OF TEXT USING COSINE SIMILARITY '''
# Importing cosine_similarity to calculate similarity of movie plots
from sklearn.metrics.pairwise import cosine_similarity

# Calculating the similarity distance
similarity_distance = 1 - cosine_similarity(tfidf_matrix)

''' IMPORTING MATPLOTLIB FOR DISPLAYING DENDROGRAM '''
# Importing matplotlib.pyplot for plotting graphs
import matplotlib.pyplot as plt


''' USING COMPLETE LINKAGE METHOD FOR HIERARCHICAL CLUSTERING '''
# Importing modules necessary to plot dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram

# Create mergings matrix 
mergings = linkage(similarity_distance, method='complete')

''' PLOTTING THE DENDROGRAM '''
# Plot the dendrogram, using title as label column
dendrogram_ = dendrogram(mergings,
            labels=[x for x in movies_df["title"]],
            leaf_rotation=90,
            leaf_font_size=16,
)

# Adjust the plot
fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)

# Show the plotted dendrogram
plt.show()