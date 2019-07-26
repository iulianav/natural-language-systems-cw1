import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

from collections import defaultdict
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.text import Text
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


# Check if the user provided the reight nr. of argments.
if ((len(sys.argv) < 5 or len(sys.argv) > 5)):
    print """\
Please provide the following arguments:

1) Name of the .txt file containing the target words to cluster
2) Nr. of clusters
3) Context size k (k words before + after target)
4) Corpus
"""
    sys.exit(0)

# Read CMD arguments
targets_file_name = str(sys.argv[1])
clusters_nr = int(sys.argv[2])
context_size = int(sys.argv[3])
corpus_name = str(sys.argv[4])

# Read target words from given file and add them to a list: targets.
with open(targets_file_name, "r") as f:
    lines = f.readlines()
    targets_only = [line.strip(" \n") for line in lines]

# Read corpus and store all tokens (duplicate too) in a list.
corpus_root = corpus_name
wordlists = PlaintextCorpusReader(corpus_root, '.*')
all_words = list(wordlists.words())

# Returns a generator of duplicat words and a list of the indices where
# they appear in the corpus.
def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)

# Pseudoword disambiguation step: for each target word, randomly
# substitute half of its occurrences in the corpus with its reverse
duplicate_words_with_indexes = list(list_duplicates(all_words))
for el in duplicate_words_with_indexes:
    if el[0] in targets_only:
        for i in el[1]:
            if random.random() < 0.5:
                all_words[i] = el[0][::-1]

targets_and_reverses = []
for target in targets_only:
    targets_and_reverses.append(target)
    targets_and_reverses.append(target[::-1])


# Create vocabulary list (dictionary) consisting of all unique words
# in the corpus.
# Each word in the list corresponds to a feature dimension.

# 1. Uncomment to use stemms for dictionary (content words only or all words).

# Uncomment this for stems of content words in dictionary.
# content_words = set(all_words) - set(stopwords.words('english'))

# Uncomment this for stems of ALL word in dictionary.
# content_words = set(all_words)

# stemmer = PorterStemmer()
# stems = [stemmer.stem(word) for word in content_words]
# dictionary = set(stems)



# Uncomment to use all words for dictionary.
dictionary = set(all_words)



# Uncomment to use just target words for dictionary.
# dictionary = set(all_words) - set(stopwords.words('english'))


corpus_words = Text(all_words)

# Compute collocational features: Bag-of-word features, converted to a vector.
# Use occurence counts as the features in the feature vectors.
# Store the feature vectors in the form of a word-by-word matrix:
# n lines = nr. of target words, N columns = nr. of unique words in vocabulary.
word_by_word_matrix = []
for target in targets_and_reverses:
    # Initialise features (occurence counts) to 0 for this target word.
    features = dict((el, 0) for el in dictionary)

    # Get a list of contexts of a set window size
    # surrounding current target word.
    current = corpus_words.concordance_list(target, width=context_size, lines=100000)

    # Increment features (occurence counts) coresponding to words foun
    # in the window around target word.cd 
    for i in current:
        for j in i[0]:
            # stem = stemmer.stem(j)  # UNCOMMENT TO USE STEMMING!
            # if stem in features:
            if j in features:  # COMMENT OUT WHEN USING STEMMING!
                features[j] += 1  # COMMENT OUT WHEN USING STEMMING!
                # features[stem] += 1  # UNCOMMENT TO USE STEMMING!
        for j in i[2]:
            # stem = stemmer.stem(j)   # UNCOMMENT TO USE STEMMING!
            # if stem in features:
            if j in features:  # COMMENT OUT WHEN USING STEMMING!
                features[j] += 1  # COMMENT OUT WHEN USING STEMMING!
                # features[stem] += 1  # UNCOMMENT TO USE STEMMING!

    # Add feature vector for current target word as row
    # in the word-by-word matrix.
    word_by_word_matrix.append([features[el] for el in dictionary])


# Use K-means clustering algorithm to cluster the target words
# into a specified nr. of clusters based on their collocational features.
X = np.array(word_by_word_matrix)

kmeans = KMeans(n_clusters=clusters_nr).fit_predict(X)

# Print clustering result.
print("\nCLUSTERING RESULT:\n")
for i in range(0, len(targets_and_reverses)):
    print(targets_and_reverses[i] + " : " + str(kmeans[i]))

count = 0
for i in range(0, len(targets_and_reverses)-1, 2):
    if kmeans[i] == kmeans[i+1]:
        count += 1

print("\nNr. of correctly clustered pairs: " +str(count) + "\n")
print("Total nr. of clusters: " + str(clusters_nr) + "\n")
print("Accuracy: " + str(count/float(clusters_nr)))

# Reduce the dimensionality of the feature vectors to 2 dimensions
# using the t-Distributed Stochastic Neighbor Embedding (t-SNE) technique
# so that we can visualise the clustering results in 2D.
X = TSNE(n_components=2).fit_transform(X)

# Plot a 2D scatter plot of the clusters and tag every data point
# with the actual target word it corresponds to.
plt.scatter(X[:, 0], X[:, 1], c=kmeans)
for i, target in enumerate(targets_and_reverses):
    plt.annotate(target, xy=(X[i, 0], X[i, 1]))
plt.title("Words clusters based on collocational features")
plt.show()
