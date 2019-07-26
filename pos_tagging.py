from collections import Counter 
from nltk import pos_tag
from nltk.corpus import PlaintextCorpusReader, inaugural
from nltk.tokenize import word_tokenize

POS_tagging = pos_tag(inaugural.words())

# Compute:
# 1. POS tags counts
POS_tags_counts = {}
# 2. Tag to tag transition counts for all pairs of subsequent POS tags.
tag_transitions_counts = {}
# 3. Word instances counts for all tags.
tags_words_counts = {}

# Loop through the first n-1 POS tagging pairs.
for i, el in enumerate(POS_tagging[:-1]):
    if el[1] not in POS_tags_counts:
        POS_tags_counts[el[1]] = 1
        tag_transitions_counts[el[1]] = {POS_tagging[i+1][1]: 1}
        tags_words_counts[el[1]] = {el[0]: 1}
    else:
        POS_tags_counts[el[1]] += 1
        
        if el[0] in tags_words_counts[el[1]]:
            tags_words_counts[el[1]][el[0]] += 1
        else:
            tags_words_counts[el[1]][el[0]] = 1

        if POS_tagging[i+1][1] in tag_transitions_counts[el[1]]:
            tag_transitions_counts[el[1]][POS_tagging[i+1][1]] += 1
        else:
            tag_transitions_counts[el[1]][POS_tagging[i+1][1]] = 1

# Deal with last POS tagging pair too.
if POS_tagging[-1][1] not in POS_tags_counts:
    POS_tags_counts[POS_tagging[-1][1]] = 1
    tags_words_counts[POS_tagging[-1][1]] = {POS_tagging[-1][0]: 1}
else:
    POS_tags_counts[POS_tagging[-1][1]] += 1
    
    if POS_tagging[-1][0] in tags_words_counts[POS_tagging[-1][1]]:
        tags_words_counts[POS_tagging[-1][1]][POS_tagging[-1][0]] += 1
    else:
        ags_words_counts[POS_tagging[-1][1]][POS_tagging[-1][0]] = 1

# a) Find 5 most frequent tags in corpus. 
k = Counter(POS_tags_counts) 
high = k.most_common(5)  
  
print('Dictionary with 5 highest POS tag frequencies:') 
print('POS tag: number of occurences') 
for i in high: 
    print(i[0] + ' : ' + str(i[1])) 

# b) Disambiguate 2 given POS tagging results using tag transition and word likelihood probabilities.
# 1) Word 'race' is NN.
tag_transition_prob1 = tag_transitions_counts['DT']['NN']/float(POS_tags_counts['DT']) # P(NN|DT)
# 2) Word 'race' is VB.
tag_transition_prob2 = tag_transitions_counts['DT']['VB']/float(POS_tags_counts['DT']) # P(VB|DT)

# First check if the word 'race' appears in the corpus with the tag NN, else the word likelihood prob. is 0.
if 'race' in tags_words_counts['NN']:
    word_likelihood_prob1 = tags_words_counts['NN']['race']/float(POS_tags_counts['NN']) # P(race|NN)
else:
    word_likelihood_prob1 = 0

# First check if the word 'race' appears in the corpus with the tag VB, else the word likelihood prob. is 0.
if 'race' in tags_words_counts['VB']:
    word_likelihood_prob2 = tags_words_counts['VB']['race']/float(POS_tags_counts['VB']) # P(race|VB) = 0
else:
    word_likelihood_prob2 = 0

# DISAMBIGUATION: we choose POS tag with the highest probability for the word 'race'.
# 1: P(race|NN)*P(NN|DT)*P(IN|NN) - prob. that 'race' has POS tag NN.
# 2: P(race|VB)*P(VB|DT)*P(IN|VB) - prob. that 'race' has POS tag VB.

# Probabilities that the POS tag 'IN' of the subsequent word 'for' follows both possible POS tags for the word 'race'.
prob_IN_given_NN = tag_transitions_counts['NN']['IN']/float(POS_tags_counts['NN']) # P(IN|NN)
prob_IN_given_VB = tag_transitions_counts['VB']['IN']/float(POS_tags_counts['VB']) # P(IN|VB)

final_probability1 = word_likelihood_prob1 * tag_transition_prob1 * prob_IN_given_NN
final_probability2 = word_likelihood_prob2 * tag_transition_prob2 * prob_IN_given_VB # = 0

#final_probability1 > final_probability2 => 'race' is NN
print('\nDisambiguation:')
print('Tag sequence probability for sentence 1) : ' + str(final_probability1))
print('Tag sequence probability for sentence 2) : ' + str(final_probability2))

if final_probability1 > final_probability2:
    print('The correct POS tagging result is 1): the correct POS tag for the word "race" is NN.')
else:
    print('The correct POS tagging result is 2): the correct POS tag for the word "race" is VB.')
