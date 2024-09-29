
import numpy as np
from nltk.corpus import brown
import pandas as pd
import nltk
from viterbi import viterbi

nltk.download('brown')
nltk.download('universal_tagset')

#The first 10,000 tagged sentences with the universal tagset is taken from the Brown corpus
universal_tagged_sentences = brown.tagged_sents(tagset='universal')[:10000]

# Extracting all the tags and words, and creating the mappings
tags = set()
words = set()
for sent in universal_tagged_sentences:
    for word, tag in sent:
        tags.add(tag)
        words.add(word.lower())

tags = sorted(tags)
words = sorted(words)
words.append('UNK')

# Mappings between tags/words and indices
tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
word_to_idx = {word: idx for idx, word in enumerate(words)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

num_tags = len(tags)
num_words = len(words)

# Initializing the counts with add-1 smoothing
initial_counts = np.ones(num_tags)
transition_counts = np.ones((num_tags, num_tags))
emission_counts = np.ones((num_tags, num_words))

# Counting the occurrences
for sent in universal_tagged_sentences:
    prev_tag = None
    for idx, (word, tag) in enumerate(sent):
        word = word.lower()
        tag_idx = tag_to_idx[tag]
        word_idx = word_to_idx.get(word, word_to_idx['UNK'])

        if idx == 0:
            initial_counts[tag_idx] += 1
        else:
            prev_tag_idx = tag_to_idx[prev_tag]
            transition_counts[prev_tag_idx, tag_idx] += 1

        emission_counts[tag_idx, word_idx] += 1
        prev_tag = tag

#Probabilities
pi = initial_counts / initial_counts.sum()
A = transition_counts / transition_counts.sum(axis=1, keepdims=True)
B = emission_counts / emission_counts.sum(axis=1, keepdims=True)

if __name__ == '__main__':
    
    # Testing sentences from 10150, 10151 and 10152
    test_sents = brown.tagged_sents(tagset='universal')[10150:10153]

    for sent in test_sents:
        words_in_sent = [word.lower() for word, _ in sent]
        true_tags = [tag for _, tag in sent]

        # Mapping the words to indices and replacing the unknown words with 'UNK'
        obs_seq = [word_to_idx.get(word, word_to_idx['UNK']) for word in words_in_sent]
        true_tag_indices = [tag_to_idx[tag] for tag in true_tags]

        # Running Viterbi algorithm (given in viterbi.py)
        pred_tag_indices, _ = viterbi(obs_seq, pi, A, B)
        pred_tags = [idx_to_tag[idx] for idx in pred_tag_indices]

        # Output results
        print("\nSentence:")
        for word, true_tag, pred_tag in zip(words_in_sent, true_tags, pred_tags):
            print(f"{word:10} True Tag: {true_tag:5} Predicted Tag: {pred_tag:5}")

        # Accuracy for each sentence
        correct = sum(1 for true, pred in zip(true_tag_indices, pred_tag_indices) if true == pred)
        total = len(true_tag_indices)
        accuracy = correct / total
        print(f"\nCorrect: {correct}")
        print(f"\nTotal: {total}")
        print(f"\nAccuracy: {accuracy:.2f}")