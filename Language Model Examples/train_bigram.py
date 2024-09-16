#Building own vocab of characters from alice in wonderland text

import nltk
import random

alice = nltk.corpus.gutenberg.raw('carroll-alice.txt')

#retriving the words from the text
alice_words = nltk.word_tokenize(alice)

num_tokens = 26
#example size of vocabulary
bigram_counts = [[0 for _ in range(num_tokens)] for i in range(num_tokens)]

for word in alice_words:
    for char_index in range(len(word)-1):
        char1 = ord(word[char_index]) - ord('a')
        char2 = ord(word[char_index+1]) - ord('a')

        if char1 >= 0 and char1 < num_tokens and char2 >= 0 and char2 < num_tokens:
            bigram_counts[char1][char2] += 1

#normalizing the counts
bigram_probs = [[count/sum(counts) for count in counts] for counts in bigram_counts]

#predicting the next character
def predict_bigram(previous_char:str):
    char_index = ord(previous_char) - ord('a')
    if char_index >= 0 and char_index < num_tokens:
        return {chr(i+ord('a')):prob for i,prob in enumerate(bigram_probs[char_index])}
    else:
        return {}

#sampling from the distribution
def sample_from_distribution(distribution:dict[str,float],k:int):
    population = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(population, weights,k=k)

if __name__ == "__main__":
    print(predict_bigram("t"))
    print(sample_from_distribution(predict_bigram("t"),20))
    

