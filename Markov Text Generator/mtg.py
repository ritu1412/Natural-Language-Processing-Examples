import random
import nltk
import copy
from collections import defaultdict

def create_n_grams(corpus, n):
    """Create a dictionary of n-grams and their frequencies."""
    n_grams = defaultdict(lambda: defaultdict(int))
    for i in range(len(corpus) - n + 1):
        key = tuple(corpus[i:i + n - 1])
        next_word = corpus[i + n - 1]
        n_grams[key][next_word] += 1
    return n_grams

def deterministic_choice(n_grams, prefix):
    """Choose the next word deterministically."""
    if prefix in n_grams:
        next_word_freq = n_grams[prefix]
        max_freq = max(next_word_freq.values())
        candidates = [word for word, freq in next_word_freq.items() if freq == max_freq]
        return min(candidates)  # Choose the first alphabetically if there's a tie
    return None


def random_choice(n_grams, prefix):
    """Choose the next word randomly based on probability distribution."""
    if prefix in n_grams:
        next_word_freq = n_grams[prefix]
        total_freq = sum(next_word_freq.values())
        words = list(next_word_freq.keys())
        probabilities = [freq / total_freq for freq in next_word_freq.values()]
        return random.choices(words, probabilities)[0]
    return None


def stupid_backoff(n_grams, sentence, corpus, n, randomize=False):
    """Performing stupid backoff using alpha."""
    alpha = 0.4
    k = 0  # Number of times we've backed off
    while n > 1:
        prefix = tuple(sentence[-(n - 1):])
        
        if randomize:
            next_word = random_choice(n_grams, prefix)
        else:
            next_word = deterministic_choice(n_grams, prefix)
        
        if next_word:
            return next_word, alpha ** k  
        
        n -= 1
        k += 1
        n_grams = create_n_grams(corpus, n) 
        prefix = tuple(sentence[-(n - 1):])
    return next_word, alpha ** k


def finish_sentence(sentence, n, corpus, randomize=False):
    """Finishing the sentence using n-grams and stupid backoff"""
    n_grams = create_n_grams(corpus, n)
    temp_sentence = copy.deepcopy(sentence)
    while len(temp_sentence) < 10 and temp_sentence[-1] not in ['.', '!', '?']:
        next_word, _ = stupid_backoff(n_grams, temp_sentence, corpus, n, randomize=randomize)
        temp_sentence.append(next_word)
    
    return temp_sentence

if __name__ == "__main__":
    nltk.download('gutenberg')
    nltk.download('punkt')
    
    # Load and tokenize the corpus
    corpus = nltk.word_tokenize(nltk.corpus.gutenberg.raw('austen-sense.txt').lower())

    # List of different 'n' values
    n_values = [2, 3, 4]

    # Seed sentence
    sentence_1 = ['she', 'was', 'hardly']

    for n in n_values:
        result = finish_sentence(sentence_1, n, corpus, randomize=False)
        print(f'Deterministic sentence_1 n={n}:', ' '.join(result))

    for n in n_values:
        result = finish_sentence(sentence_1, n, corpus, randomize=True)
        print(f'Stochastic sentence_1 n={n}:', ' '.join(result))

    #seed sentence
    sentence_2 = ['the', 'weather']

    for n in n_values:
        result = finish_sentence(sentence_2, n, corpus, randomize=False)
        print(f'Deterministic sentence_2 n={n}:', ' '.join(result))

    for n in n_values:
        result = finish_sentence(sentence_2, n, corpus, randomize=True)
        print(f'Stochastic sentence_2 n={n}:', ' '.join(result))

    




