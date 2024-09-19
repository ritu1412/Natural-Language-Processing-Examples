import numpy as np

# Load unigram frequencies from the file count_1w.txt
def load_unigrams(file_path):
    word_freqs = {}
    total_count = 0
    with open(file_path, 'r') as f:
        next(f)
        for line in f:
            word, count = line.strip().split()
            word_freqs[word] = int(count)
            total_count += int(count)
    return word_freqs, total_count

# Load bigram frequencies from the file bigrams.csv
def load_bigrams(file_path):
    bigram_probs = {}
    with open(file_path, 'r') as f:
        next(f)
        for line in f:
            bigram, count = line.strip().split(',')
            bigram_probs[bigram] = int(count)
    return bigram_probs

# Load substitution probabilities from the file substitutions.csv
def load_substitutions(file_path):
    substitution_probs = {}
    with open(file_path, 'r') as f:
        next(f)
        for line in f:
            original, substituted, count = line.strip().split(',')
            substitution_probs[(original, substituted)] = int(count)
    return substitution_probs

# Load deletion probabilities from the file deletions.csv
def load_deletions(file_path):
    deletion_probs = {}
    with open(file_path, 'r') as f:
        next(f)
        for line in f:
            prefix, deleted, count = line.strip().split(',')
            deletion_probs[(prefix, deleted)] = int(count)
    return deletion_probs

# Load insertion probabilities from the file additions.csv
def load_insertions(file_path):
    insertion_probs = {}
    with open(file_path, 'r') as f:
        next(f)
        for line in f:
            prefix, added, count = line.strip().split(',')
            insertion_probs[(prefix, added)] = int(count)
    return insertion_probs

# Load unigram letter probabilities from the file unigrams.csv
def load_unigram_letters(file_path):
    unigram_letter_probs = {}
    with open(file_path, 'r') as f:
        next(f) 
        for line in f:
            letter, count = line.strip().split(',')
            unigram_letter_probs[letter] = int(count)
    return unigram_letter_probs

def weighted_levenshtein_distance(original, candidate, deletion_probs, insertion_probs, substitution_probs, bigram_probs, unigram_letter_probs):
    n, m = len(original), len(candidate)
    dp = np.zeros((n+1, m+1))
    
    # Initialize the dp table for deletions and insertions
    for i in range(1, n+1):
        dp[i][0] = dp[i-1][0] + deletion_probs.get((original[i-1], ""), 1) / unigram_letter_probs.get(original[i-1], 1)
    for j in range(1, m+1):
        dp[0][j] = dp[0][j-1] + insertion_probs.get(("", candidate[j-1]), 1) / unigram_letter_probs.get(candidate[j-1], 1)
    
    # Fill the dp table with weighted distances
    for i in range(1, n+1):
        for j in range(1, m+1):
            # Cost for substitution
            cost = 0
            if original[i-1] != candidate[j-1]:
                cost = substitution_probs.get((original[i-1], candidate[j-1]), 1) / unigram_letter_probs.get(original[i-1], 1)
            
            # Cost for insertion
            insertion_cost = insertion_probs.get((original[:i], candidate[j-1]), 1) / unigram_letter_probs.get(candidate[j-1], 1)

            dp[i][j] = min(
                dp[i-1][j] + deletion_probs.get((original[i-1], ""), 1) / unigram_letter_probs.get(original[i-1], 1),  # Deletion
                dp[i][j-1] + insertion_cost,  # Insertion
                dp[i-1][j-1] + cost  # Substitution
            )

    return dp[n][m]  # Return the weighted distance

def generate_candidates(word):
    """ Generate all possible candidate corrections with an edit distance of exactly 1. """
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:] for L, R in splits if R]
    replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts    = [L + c + R for L, R in splits for c in letters]
    
    return set(deletes + replaces + inserts)

# Probability of x given w
def P_x_given_w(x, w, deletion_probs, insertion_probs, substitution_probs, bigram_probs,unigram_letter_probs):
    return weighted_levenshtein_distance(x, w, deletion_probs, insertion_probs, substitution_probs, bigram_probs,unigram_letter_probs)

# Correct the spelling of the original word
def correct(original, word_probs, deletion_probs, insertion_probs, substitution_probs, bigram_probs,unigram_letter_probs):
    # Generate possible candidates (edit distance = 1)
    candidates = generate_candidates(original)
    
    # Return the best candidate based on P(w|x) = P(x|w) * P(w)
    return max(candidates, key=lambda w: P_x_given_w(original, w, deletion_probs, insertion_probs, substitution_probs, bigram_probs,unigram_letter_probs) * word_probs.get(w, 0))

if __name__ == "__main__":
    # Loading the necessary data
    word_freqs, total_count = load_unigrams('data/count_1w.txt')
    bigram_probs = load_bigrams('data/bigrams.csv')
    substitution_probs = load_substitutions('data/substitutions.csv')
    deletion_probs = load_deletions('data/deletions.csv')
    insertion_probs = load_insertions('data/additions.csv')
    unigram_letter_probs = load_unigram_letters('data/unigrams.csv')

    # Calculating unigram probabilities
    word_probs = {word: freq / total_count for word, freq in word_freqs.items()}

    Examples = ["thre", "writting", "langauge", "scientifc", "recieve","acress","hapy","grate","helo world","teh","recieev"]
    for word in Examples:
        corrected = correct(word, word_probs, deletion_probs, insertion_probs, substitution_probs, bigram_probs, unigram_letter_probs)
        print(f"Original: {word}, Corrected: {corrected}")