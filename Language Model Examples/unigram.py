import random

def predict_word():
    return "the"

def predict_unigram():
    return {"the":0.6, "of": 0.3, "dinosaurs": 0.1}

def sample_from_distribution(distribution:dict[str,float],k:int):
    population = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(population, weights,k=k)

if __name__ == "__main__":
    print(predict_unigram())
    print(sample_from_distribution(predict_unigram(),20))