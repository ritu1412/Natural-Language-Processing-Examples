import random

def predict_word():
    return "the"

def predict_bigram(previous_word:str):
    return {"dinosaurs": {"the":0.3, "of": 0.7, "dinosaurs": 0.1},
            "of":{"the":0.7, "of": 0.1, "dinosaurs": 0.2},
            "the":{"the":0.1, "of": 0.3, "dinosaurs": 0.6}}[previous_word]

def sample_from_distribution(distribution:dict[str,float],k:int):
    population = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(population, weights,k=k)

if __name__ == "__main__":
    print(predict_bigram("the"))
    print(sample_from_distribution(predict_bigram("the"),20))