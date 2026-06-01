import random
from itertools import product

def create_categorical_dataset(n_samples=300, seed=42):
    random.seed(seed)

    categories = [
        [0, 1, 2],  # outlook
        [0, 1, 2],  # temperature
        [0, 1],     # humidity
        [0, 1]      # wind
    ]

    all_combinations = list(product(*categories))

    X = []

    while len(X) < n_samples:
        shuffled = all_combinations[:]
        random.shuffle(shuffled)

        for row in shuffled:
            if len(X) < n_samples:
                X.append(list(row))

    def label_rule(row):
        outlook, temperature, humidity, wind = row

        # Overcast is usually positive
        if outlook == 1:
            return 1

        # Sunny depends mostly on humidity
        if outlook == 0:
            if humidity == 0 and temperature != 2:
                return 1
            else:
                return 0

        # Rainy depends mostly on wind
        if outlook == 2:
            if wind == 0:
                return 1
            else:
                return 0

    y = [label_rule(row) for row in X]

    noise_rate = 0.07

    for i in range(len(y)):
        if random.random() < noise_rate:
            y[i] = 1 - y[i]

    return X, y


if __name__ == "__main__":
    
    X, y = create_categorical_dataset(n_samples=300, seed=42)

    print(len(X))  # 300
    print(len(y))  # 300

    print(X[:10])
    print(y[:10])