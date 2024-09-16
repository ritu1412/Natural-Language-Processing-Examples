import numpy as np
import matplotlib.pyplot as plt

# counts of apple and banana in the word list
word_list = ["apple", "apple", "apple", "apple", "apple", "apple", "banana", "banana", "banana", "banana"]
n_apple = word_list.count("apple")
n_banana = word_list.count("banana")
N = n_apple + n_banana

# Range of p_apple (from 0 to 1)
p_apple_values = np.linspace(0, 1, 100)

# Calculating the log-likelihood for each p_apple
log_likelihoods = n_apple * np.log(p_apple_values) + n_banana * np.log(1 - p_apple_values)

# Find the optimal p_apple (the one that maximizes the log-likelihood)
optimal_p_apple = p_apple_values[np.argmax(log_likelihoods)]

# Plotting the log-likelihood graph
plt.figure(figsize=(10, 6))
plt.plot(p_apple_values, log_likelihoods, label='Log-Likelihood')
plt.xlabel('p_apple')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood as a function of p_apple')
plt.axvline(x=optimal_p_apple, color='red', linestyle='--', label=f'Optimal p_apple={optimal_p_apple:.2f}')
plt.legend()
plt.show()