import random
import numpy as np

#%% package and data
import time
start_time = time.time()

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Read data
data = pd.read_excel('2-Normalized_data.xlsx', sheet_name='2-Normalized_data')

# Split into independent variables and target variable
t = data[['t']]

X = data[['Pr', 'Bu', 'TVFA', 'Ac', 'COD']]

y = data[['CH4']]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state= 42)
#%% ANN Model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', 
              optimizer='adam', 
              metrics=['mse', 'mae'])

# Fit the model on the train set
EPOCH_NUMBER = 1000

history = model.fit(X_train, 
                    y_train, 
                    epochs= EPOCH_NUMBER, 
                    batch_size= 6, 
                    verbose= 'auto',
                    validation_data=(X_test, y_test))

# Evaluate the model on the test set
y_pred_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

# Evaluate the model on the train set
y_pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)

# Predict on the full dataset
y_pred = model.predict(X)

# save the model
model.save('new_model.h5')
#%% Statistical Analysis Results
# predict on the overall data and calculate metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

#define a dictionary to store the results
results_dict = {'Type of y_predict': ['Train', 'Test', 'Overall'], 
                'MSE': [mse_train, mse_test, mse], 
                'R2': [r2_train, r2_test, r2], 
                'MAE': [mae_train, mae_test, mae]}

# create a dataframe to store the results
Results = pd.DataFrame(results_dict)

# set the index to Type of y_predict
Results = Results.set_index('Type of y_predict')

# print the results
print(Results)
#%% plot
# Set the plot size and font size
plt.rcParams['figure.figsize'] = [25, 12]
plt.rcParams.update({'font.size': 16})

# Extract the metrics for the train and test sets
train_mse = history.history['mse']
test_mse = history.history['val_mse']

# Plot the train and test data MSE VS epoch
plt.semilogy(train_mse, linewidth=2.5, marker='.', markersize=10)
plt.semilogy(test_mse, linewidth=2.5, marker='.', markersize=10)
plt.title('Model MSE', fontsize=20)
plt.ylabel('MSE', fontsize=18)
plt.xlabel('Epoch', fontsize=18)
plt.xticks(range(0, EPOCH_NUMBER+1, 50), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(['Train', 'Test'], loc='upper right', fontsize=18)
plt.grid()

# Set the range of the vertical axis
plt.ylim([0.0001, 1])

# Set the range of the horizontal axis
plt.xlim([0, EPOCH_NUMBER])

plt.show()
#%%
end_time = time.time()
running_time = end_time - start_time
print(f"Running time: {running_time} seconds")


# Define the fitness function.
def fitness(chromosome):
  # Initialize the ANN model.
  model = Sequential()

  # Add layers to the model.
  for layer_index in range(chromosome[0]):
    model.add(Dense(chromosome[1 + layer_index], activation='relu'))

  # Compile the model.
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])

  # Fit the model to the data.
  history = model.fit(X_train, y_train, epochs=EPOCH_NUMBER, batch_size=6, verbose= 'auto', validation_data=(X_test, y_test))

  # Calculate the fitness of the model.
  mse_test = history.history['val_mse'][-1]
  r2_test = history.history['val_r2'][-1]
  mae_test = history.history['val_mae'][-1]
  fitness = -(mse_test + r2_test + mae_test)

  return fitness

# Initialize the population.
POPULATION_SIZE = 100
GENERATIONS = 100
NUM_LAYERS = 3

MUTATION_RATE = 0.01
mutation = []
def crossover(chromosome1, chromosome2):
  # This function performs crossover on two chromosomes.
  # The crossover point is randomly selected.
  # The genes before the crossover point are copied from the first chromosome.
  # The genes after the crossover point are copied from the second chromosome.

  crossover_point = random.randint(0, len(chromosome1) - 1)
  child1 = chromosome1[:crossover_point] + chromosome2[crossover_point:]
  child2 = chromosome2[:crossover_point] + chromosome1[crossover_point:]

  return child1, child2

def genetic_algorithm(POPULATION_SIZE, GENERATIONS, MUTATION_RATE):
    population = []
    for i in range(POPULATION_SIZE):
      chromosome = [random.randint(1, 5) for _ in range(NUM_LAYERS + 1)]
      population.append(chromosome)
    
    # Evaluate the fitness of the population.
    fitnesses = [fitness(chromosome) for chromosome in population]
    # Iterate over the generations.
    for generation in range(GENERATIONS):
      # Select the fittest chromosomes.
      fittest_chromosomes = sorted(population, key=lambda chromosome: fitness(chromosome))[:int(POPULATION_SIZE * 0.2)]
    
      # Create offspring from the fittest chromosomes.
      offspring = []
      for i in range(0, len(fittest_chromosomes), 2):
        parent1, parent2 = fittest_chromosomes[i], fittest_chromosomes[i + 1]
        child1, child2 = crossover(parent1, parent2)
        offspring.append(child1)
        offspring.append(child2)
    
      # Mutate the offspring.
      for chromosome in offspring:
        if random.random() < MUTATION_RATE:
          (chromosome)
    
      # Replace the population with the offspring.
      population = offspring
    
      # Evaluate the fitness of the population.
      fitnesses = [fitness(chromosome) for chromosome in population]

      # Print the best fitness.
      print("Best fitness:", max(fitnesses))
    
      # Return the best chromosome.
      return population[fitnesses.index(max(fitnesses))]

if __name__ == "__main__":
  # Run the genetic algorithm.
  best_chromosome = genetic_algorithm(POPULATION_SIZE, GENERATIONS, MUTATION_RATE)

  # Print the best chromosome.
  print("Best chromosome:", best_chromosome)
