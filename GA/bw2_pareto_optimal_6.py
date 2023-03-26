import numpy as np
import pandas as pd
import csv
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MultilayerGA_v7 import peter_season as ps
import os
#from combination_calculator import Combination_Calculator as cc

class BW_Optimiser():
    def __init__(self, file_name_2):
        #Imported values
        self.num_genes = ps.num_genes
        self.num_chromosones = ps.num_chromosones
        self.num_parents = ps.num_parents
        self.num_offspring = ps.num_offspring
        self.num_data_points = 11
        self.import_data_datapoints = 2
        self.num_materials = 101
        self.crossover_point = ps.crossover_point
        self.population_size = ps.population_size
        self.data_points = ps.data_points
        self.num_generations = ps.num_generations
        self.fitness_penalty = -1000000000000000
        #self.d=ps.d

        #Penalties
        self.fitness_penalty = -1000
        self.max_number = self.fitness_penalty*10
        
        #Optimiser values
        self.best_data = pd.read_csv(file_name_2, delimiter = ',').to_numpy()
        self.freq_1 = 8
        self.freq_interval = 0.1
        self.freq_2 = self.freq_1+self.freq_interval
        self.num_freqs = 85
        self.max_freq = ((self.num_freqs)*self.freq_interval)+self.freq_1
        self.num_x_layers = ps.num_genes-1
        self.percentage_score = 10000
        self.unique_materials_penalty = 10000000000
        self.impedance_penalty = 10000000000
        self.x_pen_2 = 560000000000
        self.epoch_bandwidth = 1
        self.generation_bandwidth = 10
        self.number_of_frequencies = 85
        self.j = complex(0,1)
        self.minimum_frequency = 8.2e9
        self.increment = 50000000
        self.c = 3e10
        self.f = 8e9
        self.pi = np.pi
        self.dataset = 'Dataset 1'
        self.permeability = 1-(1*self.j)
        self.thickness = 0.3
        
        #GA Factors
        self.num_epochs_bw = 100
        self.num_generations_bw = 100
        self.restricted_mode = False

        #Data Acquisition
        self.full_data_array = pd.read_csv(file_name_2, delimiter = ",", encoding = "ISO-8859-1").to_numpy()
        self.optimal_data = np.concatenate((self.full_data_array[:,0].reshape(self.number_of_frequencies,1) , self.full_data_array[:,4].reshape(self.number_of_frequencies,1)), axis = 1)

        #Non-elitism variables
        self.number_from_top = 6
        self.number_from_bottom = int(self.num_parents-self.number_from_top)
        
    def generate_population(self):
        self.population = np.random.choice(self.num_materials-1, size = self.population_size).reshape(self.num_chromosones, self.num_genes)
        
    def select_optimal_values_and_relevant_data(self):
        self.optimal_values_array = (self.optimal_data[:,1][:self.num_freqs])
        self.relevant_data_array = np.zeros((self.num_freqs, self.num_materials, self.import_data_datapoints), dtype = float)
        for i in range(0, self.num_freqs):
            current_freq = (self.minimum_frequency+(self.increment*i))/1e9
            import_data = pd.read_csv(os.getcwd()+f'/GA/{self.dataset}/{current_freq}_data.csv', index_col = 0)
            #import_data = import_data.drop("Filename", 1)
            #import_data = import_data.pop(import_data.columns[1:9])
            #np.delete(import_data, np.s_[0:self.num_materials], axis = 1)
            self.relevant_data_array[i] = import_data
            #print(self.relevant_data_array)
            
    def calculate_fitness_2(self):
        #print(self.relevant_data_array)
        #self.RL_values_array = np.zeros((self.num_freqs, self.num_chromosones, 1), dtype = float)
        self.RL_fitness_array = np.zeros((self.num_freqs, self.num_chromosones, 1), dtype = float)
        self.fitness_penalty_array = np.zeros((self.num_freqs, self.num_chromosones, 1), dtype = float)
        #print(RL_array)
        for f in range(0, self.num_freqs):
            frequency = self.minimum_frequency+(f*self.increment)
            self.wl = self.c/frequency
            #print(self.relevant_data_array[0])
            for j in range(0, self.num_chromosones):
              #Top layer
              self.first_layer = self.population[j][0]
              self.first_layer_e = self.relevant_data_array[f][self.first_layer][0]+(self.j * self.relevant_data_array[f][self.first_layer][1])
              self.first_layer_u = self.permeability
              self.first_layer_d = self.thickness
              self.zm_first_layer = np.sqrt(self.first_layer_u/self.first_layer_e)
              #z_first = self.zm_first_layer*(np.tanh((self.pi*np.sqrt(self.first_layer_e*self.first_layer_u))/((self.c/float(frequency))))*self.first_layer_d) #incorrect
              z_first = self.zm_first_layer *np.tanh((2*np.pi*np.sqrt(self.first_layer_e * self.first_layer_u)*self.first_layer_d*1/(self.c/float(frequency))))
              z_list = [z_first]
              zm_list = [self.zm_first_layer]
              #Calculation of RL
              for i in range(1, self.num_genes):
                  self.layer = self.population[j][i]
                  self.layer_e = self.relevant_data_array[f][self.layer][0]+(self.j*self.relevant_data_array[f][self.layer][1])
                  self.layer_u = self.permeability
                  self.layer_d = self.thickness
                  self.zm_layer = np.sqrt(self.layer_u/self.layer_e)
                  self.z_layer_minus1 = z_list[i-1]
                  self.zm_layer_minus1 = zm_list[i-1]
                  self.z_layer = self.zm_layer *((self.z_layer_minus1+(self.zm_layer*np.tanh((2*np.pi*np.sqrt(self.layer_e*self.layer_u))/((self.c/float(frequency))))*self.layer_d)))/(self.zm_layer+(self.z_layer_minus1*np.tanh(((2*np.pi*np.sqrt(self.layer_e*self.layer_u))/self.wl)*self.layer_d)))
                  #self.z_layer = self.zm_layer *((self.z_layer_minus1+(self.zm_layer*np.tanh((2*np.pi*np.sqrt(self.layer_e*self.layer_u)*self.layer_d)/(self.c/float(frequency))/(self.zm_layer+(self.z_layer_minus1*np.tanh((2*np.pi*np.sqrt(self.layer_e*self.layer_u)*self.layer_d)/(self.c/float(frequency)))))))))               
                  zm_list.append(self.zm_layer)
                  z_list.append(self.z_layer)
              z_array= np.array(z_list)
              zm_array = np.array(zm_list)
              #Final RL Calculation
              RL = 20*np.log(abs((z_array[self.num_genes-1]-1)/(z_array[self.num_genes-1]+1)))
              
              #Impedance Matching
              zm_array_sorted = np.sort(zm_array)
              if (np.array_equal(zm_array, zm_array_sorted)):
                  impedance_matcher = False
              else:
                  impedance_matcher = True
              impedance_penalty = (int(impedance_matcher)*self.fitness_penalty)
              #Unique Materials 
              unique_materials = (len(np.unique(self.population[j])))
              unique_penalty=(int(unique_materials != self.num_genes)*self.fitness_penalty)
              #Better than optimal solution penalty
              optimal_solution = self.optimal_values_array[f]
              better_than_optimal_penalty = (int(RL <= optimal_solution)*self.fitness_penalty)
              self.fitness_penalty_array[f][j] = unique_penalty+better_than_optimal_penalty#+impedance_penalty
              self.RL_fitness_array[f][j] = RL/optimal_solution
        #self.RL_values_array = self.RL_values_array + self.fitness_penalty_array
        self.RL_fitness_array = self.RL_fitness_array + self.fitness_penalty_array
        self.compiled_fitness_array = np.sum(self.RL_fitness_array, axis = 0)
        #print(self.compiled_fitness_array)
        
    def select_mating_pool(self):
        self.parents = np.empty((self.num_parents, self.num_genes), dtype = np.object)
        for i in range(self.num_parents):
            self.max_fitness_idx = np.where(self.compiled_fitness_array==np.amax(self.compiled_fitness_array))
            self.max_fitness_idx=self.max_fitness_idx[0][0]
            self.parents[i, :] = self.population[self.max_fitness_idx, :]
            self.compiled_fitness_array[self.max_fitness_idx] = (self.fitness_penalty-1000000000)
            
    def select_mating_pool_non_elite(self, fitness_array):
       self.parents = np.zeros((self.num_parents, self.num_genes), dtype = int)
       self.ranked_chromosones = np.zeros((self.num_chromosones,self.num_genes), dtype=float)
       for i in range(self.num_chromosones):
            self.max_fitness_idx = np.where(fitness_array==np.amin(fitness_array))
            self.max_fitness_idx=self.max_fitness_idx[0][0]
            self.ranked_chromosones[i, :] = self.population[self.max_fitness_idx, :]
            fitness_array[self.max_fitness_idx] = self.fitness_penalty
       self.samples_from_top = random.sample(range(0, self.num_parents), self.number_from_top)
       self.samples_from_bottom = random.sample(range(self.num_parents,self.num_chromosones), self.number_from_bottom)
       self.samples = np.concatenate((self.samples_from_top, self.samples_from_bottom), axis = 0)
       for i in range(0, self.num_parents):
           self.parents[i] = self.ranked_chromosones[self.samples[i]]       
        
    def crossover(self):
        self.offspring = np.empty((self.num_offspring, self.num_genes), dtype = np.object)
        for i in range(self.num_offspring):
            self.parent_1_idx = i%self.num_parents
            self.parent_2_idx = (i+1)%self.num_parents
            self.offspring[i, 0:self.crossover_point] = self.parents[self.parent_1_idx, 0:self.crossover_point]
            self.offspring[i, self.crossover_point:] = self.parents[self.parent_2_idx, self.crossover_point:]

    def mutation(self):
        for i in range(10):
          self.random_gene = random.randint(0, self.num_genes-1)
          self.random_material = random.randint(0, self.num_materials-1)
          self.mutant_material = random.randint(0, self.num_materials-1)
          self.offspring[i][self.random_gene] = self.mutant_material
          
    def stacker(self):
        self.population[0:self.num_parents, :] = self.parents
        self.population[self.num_parents:, :] =  self.offspring
        
    def BW_GA(self):
        self.complete_population = np.zeros((self.num_epochs_bw, self.num_generations_bw, self.num_chromosones, self.num_genes), dtype = int)
        self.complete_RL_values_array = np.zeros((self.num_epochs_bw, self.num_generations_bw, self.num_freqs, self.num_chromosones, 1), dtype = float)
        self.best_combination_array = np.zeros((self.num_epochs_bw, self.num_genes), dtype = int)
        self.best_RL_values = np.zeros((self.num_epochs_bw, self.num_freqs), dtype = float)
        for j in range(self.num_epochs_bw):
              print("Epoch " +str(j+1))
              #best_combinations = []
              self.generate_population()
              self.select_optimal_values_and_relevant_data()
              for i in range(self.num_generations_bw):
                print("Generation " +str(i+1))
                self.calculate_fitness_2()
                self.complete_RL_values_array[j][i] = self.RL_fitness_array
                self.complete_population[j][i] = self.population
                self.select_mating_pool()
                self.crossover()
                self.mutation()
                self.stacker()
              self.best_combination_array[j] = self.population[0]
        print(self.best_combination_array)
        df = pd.DataFrame(self.best_combination_array)
        df.to_csv(os.getcwd()+"/Multilayer Database/Best Data/Best_combinations.csv", index = None)
        #cc.calculator(os.getcwd()+"/Multilayer Database/Best Data/Best_combinations.csv")
          
bandwidth_optimiser = BW_Optimiser(os.getcwd()+f"/Multilayer Database/Best Data/Best_Data_GA.csv")
bandwidth_optimiser.BW_GA()


