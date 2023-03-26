import numpy as np
import pandas as pd
import csv
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from MultilayerGA_v7 import peter_season as ps
import os


class Combination_Calculator():
    def __init__(self, file_name):
        self.combination_array = pd.read_csv((file_name)).to_numpy()
        self.f = 8.2e9
        self.increment = 1312500
        self.num_freqs = 1000
        self.j = complex(0,1)
        self.pi = np.pi
        self.c = 3e8
    def calculator(self):
        self.num_combinations = self.combination_array.shape[0]
        self.num_materials = self.combination_array.shape[1]
        RL_array =np.zeros((self.num_combinations, self.num_freqs))
        #print(RL_array)
        
        for x in range(0, self.num_freqs):
          frequency = self.f+(self.increment*x)
          multilayer_database  = pd.read_csv(os.getcwd()+"/Multilayer Database/Multilayer_database_"+str(int(frequency))+"."+str(0) +".csv").drop("Filename", 1).to_numpy()
          
          
          
          for j in range(0, self.num_combinations):
            self.first_layer = self.combination_array[j][0]
            self.first_layer_e = (multilayer_database[self.first_layer][6]-(self.j*multilayer_database[self.first_layer][7]))#/sc.epsilon_0
            self.first_layer_u = (multilayer_database[self.first_layer][9]*self.j)#/sc.mu_0
            self.first_layer_d = multilayer_database[self.first_layer][8]
            self.zm_first_layer = np.sqrt(self.first_layer_u/self.first_layer_e)
            
            #z_first = self.zm_first_layer*(np.tanh((self.pi*np.sqrt(self.first_layer_e*self.first_layer_u))/((self.c/float(frequency))))*self.first_layer_d) #incorrect
            z_first = self.zm_first_layer *np.tanh((2*np.pi*np.sqrt(self.first_layer_e * self.first_layer_u)*self.first_layer_d*1/(self.c/float(frequency)))) 
            
            z_list = [z_first]
            zm_list = [self.zm_first_layer]
            
             
            #Calculation of RL   
            for i in range(1, self.num_materials):
                self.layer = self.combination_array[j][i]
                self.layer_e = (multilayer_database[self.layer][6]-(self.j*multilayer_database[self.layer][7]))#/sc.epsilon_0
                self.layer_u = (multilayer_database[self.layer][9]*self.j)#/sc.mu_0
                self.layer_d = multilayer_database[self.layer][8]
                self.zm_layer = np.sqrt(self.layer_u/self.layer_e)
                self.z_layer_minus1 = z_list[i-1]
                self.zm_layer_minus1 = zm_list[i-1]
                self.wl = self.c/frequency
                
                self.z_layer = self.zm_layer *((self.z_layer_minus1+(self.zm_layer*np.tanh((2*np.pi*np.sqrt(self.layer_e*self.layer_u))/((self.c/float(frequency))))*self.layer_d)))/(self.zm_layer+(self.z_layer_minus1*np.tanh(((2*np.pi*np.sqrt(self.layer_e*self.layer_u))/self.wl)*self.layer_d)))
                #self.z_layer = self.zm_layer *((self.z_layer_minus1+(self.zm_layer*np.tanh((2*np.pi*np.sqrt(self.layer_e*self.layer_u)*self.layer_d)/(self.c/float(frequency))/(self.zm_layer+(self.z_layer_minus1*np.tanh((2*np.pi*np.sqrt(self.layer_e*self.layer_u)*self.layer_d)/(self.c/float(frequency)))))))))               
                
                zm_list.append(self.zm_layer)
                z_list.append(self.z_layer)
                
            z_array= np.array(z_list)
            zm_array = np.array(zm_list)

            #Final RL Calculation
            RL = 20*np.log(abs((z_array[self.num_materials-1]-1)/(z_array[self.num_materials-1]+1)))
            RL_array[j][x] = RL
            
        average_absorption=RL_array.mean(axis=1)

        print("RL")
        print(RL_array)
        print("Average")
        print(average_absorption)
        


          

        
file = (os.getcwd()+"/Multilayer Database/Best Data/Best_combinations.csv")
cc = Combination_Calculator(file)
cc.calculator()
        
