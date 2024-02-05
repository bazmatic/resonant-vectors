import numpy as np
from Trainer import Trainer
from settings import NOISE

class Breeder:
    def __init__(self, population_size: int, genome_size: int):
        self.population = []
        self.population_size = population_size
        self.genome_size = genome_size
        # Initialise the population
        for i in range(self.population_size):
            genome = (np.random.random(self.genome_size) - 0.5) * 0.1
            self.population.append(genome)

    def run(self):
        # For each individual in the population, calculate the fitness by running a Trainer
        fitness = []
        id = 1
        for genome in self.population:
            print ("Training individual " + str(id) + ": " + str(genome))
            name = "lander_" + str(id)
            trainer = Trainer(name, genome, True)
            individual_fitness = trainer.train()
            fitness.append(individual_fitness)
            id += 1
        self.population = self.breed(self.population, fitness)
 
    def breed(self, population, fitness):
        # Sort the population by fitness
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]
        
        # Store the best genome
        self.best_genome = sorted_population[-1]
        
        # Select the top 50% of the population
        top_population = sorted_population[:len(population)//2]
        # Breed the top 50% of the population in pairs using the shuffle method
        offsprint_required = len(population) - len(top_population)
        offspring = []
        for i in range(offsprint_required):
            parent1 = top_population[i % len(top_population)]
            parent2 = top_population[(i + 1) % len(top_population)]
            offspring.append(self.shuffle(parent1, parent2))

        # Mutate the offspring
        for i in range(len(offspring)):
            offspring[i] = self.mutate(offspring[i])
        # Replace the bottom 50% of the population with the offspring
        new_population = sorted_population[:len(population)//2] + offspring
        return new_population
    
    def crossover(self, parent1, parent2):
        # Single point crossover
        crossover_point = np.random.randint(0, len(parent1))
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def merge(self, parent1, parent2):
        # Merge the genomes of the parents
        return (parent1 + parent2) / 2
    
    def shuffle(self, parent1, parent2):
        # Shuffle the genomes of the parents. For each gene, randomly select from either parent
        child = np.zeros(self.genome_size)
        for i in range(self.genome_size):
            if np.random.random() > 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child
       
    
    def mutate(self, genome):
        # Add noise to the genome
        # return genome
        MUTATION_RATE = 0.001
        return genome + (np.random.random(self.genome_size) - 0.5) * 2 * MUTATION_RATE
     
            
            
            