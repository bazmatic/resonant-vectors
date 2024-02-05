from Breeder import Breeder

breeder = Breeder(population_size=10, genome_size=8)

GENERATIONS = 5
for generation in range(GENERATIONS):
    print(f"### GENERATION {generation}")
    breeder.run()

# Print the best genome
print(breeder.best_genome)



