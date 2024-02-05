from Breeder import Breeder


def run_breeder():
    breeder = Breeder(population_size=10, genome_size=8)

    GENERATIONS = 5
    for generation in range(GENERATIONS):
        print(f"### GENERATION {generation}")
        breeder.run()

    # Print the best genome
    print(breeder.best_genome)


def run_trainer():
    from Trainer import Trainer
    trainer = Trainer("lander4", [1, 1, 1, 1, 1, 1, 1, 1], True)
    trainer.train(1000)

run_trainer()