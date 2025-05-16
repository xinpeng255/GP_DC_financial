# GENERIC EVOLUTIONARY ALGORITHM ABSTRACT CLASS. CAN BE USED FOR BOTH GENETIC ALGORITHMS AND GENETIC PROGRAMMING. THE
# FUNCTIONS DEFINED HERE CAN BE USED BY ANY EVOLUTIONARY ALGORITHM. WE ALSO DEFINE METHODS, SUCH AS CROSSOVER AND
# MUTATION, WHICH ARE ALGORITHM-SPECIFIC (I.E. THERE IS A NEED FOR A DIFFERENT IMPLEMENTATION OF THESE METHODS IN A GA
# AND A DIFFERENT IN A GP).
from abc import ABC, abstractmethod
from enum import Enum, auto
from statistics import mean
from random import random, randint
import FileWriter
import math


class Problem(Enum):
    MAXIMISATION = auto()
    MINIMISATION = auto()


class EA(ABC):

    def __init__(self, problem):
        self.problem = problem

    class Individual:
        fitness = 0

        def __init__(self, model):
            self.model = model

    @abstractmethod
    def initialise_population(self):
        pass

    @abstractmethod
    def crossover(self, *args):
        pass

    @abstractmethod
    def mutate(self, *args):
        pass

    @abstractmethod
    def evaluate(self, *args):
        pass

    # Tournament selection
    def tournament(self, TOURNAMENT_SIZE, population):
        tournament_participants = []  # Initialise empty list of random tournament participants.
        for i in range(TOURNAMENT_SIZE):  # select population individuals at random for tournament selection
            random_index = randint(0, len(population)-1)  # Random index to obtain a tournament participant
            tournament_participants.append(population[random_index])  # Place this individual in the list

        tournament_participants.sort(key=lambda x: x.fitness, reverse=False) if self.problem == Problem.MINIMISATION \
            else tournament_participants.sort(key=lambda x: x.fitness, reverse=True)

        return tournament_participants[0].model  # Return the individual with the best fitness

    # Main evolutionary process
    def evolve(self, nGENS, population, tournament_size, xover_prob, current_run, dataset, flag):
        FileWriter.save_logger("Log", current_run)
        for g in range(0, nGENS):
            maxtotal=-100
            maxROR=-100
            maxrisk=-100
            mintotal=100
            minROR=100
            minrisk=100
            for individual in population:
                self.evaluate(individual, dataset, flag)  # Calculate fitness for each  individual in the population
                if individual.total !=0 and individual.rate_of_return!=0 and individual.risk!=0:
                    if individual.total > maxtotal:
                        maxtotal=individual.total
                    if individual.rate_of_return > maxROR:
                        maxROR=individual.rate_of_return
                    if individual.risk > maxrisk:
                        maxrisk=individual.risk
                    if individual.total < mintotal:
                        mintotal=individual.total
                    if individual.rate_of_return < minROR:
                        minROR=individual.rate_of_return
                    if individual.risk < minrisk:
                        minrisk=individual.risk
            for individual in population:
                if (maxtotal-mintotal)==0 or (maxROR-minROR)==0 or (maxrisk-minrisk)==0:
                    individual.fitness=0
                else:
                    if individual.total !=0 and individual.rate_of_return!=0 and individual.risk!=0:
                        individual.fitness=(((individual.total-mintotal)/(maxtotal-mintotal)+1)**(0))*(((individual.rate_of_return-minROR)/(maxROR-minROR)+1)**(0.5))/(((individual.risk-minrisk)/(maxrisk-minrisk)+1)**(0.5))
            print("---------Generation ", g, "-----------")
            # Sort them by fitness value.
            population.sort(key=lambda x: x.fitness, reverse=False) if self.problem == Problem.MINIMISATION \
                else population.sort(key=lambda x: x.fitness, reverse=True)  # Sort population by fitness, best on top
            mean_pop_fitness = mean([p.fitness for p in population])
            FileWriter.generation_printouts(  # Print results in the terminal and save them in the log file
                best_fitness=population[0].fitness, mean_fitness=mean_pop_fitness,
                worse_fitness=population[-1].fitness, current_generation=g, current_run=current_run)

            if g < nGENS - 1:  # Breeding. Evolution happens up to the generation before the last.
                # Elitism; copy the best individual into the next generation
                intermediate_pop = [population[0].model]

                for i in range(1, len(population)):  # Evolutionary process.
                    parent1 = EA.tournament(self, tournament_size, population)
                    r = random()
                    if r < xover_prob:  # Crossover
                        parent2 = self.tournament(tournament_size, population)
                        child = self.crossover(parent1, parent2)
                        intermediate_pop.append(child)
                    else:  # Mutation
                        intermediate_pop.append(self.mutate(parent1))

                for individual, intermediate in zip(population, intermediate_pop):
                    individual.model = intermediate  # Copy the intermediate population into the actual population list
    # Main method to run the GA
    def run(self, population_size: object, xover_prob: object, tournament_size: object, no_of_gens: object, no_of_runs: object, training_dataset: object,
            test_dataset: object,
            class_object: object, name,day,r,min_no_of_trades,MAX_DEPTH,threshold,POP_SIZE,XOVER_PROB,TOURNAMENT_SIZE) -> object:
        FileWriter.log_experimental_setup(population_size, xover_prob, tournament_size, no_of_gens)
        FileWriter.create_results_files(class_object)
        for n in range(no_of_runs):
            print("--------------- Run ", n, " ----------------")
            population = self.initialise_population()
            self.evolve(no_of_gens, population, tournament_size, xover_prob, n, training_dataset, flag=0)
            FileWriter.run_printouts(population[0], n, flag=0)
            FileWriter.output_long(population[0], name,no_of_runs=n,min_no_of_trades=min_no_of_trades,days=day,price=r,MAX_DEPTH=MAX_DEPTH,POP_SIZE=POP_SIZE,XOVER_PROB=XOVER_PROB,TOURNAMENT_SIZE=TOURNAMENT_SIZE,no_of_gens=no_of_gens,threshold=threshold,flag=0)
            output=self.evaluate(population[0], test_dataset, flag=1)
            FileWriter.run_printouts(population[0], n, flag=1)
            FileWriter.output_long(population[0],name,no_of_runs=n,min_no_of_trades=min_no_of_trades,days=day,price=r,MAX_DEPTH=MAX_DEPTH,POP_SIZE=POP_SIZE,XOVER_PROB=XOVER_PROB,TOURNAMENT_SIZE=TOURNAMENT_SIZE,no_of_gens=no_of_gens,threshold=threshold,flag=1)

        return output
