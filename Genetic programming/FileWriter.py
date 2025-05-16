# AUXILIARY MODULE THAT CREATES, PRINTS, AND SAVES RESULTS FILES
import os

import pandas as pd

folder = "Results/"
folder2 = "Logs/"
if not os.path.isdir(folder):
    os.mkdir(folder)
if not os.path.isdir(os.path.join(folder, folder2)):
    os.mkdir(os.path.join(folder, folder2))


def log_experimental_setup(pop_size, xover_prob, tournament_size, no_of_gens):
    f = open(folder + "Setup.txt", "w")
    f.write("Population size: " + str(pop_size) + "\n")
    f.write("Number of generations: " + str(no_of_gens) + "\n")
    f.write("Crossover probability: " + str(xover_prob) + "\n")
    f.write("Tournament size: " + str(tournament_size) + "\n")
    f.close()


def log_experimental_setup_gp(initial_depth, max_depth, flip_mut_prob, function_set, terminal_set):
    f = open(folder + "Setup.txt", "a")
    f.write("Initial depth: " + str(initial_depth) + "\n")
    f.write("Maximum depth: " + str(max_depth) + "\n")
    f.write("Flip node mutation probability: " + str(flip_mut_prob) + "\n")
    # f.write("Fitness function threshold theta: " + str(theta) + "\n")
    f.write("Function set: " + str(function_set) + "\n")
    f.write("Terminal set: " + str(terminal_set) + "\n")
    f.close()


def create_results_files(class_object):
    # Find and save the attributes of the Individual object.
    attributes = [attribute for attribute in dir(class_object)
                  if not attribute.startswith('__') if not attribute == "model" if not attribute == "signal"]
    f = open(folder + "BestTraining.txt", "w")
    [f.write("\t" + attribute) for attribute in attributes]
    f.write("\n")
    f.close()
    f = open(folder + "BestTest.txt", "w")
    [f.write("\t" + attribute) for attribute in attributes]
    f.write("\n")
    f.close()
    f = open(folder + "BestModel.txt", "w")
    f.close()
    f = open(folder + "SignalsBestTest.txt", "w")
    f.close()


from csv import writer


def output_long(class_object, name,no_of_runs,min_no_of_trades,days,price,MAX_DEPTH,POP_SIZE,XOVER_PROB,TOURNAMENT_SIZE,no_of_gens,threshold,flag):
    allresult = ([name,class_object.total,class_object.no_of_trades, class_object.rate_of_return,class_object.risk,
                 class_object.sharpe_ratio,flag,no_of_runs])
                 #min_no_of_trades,no_of_runs,days,price,MAX_DEPTH,POP_SIZE,XOVER_PROB,TOURNAMENT_SIZE,no_of_gens,days,price,threshold
    with open('./output/test/'+name+'.txt', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(allresult)
        f_object.close()
    #with open('./output/ROR/'+name+'.txt', 'a') as f_object:
    #    writer_object = writer(f_object)
    #    writer_object.writerow([flag,no_of_runs,class_object.trades])
    #    f_object.close()


def save_results_to_file(filename, to_write, current_run):
    f = open(folder + filename, "a")
    f.write("Run" + str(current_run) + "\t")
    for tw in to_write:
        f.write(str(tw) + "\t")
    f.write("\n")


def save_logger(filename, current_run):
    f = open(folder + folder2 + filename + str(current_run) + ".txt", "w")
    f.write("\tBest\tMean\tWorse\n")
    f.close()


def write_to_logger(best_fitness, mean_fitness, worse_fitness, current_generation, current_run):
    f = open(folder + folder2 + "Log" + str(current_run) + ".txt", "a")
    f.write("Generation " + str(current_generation) + "\t" + str(best_fitness) + "\t" + str(mean_fitness) +
            "\t" + str(worse_fitness) + "\n")
    f.close()


# End-of-run results printing and saving into file
def run_printouts(individual, current_run, flag):
    # Find and save the attributes of the Individual object.
    attributes = vars(individual)
    if flag == 0:  # Training set
        print("**********************************************")
        print("Best model: ", individual.model)
        print("Best training fitness ", individual.fitness)
        save_results_to_file("BestTraining.txt",
                             [item for (key, item) in attributes.items() if key != "model" if key != "signal"],
                             current_run)  # We save all Individual object attributes - this is problem-dependent.
    else:
        print("Best model's performance in test set: ", individual.fitness)
        print("**********************************************")
        save_results_to_file("BestTest.txt",
                             [item for (key, item) in attributes.items() if key != "model" if key != "signal"],
                             current_run)
        save_results_to_file("BestModel.txt", individual.model, current_run)


# Generation results printing and saving into files
def generation_printouts(best_fitness, mean_fitness, worse_fitness, current_generation, current_run):
    print("Best fitness in population so far: ", best_fitness)
    print("Average population fitness: ", mean_fitness)
    print("Worse fitness in population so far: ", worse_fitness)
    write_to_logger(best_fitness, mean_fitness, worse_fitness, current_generation, current_run)
