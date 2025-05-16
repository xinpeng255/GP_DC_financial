#!/usr/bin/python36
import sys
import FileWriter
from EA import Problem, EA
from GP import GP, Util
import numpy as np
import pandas as pd
from random import seed
from statistics import mean, stdev 
#threshold=0.001    

name=sys.argv[3]
seed(111)
transaction_cost=0.00025
riskfree=0.00022
problem = Problem.MAXIMISATION  # Setup the current problem as a maximisation one (i.e. maximise the fitness value).
terminalNodeCrossBias = 0.1  # Probability to select a terminal node during crossover. Value given by Koza (1992).
nRuns = 50  # Individual and independent GP runs
INITIAL_DEPTH = 2  # Initial tree depth during population initialisation
MAX_DEPTH = 6  # Maximum tree depth
POP_SIZE = 500  # Population size
XOVER_PROB = 0.95  # Crossover probability. Mutation probability is (1 - XOVER_PROB)
FLIP_MUT_PROB = 0.5  # Probability of selecting a tree node for mutation during the point mutation operator
TOURNAMENT_SIZE = 2  # Tournament size
nGENS = 50  # Number of generations
ERC_LB = -1  # Lower bound for ERC numbers
ERC_UB = 1  # Upper bound for ERC numbers
TYPE = "Predicate"
current_run = 0  # Tracks the current GP run
parameter_strategy=np.array(pd.read_csv('./output/parameter/solution2_parameter.csv',header=None))
#parameter_min_no_of_trades=np.array(pd.read_csv('./output/parameter_DC.csv',header=None))
#parameter_strategy2=np.array(pd.read_csv('./output/parameter_combined_n_r_total.csv',header=None))
for l in range(len(parameter_strategy[:,1])):
    if parameter_strategy[l,1]==name or parameter_strategy[l,1]==name:               
        THETA_of_price=float(parameter_strategy[l,3])
        n = float(parameter_strategy[l, 2])
        threshold=float(parameter_strategy[l,4])
        break
training_data = np.array(pd.read_csv(sys.argv[1]+str(threshold)+'.csv'))[:,1:]
test_data = np.array(pd.read_csv(sys.argv[2]+str(threshold)+'.csv'))[:,1:]
#for b in range(len(parameter_strategy2[:,1])):
#    if parameter_strategy[l,1]==name or parameter_strategy[l,1]==name:
#         THETA_of_price=float(parameter_strategy[l,8])
        #THETA_of_price=j
#        n = float(parameter_strategy[l,7])
        #threshold=float(parameter_strategy[l, 8])
#        break
#for h in range(len(parameter_min_no_of_trades[:,1])):
#    if parameter_min_no_of_trades[h,1]==name :
#        MIN_NO_OF_TRADES = float(parameter_min_no_of_trades[h,7])
if name[-1]=='0':
    MIN_NO_OF_TRADES=160
else:
    MIN_NO_OF_TRADES=80
#if name[-1]=='0':
#    MIN_NO_OF_TRADES=160
#else:
#    MIN_NO_OF_TRADES=80
F_SET = ["AND", "OR", "GT", "LT"]  # Function Set
# Number of variables  in the terminal set depends on number of features in the excel input file
# ERC: Ephemeral Random Constant; it should always be in the last place of the array
T_SET = ["Var0", "Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7",
         "Var8", "Var9", "Var10", "Var11", "Var12", "Var13", "Var14", "Var15",
         "Var16", "Var17", "Var18", "Var19", "Var20", "Var21", "Var22", "Var23",
         "Var24", "Var25", "Var26", "Var27", "Var28", "Var29", "Var30", "Var31", "Var32", "Var33", "Var34", "Var35", "Var36", "Var37", "Var38", "Var39", "Var40", "Var41"
         , "Var42", "Var43", "Var44", "Var45", "Var46", "Var47", "Var48", "Var49", "Var50", "Var51", "Var52", "Var53", "Var54", "Var55", "ERC"]
grammar = {  # Acts as a lookup type table, finding what are the compatible types and their values
    "Root": ["AND", "OR"],  # Force the root of the trees to be one of these functions
    "AND": F_SET,  # These are the compatible types when the parent node is AND
    "OR": F_SET,  # These are the compatible types when the parent node is OR
    "LT": T_SET,  # These are the compatible types when the parent node is LT
    "GT": T_SET  # These are the compatible types when the parent node is GT
}

types_lookup = {  # Defines the types of each node; the function set is booleans, whereas the terminal set is arithmetic
    "Boolean": F_SET,
    "Arithmetic": T_SET
}


class Trading(GP):
    # Properties of an individual. An individual is a trading strategy represented as a prefix expression.
    class Individual(EA.Individual):  # Extends the EA Individual class
        no_of_trades = 0
        sharpe_ratio = 0
        rate_of_return = 0
        risk = 0  # standard deviation of returns
        signal = []  # array holding the predictions of the individual

    def __init__(self):
        super().__init__(POP_SIZE, INITIAL_DEPTH, MAX_DEPTH, grammar, types_lookup, F_SET, T_SET, terminalNodeCrossBias,
                         FLIP_MUT_PROB, ERC_LB, ERC_UB, TYPE, problem)

    # Fitness evaluation based on trading metrics. Flag is used to denote training or test dataset.
    @staticmethod
    def evaluate(individual, data, flag):
        global current_run

        rate_of_return = []
        total_return=0
        q = 0  # Shares quantity. For now we allow a single quantity only.
        no_of_trades = 0
        price_at_buy = data[0, 0]  # initialise value to the first price in the time series
        signal = Util.interpret(individual.model,
                                data)  # boolean depending on the evaluation of the expression (tree)

        for i in range(len(data)):
            price = data[i, 0]  # Price is Column 0 in the dataset
            if signal[i]:  # If expression output is True, then take an action; otherwise take a different action.
                if q == 0:
                    q += 1  # Open position
                    no_of_trades += 1
                    price_at_buy = price
                    day_at_buy = i
                    if no_of_trades==1:
                        price_first=price
            else:
                # Ensure we have quantity to sell&that the sell price is greater than the purchase price by a threshold.
                if q == 1 and ((price - price_at_buy) / price_at_buy > THETA_of_price or i-day_at_buy>n):
                    q -= 1  # CLose position
                    no_of_trades += 1
                    rate_of_return.append((price*(1-transaction_cost) - price_at_buy*(1+transaction_cost)) / price_at_buy*(1+transaction_cost))
                    total_return +=price*(1-transaction_cost) - price_at_buy*(1+transaction_cost)

        individual.no_of_trades = no_of_trades
        individual.trades=rate_of_return
        if flag == 0:  # Training set calculation
            # Encourage individuals to close open positions (no_of_trades is an even number) & have a min no of trades
            individual.rate_of_return = mean(rate_of_return) \
                if (no_of_trades % 2 == 0 and no_of_trades >= MIN_NO_OF_TRADES ) else -10
            if total_return==0:
                individual.total=0
            else:
                individual.total=total_return/price_first \
                    if (no_of_trades % 2 == 0 and no_of_trades >= MIN_NO_OF_TRADES ) else -10                    
        else:  # Test set calculation
            individual.rate_of_return = mean(
                rate_of_return) if rate_of_return else 0  # Calculate mean if list not empty
            if total_return==0:
                individual.total=0
            else:
                individual.total=total_return/price_first                    
            individual.signal = signal
            FileWriter.save_results_to_file("SignalsBestTest.txt", signal,
                                            current_run)  # Signals from best model in test set
            current_run += 1

        # Standard deviation calculation requires at least 2 values, which are derived from a minimum of 4 trades
        individual.risk = stdev(rate_of_return) if no_of_trades >= 4 else 0
        if (no_of_trades <=MIN_NO_OF_TRADES or no_of_trades % 2 != 0 or individual.total<0 or individual.rate_of_return<0) and flag ==0 : 
            individual.total=-10
            individual.rate_of_return=-10
            individual.risk=10
            individual.sharpe_ratio=0
        individual.sharpe_ratio = ((individual.rate_of_return-riskfree) / individual.risk \
            if individual.risk != 0 else individual.rate_of_return) # Sharpe ratio calculation
        #if individual.sharpe_ratio<4:
        #    individual.sharpe_ratio=individual.sharpe_ratio
        #else:
        #    individual.sharpe_ratio=0
        individual.fitness = 0  # Fitness function value
        #individual.fitness=individual.risk
        #individual.fitness = individual.rate_of_return


# Runs the EA algorithm over a number of independent runs and prints and saves results.
if __name__ == '__main__':
    task = Trading()
    task.run(population_size=POP_SIZE, xover_prob=XOVER_PROB, tournament_size=TOURNAMENT_SIZE,
             no_of_gens=nGENS, no_of_runs=nRuns, training_dataset=training_data, test_dataset=test_data,
             class_object=Trading.Individual(""),name=name,r=THETA_of_price,day=n,MAX_DEPTH=MAX_DEPTH,threshold=threshold,min_no_of_trades=MIN_NO_OF_TRADES,POP_SIZE=POP_SIZE,XOVER_PROB=XOVER_PROB,TOURNAMENT_SIZE=TOURNAMENT_SIZE)
    FileWriter.log_experimental_setup_gp(INITIAL_DEPTH, MAX_DEPTH, FLIP_MUT_PROB, F_SET, T_SET)
