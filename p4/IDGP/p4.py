import argparse
import operator
import random
import time

import evalGP_main as evalGP
import feature_function as fe_fs
# only for strongly typed GP
import gp_restrict
import numpy as np
import pandas as pd
import pygraphviz as pgv
# deap package
from deap import base, creator, gp, tools
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from strongGPDataType import Img, Int1, Int2, Int3, Region, Vector
import p4_2 



randomSeeds = 2

# 'FLGP'
# get the user arg from command line
# user defined f1 or f2 parameters
parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()
user_input_arg_f1_or_f2 = args.path
if user_input_arg_f1_or_f2  not in ["f1" , "f2"]:
    raise ValueError("Please input f1 or f2")


# user_input_arg_f1_or_f2 = 'f1'


# dataset_prefix_path = "/home/zhouyun/Desktop/aiml426-a2/p4/FEI-dataset"
dataset_prefix_path = "../FEI-dataset"
dataset_full_path = dataset_prefix_path + "/" + user_input_arg_f1_or_f2 + "/" + user_input_arg_f1_or_f2

x_train = np.load(dataset_full_path + '_train_data.npy') / 255.0
y_train = np.load(dataset_full_path + '_train_label.npy')
x_test = np.load(dataset_full_path + '_test_data.npy') / 255.0
y_test = np.load(dataset_full_path + '_test_label.npy')


print(dataset_full_path)

'FLGP'


# parameters:
population = 100
generation = 50
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 8

bound1, bound2 = x_train[1, :, :].shape
# GP

pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')
# Feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector, name='FeaCon2')
pset.addPrimitive(
    fe_fs.root_con, [Vector, Vector, Vector], Vector, name='FeaCon3')
# Global feature extraction
pset.addPrimitive(fe_fs.all_dif, [Img], Vector, name='Global_DIF')
pset.addPrimitive(fe_fs.all_histogram, [Img], Vector, name='Global_Histogram')
pset.addPrimitive(fe_fs.global_hog, [Img], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='Global_SIFT')
# Local feature extraction
pset.addPrimitive(fe_fs.all_dif, [Region], Vector, name='Local_DIF')
pset.addPrimitive(fe_fs.all_histogram, [
                  Region], Vector, name='Local_Histogram')
pset.addPrimitive(fe_fs.local_hog, [Region], Vector, name='Local_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Region], Vector, name='Local_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Region], Vector, name='Local_SIFT')
# Region detection operators
pset.addPrimitive(
    fe_fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
pset.addPrimitive(
    fe_fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')
# Terminals
pset.renameArguments(ARG0='Grey')
pset.addEphemeralConstant('X', lambda: random.randint(0, bound1 - 20), Int1)
pset.addEphemeralConstant('Y', lambda: random.randint(0, bound2 - 20), Int2)
pset.addEphemeralConstant('Size', lambda: random.randint(20, 51), Int3)

# fitnesse evaluaiton
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD,
                 pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mapp", map)


def evalTrain(individual):
    # print(individual)
    func = toolbox.compile(expr=individual)
    train_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    # lsvm = LinearSVC(max_iter=2000)
    clf = LinearSVC(random_state=randomSeeds, tol=1e-5, max_iter=2000)
    
    accuracy = round(100 * cross_val_score(clf,
                     train_norm, y_train, cv=3).mean(), 2)
    return accuracy,


def evalTrainb(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf = []
        for i in range(0, len(y_train)):
            train_tf.append(np.asarray(func(x_train[i, :, :])))
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
        lsvm = LinearSVC(max_iter=100)
        accuracy = round(100 * cross_val_score(lsvm,
                         train_norm, y_train, cv=3).mean(), 2)
    except:
        accuracy = 0
    return accuracy,


# genetic operator
toolbox.register("evaluate", evalTrain)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("selectElitism", tools.selBest)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(
    key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(
    key=operator.attrgetter("height"), max_value=maxDepth))


def GPMain(randomSeeds):
    random.seed(randomSeeds)

    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(
        fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof


def evalTest(individual):
    func = toolbox.compile(expr=individual)
    train_tf = []
    test_tf = []
    for i in range(0, len(y_train)):
        train_tf.append(np.asarray(func(x_train[i, :, :])))
    for j in range(0, len(y_test)):
        test_tf.append(np.asarray(func(x_test[j, :, :])))
    train_tf = np.asarray(train_tf)
    test_tf = np.asarray(test_tf)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
    test_norm = min_max_scaler.transform(np.asarray(test_tf))
    lsvm = LinearSVC()
    lsvm.fit(train_norm, y_train)
    accuracy = round(100*lsvm.score(test_norm, y_test), 2)
    return train_tf.shape[1], accuracy



if __name__ == "__main__":

    beginTime = time.process_time()
    pop, log, hof = GPMain(randomSeeds)
    endTime = time.process_time()
    trainTime = endTime - beginTime

    num_features, testResults = evalTest(hof[0])
    endTime1 = time.process_time()
    testTime = endTime1 - endTime
    # draw tree graph and save it to pdf
    nodes, edges, labels = gp.graph(hof.items[0])
    graph = pgv.AGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    graph.layout(prog="dot")
    
    for i in nodes:
        n = graph.get_node(i)
        n.attr["label"] = labels[i]
    # graph.draw(f"{dataset_prefix_path}/tree_{user_input_arg_f1_or_f2}.pdf")
    graph.draw(f"tree_{user_input_arg_f1_or_f2}.pdf")
    graph.write(f"tree_{user_input_arg_f1_or_f2}.dot")
    
    
    #    	      	                    fitness                    	                   size_tree
    #    	      	-----------------------------------------------	-----------------------------------------------
    # gen	nevals	avg    	gen	max  	min  	nevals	std    	avg 	gen	max	min	nevals	std
    # 0  	100   	67.5665	0  	95.33	39.33	100   	19.0403	7.51	0  	46 	2  	100   	7.43034

    # then generate 2 csv files for the results
    def generate_pattern_csv(individual, x, y, output_name):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # append the results to a list
        x_out = []
        for i in range(0, len(y)):
            x_out.append(np.asarray(func(x[i, :, :])))
        x_out = np.asarray(x_out)    
        # apply normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        x_out_norm = min_max_scaler.fit_transform(x_out)
        df = pd.DataFrame(x_out_norm)
        df["label"] = y.reshape((-1, 1))
        df.to_csv(f"{output_name}.csv")
        
        # check if the pattern file is correctly saved
        df = pd.read_csv(f"{output_name}.csv")
        

    # generate_pattern_csv(hof[0], toolbox, x_train, y_train, f"{dataset_prefix_path}/train_{user_input_arg_f1_or_f2}")
    # generate_pattern_csv(hof[0], toolbox, x_test, y_test, f"{dataset_prefix_path}/test_{user_input_arg_f1_or_f2}")
    
    generate_pattern_csv(hof[0], x_train, y_train, f"train_{user_input_arg_f1_or_f2}")
    generate_pattern_csv(hof[0], x_test, y_test, f"test_{user_input_arg_f1_or_f2}")
    
    
    print('Best individual ', hof[0])
    print('Test results  ', testResults)
    print('Train time  ', trainTime)
    print('Test time  ', testTime)
    print('End')
    
    # bwlow is for 4.2, it is included in the p4_2.py file
    p4_2.image_classification_using_feature_extracted_by_GP()

