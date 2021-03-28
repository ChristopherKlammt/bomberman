import numpy as npimport tablesfrom matplotlib import pyplot as pltfrom argparse import ArgumentParserfrom agent_code.dqn.parameters import (FILENAME, ROW_SIZE)# 1. Change filename in parameters.py# 2. 'python evaluate_training.py create'# 3. Adapt training hyperparameters in parameters# 4. train model with saving data (EVALUATION = True)# 5. python evaluate_training.py # The file currently contains:# number of steps, total auxiliary reward, collected coins per round, killed opponents per round, number of self killsdef evaluate(filename, filename2 = None, stop=-1, start=0):    step_size = 50        # Combine two files in one data,     # stop is the end of the first file, start is the start of the second    if filename2:        file1 = tables.open_file(filename, mode='r')        file2 = tables.open_file(filename2, mode='r')        y = np.concatenate((np.array(file1.root.data)[0:stop], np.array(file2.root.data)[start:]))    else: # only one file to plot        # read the data saved in <filename>        file = tables.open_file(filename, mode='r')        y = np.array(file.root.data)        len_of_y = int(float(np.shape(y)[0]/step_size)) # number of groups after combining STEP_SIZE values to one     y_mean = np.empty((len_of_y, np.shape(y)[1]))    y_std = np.empty((len_of_y, np.shape(y)[1]))        y[:,4] *= 100 # multiply self kills with 100 to get the percentage of self kills by computing the mean    y[:,2] *= 100/9 # percentage of collected coins        # compute mean and standard deviation over step_size training rounds    for i in range(len_of_y):        y_i = y[step_size*i:step_size*(i+1)].astype(np.float)        for j in range(np.shape(y)[1]):            y_mean[i,j] = np.mean(y_i[:,j])            y_std[i,j]  = np.std(y_i[:,j])       x = np.arange(step_size, (len_of_y+1)*step_size, step_size)            plt.errorbar(x, y_mean[:,0], y_std[:,0], label = 'number of steps')    plt.errorbar(x, y_mean[:,1], y_std[:,1], label = 'total reward including auxiliary reward')    plt.errorbar(x, y_mean[:,2], y_std[:,2], label = 'percentage of collected coins per round')    plt.errorbar(x, y_mean[:,3], y_std[:,3], label = 'killed opponents')    plt.plot(x, y_mean[:,4], label = 'percentage of rounds ended by self kill')        plt.plot(x, y_mean[:,5], label = 'number of crates destroyed')    plt.plot(x, y_mean[:,6], label = 'number of points')    plt.plot(x, y_mean[:,7], label = 'averge points of enemies')    plt.plot(x, y_mean[:,8], label = 'number of invalid actions')        plt.xlabel('number of training rounds')    plt.legend()    plt.show()    def create(filename, row_size, num_columns):    # create an extendable EArray storage        file = tables.open_file(filename, mode='w')    atom = tables.Float64Atom()    data_array = file.create_earray(file.root, 'data', atom, (0, row_size))    file.close()###parser = ArgumentParser()subparsers = parser.add_subparsers(dest='command_name', required=False)create_parser = subparsers.add_parser("create")args = parser.parse_args()if args.command_name == "create":    create(FILENAME, ROW_SIZE, 0)else:    evaluate(FILENAME)        