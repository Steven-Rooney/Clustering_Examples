import sys
import argparse
import pylab as plt
import numpy as np
import pickle

import utils
import kmeans
import dbscan

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=['1', '2', '3'])

    io_args = parser.parse_args()
    question = io_args.question


    if question == '1':
        #X = utils.load_dataset('clusterData')['X']
        with open("C:\\Users\\Steven\\Desktop\\Assignment2-Codes\\data\\clusterData.pkl", 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        X = data['X']


        model = kmeans.fit(X, k=4)
        utils.plot_2dclustering(X, model['predict'](model, X))
        print("Displaying figure...")
        plt.show()
        # part 1: implement kmeans.error
        # part 2: get clustering with lowest error out of 50 random initialization
        error = 0
        for i in range(50):
            print('iteration', i)
            model = kmeans.fit(X, k=4)
            temp_err = model['error'](model, X)
            if error == 0 or error < temp_err:
                error = model['error'](model, X)
                best_model = model

        utils.plot_2dclustering(X, best_model['predict'](best_model, X))  
        plt.show()
            


    if question == '2':
        X = utils.load_dataset('clusterData2')['X']
        model = dbscan.fit(X, radius2=1, min_pts=3)
        y = model['predict'](model, X)
        utils.plot_2dclustering(X,y)
        print("Displaying figure...")
        plt.show()

    if question == '3':
        dataset = utils.load_dataset('animals')
        X = dataset['X']
        animals = dataset['animals']
        traits = dataset['traits']

        #model = kmeans.fit(X, k=5)
        #y = model['predict'](model, X)
        
        model = dbscan.fit(X, 5, 3)
        utils.plot_2dclustering(X, model['predict'](model, X))
        print("Displaying figure...")
        plt.show()
        for kk in range(max(y)+1):
            print('Cluster {}: {}'.format(kk+1, ' '.join(animals[y==kk])))

