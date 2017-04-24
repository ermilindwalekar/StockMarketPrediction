from StockPredictor import StockPredictor, ParseData, PlotData
import pandas as pd
from pandas.io.data import DataReader
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.ensemble import RandomForestRegressor
from TFMLP import MLPR
import sys, requests

from datetime import datetime

def PrintUsage():
    print('Usage:\n')
    print('\tpython stocks.py <csv file> <start date> <end date> <D|W|M>')
    print('\tD: Daily prediction')
    print('\tW: Weekly prediction')
    print('\tM: Montly prediction')


def Main(args):
    if(len(args) != 3 and len(args) != 4):
        PrintUsage()
        return
        
    #Obtain CSV from Internet
    print('\n#############   Downloading Historical Data from Internet   #############')
    file = args[0]+'.csv'
    str2 = 'http://real-chart.finance.yahoo.com/table.csv?s='+args[0]+'.NS&d=02&e=16&f=2017&g=d&a=0&b=1&c=1996&ignore=.csv'
    f = open(file, 'wb')
    f.write(requests.get(str2).content)
    f.close()
    print('\n#############   Download Complete   ##############')
    #Test if file exists
    try:
        open(file)
    except Exception as e:
        print('Error opening file: ' + file)
        print(str(e))
        PrintUsage()
        return
    #Test validity of start date string
    try:
        datetime.strptime(args[1], '%Y-%m-%d').timestamp()
    except Exception as e:
        print('Error parsing date: ' + args[1])
        PrintUsage()
        return
    #Test validity of end date string
    try:
        datetime.strptime(args[2], '%Y-%m-%d').timestamp()
    except Exception as e:
        print('Error parsing date: ' + args[2])
        PrintUsage()
        return    
    #Test validity of final optional argument
    if(len(args) == 4):
        predPrd = args[3].upper()
        if(predPrd == 'D'):
            predPrd = 'daily'
        elif(predPrd == 'W'):
            predPrd = 'weekly'
        elif(predPrd == 'M'):
            predPrd = 'monthly'
        else:
            PrintUsage()
            return
    else:
        predPrd = 'daily'
    print('\n#############   Processing on Data   ##############')
    #Everything looks okay; proceed with program
    D = ParseData(file)
    #The number of previous days of data used
    #when making a prediction
    numPastDays = 20
    PlotData(D,args[0])
    #Number of neurons in the input layer
    i = numPastDays * 7 + 1
    #Number of neurons in the output layer
    o = D.shape[1] - 1
    #Number of neurons in the hidden layers
    h = int((i + o) / 2)
    #The list of layer sizes
    layers = [i, h, h,o]
    #Type of Regressor Used
    #R = RandomForestRegressor(n_estimators = 5) 
    R = MLPR(layers, maxItr = 1000, tol = 0.40, reg = 0.001, verbose = True)
    sp = StockPredictor(R, nPastDays = numPastDays)
    #Learn the dataset and then display performance statistics
    sp.Learn(D)
    #sp.TestPerformance()
    #Perform prediction for a specified date range
    P,R = sp.PredictDate(args[1], args[2], predPrd)
    print('\n#############   Predection is on the way   ##############')
    #Keep track of number of predicted results for plot
    n = P.shape[0]
    #Append the predicted results to the actual results
    D = P.append(D)
    #Predicted results are the first n rows
    PlotData(D,args[0],range(n + 1))
    
    return (R, n)
    

#Main entry point for the program
if __name__ == "__main__":
    p, n = Main(sys.argv[1:])
    #p, n = Main(['SENSEX.csv', '2010-12-01', '2016-12-31', 'D'])
   
	#p, n = Main(['GAIL', '2016-12-16', '2016-12-17', 'D'])
print('\n#############   Predection Complete   ##############')
print(str(p))
