# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:20:51 2022

@author: danie
"""
import sys
import os 
import pandas as pd 
from distutils.util import strtobool

parameters = ["dt", "steps", "stepsCont","delay","frequency","spectralwidth","aFieldStrength","b","c","ODscaler","dim","CoupledHamiltonian","spectrum","optdensity"]
filename = "dataset"


def setup(filename = filename):
    if os.path.exists(filename):
        if user_yes_no_query("File already exists, overwrite and create a new one?"):
            print("new")
            data = pd.DataFrame(columns = parameters)
            data.to_csv(f'{filename}.csv',index=False)  
        else:
            print("aborted")
    else: 
        data = pd.DataFrame(columns = parameters)
        data.to_csv(f'{filename}.csv',index=False)  

def user_yes_no_query(question):
    sys.stdout.write('%s [y/n]\n' % question)
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            sys.stdout.write('Please respond with \'y\' or \'n\'.\n')
if __name__ == '__main__':
    setup()