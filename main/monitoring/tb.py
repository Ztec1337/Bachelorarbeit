# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 17:11:31 2022

@author: danie
"""
from tensorboard import program

tracking_address = 'logs/fit/' # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
