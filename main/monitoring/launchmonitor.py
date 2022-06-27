# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:35:29 2022

@author: danie
"""
import webbrowser
from tensorboard import program

tracking_address = 'logs/fit/' # the path of your log file.

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    tb.launch()
    
    webbrowser.register('chrome',
    	None,
    	webbrowser.BackgroundBrowser("C://Program Files (x86)//Google//Chrome//Application//chrome.exe"))
    webbrowser.get('chrome').open_new_tab('http://localhost:6006/')
