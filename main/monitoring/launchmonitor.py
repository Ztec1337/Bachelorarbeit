# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 16:35:29 2022

@author: danie
"""
import subprocess
import sys
import webbrowser

if __name__ == "__main__":
    webbrowser.register('chrome',
    	None,
    	webbrowser.BackgroundBrowser("C://Program Files (x86)//Google//Chrome//Application//chrome.exe"))
    subprocess.Popen((sys.executable, 'tb.py'))
    webbrowser.get('chrome').open_new_tab('http://localhost:6006/')
