# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 12:04:00 2023

@author: kasj
"""

# Function set_dirs returns directory of choice.
def set_dirs(choice: str):    
    
    dirs = {'local': 'C:/Users/kasj/mass_balance_model/',
            'server': '/mirror/khsjursen/mass_balance_model/'}

    if choice == 'local':
        return dirs['local']
    elif choice == 'server':
        return dirs['server']
    else:
        print('Choose directory local or server.')
