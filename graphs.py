import json
import numpy as np
import matplotlib.pyplot as plt

def find_lowest():
    with open('function_bank.json', 'r') as file:
        json_array = json.load(file)
        
        min_obj = None
        minimum = float('inf')
        for obj in json_array:
            if obj['class_loss'] + obj['regress_loss'] < minimum:
                minimum = obj['class_loss'] + obj['regress_loss']
                min_obj = obj
        
        print(min_obj['class_loss'] + min_obj['regress_loss'])


def find_all_losses():
    with open('function_bank.json', 'r') as file:
        json_array = json.load(file)
        
        losses = []
        for obj in json_array:
            losses.append(obj['class_loss'] + obj['regress_loss'])
        
        return losses


find_lowest()
# find_all_losses()

# data = find_all_losses()

# # Create the plot
# plt.plot(data, marker='o', linestyle='-', color='b', label='Sum of Class and Regress Loss')

# # Add labels and title
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Losses at Exploration and Modification Phase')

# # Add a legend
# plt.legend()

# # Display the plot
# plt.show()