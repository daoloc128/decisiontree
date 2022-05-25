import numpy as np
import pandas as pd
import json

data = pd.read_excel(open('myData.xlsx', 'rb'), sheet_name='rống-data') 
data.info()
data.head()

def caculate_H_S(data, y_label, class_list):
    number_of_row = data.shape[0] 
    H_S = 0.0
    
    for c in class_list:
        number_of_c = data[data[y_label] == c].shape[0] 

        if (number_of_c != 0 and number_of_row != 0):
          probability = number_of_c / number_of_row 
          entropy = - probability * np.log2(probability) 
          H_S += entropy 
    
    return H_S 


def caculate_H_S_i(H_S_i_data, y_label, class_list):
    number_of_row = H_S_i_data.shape[0]
    H_S_i = 0.0
    
    for c in class_list:
        number_of_c = H_S_i_data[H_S_i_data[y_label] == c].shape[0] 
        entropy = 0.0
        
        if (number_of_c != 0 and number_of_row != 0):
          probability = number_of_c / number_of_row 
          entropy = - probability * np.log2(probability) 
          H_S_i += entropy
    
    return H_S_i

def caculate_information_gain(H_S_name, data, y_label, class_list):
    H_S_value_list = data[H_S_name].unique() 
    number_of_row = data.shape[0] 
    I = 0.0
    
    for value in H_S_value_list:
        H_S_i_data = data[data[H_S_name] == value] 
        number_of_H_S_i_row = H_S_i_data.shape[0] 
        H_S_i = caculate_H_S_i(H_S_i_data, y_label, class_list) 

        if (number_of_row != 0 and number_of_H_S_i_row != 0):
          probability = number_of_H_S_i_row / number_of_row 
          I += probability * H_S_i
               
    information_gain = caculate_H_S(data, y_label, class_list) - I

    return information_gain

def find_max_information_gain_feature(data, y_label, class_list):
    H_S_name_list = data.columns.drop(y_label) 
    max_information_gain = -1
    max_information_gain_feature = None
    
    for H_S_name in H_S_name_list:
        information_gain = caculate_information_gain(H_S_name, data, y_label, class_list) 
        
        if (max_information_gain < information_gain):
            max_information_gain = information_gain
            max_information_gain_feature = H_S_name
    
    return max_information_gain_feature

def generate_sub_tree(H_S_name, data, y_label, class_list):
    H_S_count_dict = data[H_S_name].value_counts(sort=False) 
    tree = {}
    
    for name, count in H_S_count_dict.iteritems():
        H_S_i_data = data[data[H_S_name] == name] 
        stop = False 
        
        for c in class_list:
            number_of_c = H_S_i_data[H_S_i_data[y_label] == c].shape[0]
            
            if (number_of_c == count): 
                tree[name] = c 
                data = data[data[H_S_name] != name]
                stop = True 
                
        if (not stop):
            tree[name] = "continue" 
        
    return tree, data

def generate_tree(root, prev_H_S_i, data, y_label, class_list):
    if (data.shape[0] != 0): 

        max_information_gain_feature = find_max_information_gain_feature(data, y_label, class_list) 
        tree, data = generate_sub_tree(max_information_gain_feature, data, y_label, class_list) 
        next_root = None 
        
        if (prev_H_S_i != None):
            root[prev_H_S_i] = dict()
            root[prev_H_S_i][max_information_gain_feature] = tree
            next_root = root[prev_H_S_i][max_information_gain_feature]

        else: 
            root[max_information_gain_feature] = tree
            next_root = root[max_information_gain_feature]
        
        for node, branch in list(next_root.items()): 
            if (branch == "continue"): 
                H_S_i_data = data[data[max_information_gain_feature] == node]
                generate_tree(next_root, node, H_S_i_data, y_label, class_list)

def decision_tree(data, y_label):
    data_copy = data.copy()
    tree = {}
    class_list = data[y_label].unique()
    
    generate_tree(tree, None, data_copy, y_label, class_list)
     
    return tree

tree = decision_tree(data, 'con vật')


