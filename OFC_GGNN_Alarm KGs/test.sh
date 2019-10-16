#!/bin/bash

#Pah to alarm data
AlarmdataPath="alarm_data/create_trainset.txt"

#Path to graph structure
GraphstructurePath="alarm_data/adjacency_matrix.txt"



python main.py --alarm_data $AlarmdataPath --Graph_structure $GraphstructurePath 
