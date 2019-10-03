#!/bin/bash

#Pah to alarm data
AlarmdataPath="alarm_data/create_trainset.txt"

#Path to graph structure
GraphstructurePath="alarm_data/alarm_graph_test_reason1.txt"

python main.py --alarm_data $AlarmdataPath --Graph_structure $GraphstructurePath 
