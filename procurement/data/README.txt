[FILE LIST]

Input data

- realhist5.csv, real historical data of years 2016-2020 (thus 5 scenarios)
- genhist100s0.csv, simulated historical data of 100 scenarios (for modelling)
- gentest100s0.csv, simulated test data of 100 scenarios 


[FILE FORMAT]

All CSV files have the same format:

line 1: [m]
line 2: [n]
line 3: [start month]
line 4: [list of data labels]
next m*k lines: [m blocks of data, each block has n columns and k rows]

Here k is the number of data labels in line 4. The rows of each block show 
the data corresponding to the labels, and they appear in the exact order as 
listed in line 4.

Example of the scenario file:

m = number of scenarios
n = length of the planning horizon in months
start month = Jan
list of data labels = refugee, price (so number of refugees and rice prices)

