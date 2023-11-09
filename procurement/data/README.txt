[FILE LIST]


Input data

- realhist5.csv, real historical data of years 2016-2020 (thus 5 scenarios)
- genhist100s0.csv, simulated historical data of 100 scenarios (for modelling)
- gentest100s0.csv, simulated test data of 100 scenarios 


Output results

- output-[historical data]-[test data]-[method].csv:
  results for 12 months with safety stock of 10 tonne per municipality
  as used in practice 

- outputn13r0-[historical data]-[test data]-[method].csv:
  results for 13 months with zero safety stock
  


[FILE FORMAT]

All files has the same format:

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

