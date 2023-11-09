[FILE LIST]

Output results

- output-[historical data]-[test data]-[method].csv:
  results for 12 months with safety stock of 10 tonne per municipality

- outputn13r0-[historical data]-[test data]-[method].csv:
  results for 13 months with zero safety stock


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

