#!/usr/bin/env python3
""" Create result tables for simulation """

import pandas as pd
import numpy as np

from scipy import stats


from procure_plotting import load_result, load_ws, read_budget

## -- Name conversions

SCEN2STR = { 
    0: "2016",
    1: "2017",
    2: "2018",
    3: "2019",
    4: "2020" 
}

METHOD2STR = { # these are macros used in LaTeX
    "pct90"        : r"\nameoptpct{90}",
    "pct100"       : r"\nameoptpct{100}", 
    "stoch"        : r"\nameoptstoch", 
    "stochcvar90"  : r"\nameoptcvar{90}",
    "stochcvar95"  : r"\nameoptcvar{95}",
    "stochcvarw90" : r"\nameoptcvarw{90}",
    "stochcvarw95" : r"\nameoptcvarw{95}",
    "affine"       : r"\nameoptaffine" 
}

LABEL2STR = { #
    "shortage" : r"Shortage ($10^3$ kg)",
    "waste"    : r"Waste ($10^3$ kg)",
    "budget"   : r"Budget spent ($10^6$ Rp)"
}

## Aggregation, statistic and conversion rules

AGGREGATE = {
    "shortage" : (lambda x,y: x.sum()/1e3), # sum and convert kg to tonne
    "waste"    : (lambda x,y: x.sum()/1e3), # same
    "budget"   : (lambda x,y: (y-x.iloc[-1,:])/1e6) # sum and convert Rp to MRp
}

DIFFERENTIATE = {
    "shortage" : (lambda x,y: (x.sum()-y.sum())/1e3), # sum the difference and convert kg to tonne
    "waste"    : (lambda x,y: (x.sum()-y.sum())/1e3), # same
    "budget"   : (lambda x,y: (y.iloc[-1,:]-x.iloc[-1,:])/1e6) # sum the difference of remained budgets and convert Rp to MRp
}

COMPUTE = {
        "min" : (lambda x: x.quantile(0.0)),
        "med" : (lambda x: x.quantile(0.5)),
        "q75" : (lambda x: x.quantile(0.75)),
        "q90" : (lambda x: x.quantile(0.9)),
        "q95" : (lambda x: x.quantile(0.95)),
        "max" : (lambda x: x.quantile(1.0)),
        "mean": (lambda x: x.mean()),
        "evpi": (lambda x: x.mean()), # also a mean but use differentiate instead of aggregate
        "std" : (lambda x: x.std()),
        "ci95": (lambda x: stats.norm.interval(0.95,loc=x.mean(),scale=x.sem())
                           if x.sem()>0 else "-") 
}

NUMERIC = {
    "shortage" : (lambda x: np.round(x, 3)), 
    "waste"    : (lambda x: np.round(x, 3)), 
    "budget"   : (lambda x: np.round(x, 3))     
}

def aggregate(data, label, budget):
    """ aggregate the data """

    return NUMERIC[label](AGGREGATE[label](data, budget))
    
def differentiate(data, label, ref):
    """ differenciate compared to a reference result """
    
    return NUMERIC[label](DIFFERENTIATE[label](data, ref))


def compute(data, label, budget, ws, stat):
    """ aggregate, differentiate and compute a stat of the data """
    
    val = COMPUTE[stat](AGGREGATE[label](data, budget) 
                        if stat!="evpi" else DIFFERENTIATE[label](data, ws))
    
    try:
        numval = NUMERIC[label](val)
    except TypeError:
        return val
    
    return numval
        
## -- Table construction

def create_full_table(folder, config, model, test, lstmethod, lstlabel):
    """ create tables containing simulated results aggregated for each scenario """
    
    # load data 
    budget = read_budget(folder, config)
    df = load_result(folder, config, model, test, lstmethod)
    
    # write the aggregated results to tables
    for label in lstlabel:
        output = "tab-{}-{}-{}-{}.tex".format("default" if config=="" else config, model, test, label)
        res = pd.DataFrame([aggregate(df[method][label], label, budget)
                            for method in lstmethod], index=lstmethod).rename(columns=SCEN2STR, index=METHOD2STR) 
        res.columns.names=["Method"]
        res.to_latex(output, escape=False)

def create_stat_table(folder, config, model, test, lstmethod, lstlabel, lststat):
    """ create tables containing summary statistics over the scenarios """
    
    # load data 
    budget = read_budget(folder, config)
    ws = load_ws(folder, config, test)
    df = load_result(folder, config, model, test, lstmethod)
    
    # compute the statistics and write to files
    for label in lstlabel:
        output = "tab-{}-{}-{}-{}.tex".format("default" if config=="" else config, model, test, label)
        res = pd.concat([pd.DataFrame([compute(df[method][label], label, budget, ws[label], stat) 
                                       for stat in lststat], 
                                      index=lststat, columns=[method]).transpose()
                         for method in lstmethod]).rename(index=METHOD2STR)
        res.columns.names=["Method"]
        res.to_latex(output, escape=False)

def create_stat_table1(folder, config, model, test, lstmethod, lstlabel, lststat):
    """ create a single table containing summary statistics over the scenarios """
    
    # load data 
    budget = read_budget(folder, config)
    ws = load_ws(folder, config, test)
    df = load_result(folder, config, model, test, lstmethod)
    
    # compute the statistics and write to the file
    output = "tab-{}-{}-{}.tex".format("default" if config=="" else config, model, test)    
    res = pd.concat([pd.concat([pd.DataFrame([compute(df[method][label], label, budget, ws[label], stat) 
                                              for stat in lststat], 
                                             index=lststat, columns=[method]).transpose()
                                for method in lstmethod]).rename(index=METHOD2STR)
                     for label in lstlabel], axis=1)
    res.columns = [sum([[LABEL2STR[label]]*len(lststat) for label in lstlabel],[]),
                   lststat*len(lstlabel)]
    res.columns.names=["Method"," "]
    res.to_latex(output, escape=False, multicolumn_format="c")
    
## -- Main program
    
if __name__ == "__main__":  
    
    def generate_tables(folder, config=""):
        """ generate all tables for a specific configuration """
    
        model, test = "realhist5", "realhist5"
        lstlabel = ["shortage", "waste", "budget"]
        lstmethod = ["pct100", "stoch", "affine"]
        create_full_table(folder, config, model, test, lstmethod, lstlabel)
        create_stat_table1(folder, config, model, test, lstmethod, lstlabel, lststat=["med", "min", "max", "evpi"]) 
        
        model, test = "realhist5", "gentest100s0"
        create_stat_table(folder, config, model, test, lstmethod, lstlabel, lststat=["med", "q75", "q90", "q95", "max", "mean", "std", "ci95"]) 
        create_stat_table1(folder, config, model, test, lstmethod, lstlabel, lststat=["med", "q90", "max", "evpi"]) 
        
        model, test = "genhist100s0", "gentest100s0"
        lstmethod = ["pct90", "pct100", "stoch", "stochcvar90", "stochcvar95", "stochcvarw90", "stochcvarw95", "affine"]
        create_stat_table(folder, config, model, test, lstmethod, lstlabel, lststat=["med", "q75", "q90", "q95", "max", "mean", "std", "ci95"]) 
        create_stat_table1(folder, config, model, test, lstmethod, lstlabel, lststat=["med", "q90", "max", "evpi"]) 
    
    # location of output data files
    folder = "output"
    
    generate_tables(folder, "")
    generate_tables(folder, "n13r0")
    
