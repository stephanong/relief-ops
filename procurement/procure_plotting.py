#!/usr/bin/env python3
""" Plot the results of simulation """

from itertools import product
import pandas as pd

## -- Constant and setting
PLYTEMPLATE = "simple_white" # plotly theming

METHOD2TITLE = { # setup the LaTeX-style titles for methods
    "pct90"         : r"$\rm P{\small CT}90$", 
    "pct100"        : r"$\rm P{\small CT}100$", 
    "stoch"         : r"$\rm S{\small TOCH}$", 
    "stochcvar90"   : r"$\rm S{\small TOCH}CV{\small A}R90$", 
    "stochcvar95"   : r"$\rm S{\small TOCH}CV{\small A}R95$", 
    "stochcvarw90"  : r"$\rm S{\small TOCH}CV{\small A}RW90$", 
    "stochcvarw95"  : r"$\rm S{\small TOCH}CV{\small A}RW95$", 
    "affine"        : r"$\rm ARO$", 
}

WSOPT = "clairvoyant" # name of the optimiser that produces the wait-and-see solutions

## -- Read functions
from utils import STSRecording, index_to_month, Config
#from procure_simulator import Config

def load_csv(filename):
    """ read a single CSV result file and produce a dataframe"""   
    
    data = STSRecording()
    data.read_csv(filename)
    
    header = [index_to_month((data.start+i)%12)+"-"+str(1+(data.start+i)//12)
              for i in range(data.n)]
    return pd.DataFrame(
        sum(data.detail, []), 
        index=pd.MultiIndex.from_tuples(
            product(range(len(data.detail)), data.label), 
            names=["scen", "label"]), 
        columns=header).swaplevel("scen", "label").transpose()

def load_result(folder, config, model, test, lstmethod):
    """ load a set of results and produce a dictionary of dataframe for each method """
    
    return {
        method:load_csv("{}/output{}-{}-{}-{}.csv".format(folder, config, model, test, method))
        for method in lstmethod
    }

def load_ws(folder, config, test):
    """ load the wait-and-see results and produce a dataframe """
    
    return load_csv("{}/output{}-{}-{}.csv".format(folder, config, test, WSOPT))

def read_budget(folder, config):
    """ read the budget setting from the configuration file """
    
    return Config("{}/simconfig{}.json".format(folder, config)).simulator["budget"]
    
## -- Plot functions

import plotly.offline as ply
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_boxplot(df, lstmethod, label, output):
    """ plot to compare methods on a single labeled data """    
    
    n = len(lstmethod)
    fig = make_subplots(rows=1, cols=n, shared_yaxes=True,
                        subplot_titles=[METHOD2TITLE[method] for method in lstmethod])
    fig.update_annotations(font_size=13, font_color="gray")                        
    for i, method in enumerate(lstmethod):
        toplot = df[method][label].transpose()
        nrow = len(toplot.index)
        fig.add_trace(go.Box(x=toplot.columns.tolist()*nrow,
                             y=toplot.values.flatten().tolist(),
                             marker_size=2, line_width=1.2, 
                             showlegend=False), 
                      row=1, col=i+1)
    fig.update_layout(plot_bgcolor="white", font=dict(family="sans-serif", size=11), 
                      template=PLYTEMPLATE, autosize=False, height=420, width=300*len(lstmethod))
    fig.update_yaxes(showline=True, linecolor="black", gridcolor="lightgrey", gridwidth=0.2)
    fig.update_xaxes(showline=True, linecolor="black", gridcolor="lightgrey", gridwidth=0.2)
    fig.write_image(output)

def plot_spent(df, lstmethod, budget, output):
    """ plot to compare the budget spent in each method """
    
    fig = go.Figure()
    ncol = len(df[lstmethod[0]]["budget"].columns) if lstmethod else 0
    fig.add_trace(go.Box(x=sum([[METHOD2TITLE[method]]*ncol for method in lstmethod], []),
                         y=sum([(budget-df[method]["budget"].iloc[-1,:].values.flatten()).tolist() for method in lstmethod], []), 
                         marker_size=2, line_width=0.8))       
    fig.update_layout(plot_bgcolor="white", font=dict(family="sans-serif", size=11),
                      template=PLYTEMPLATE, autosize=False, height=420, width=300)
    fig.update_yaxes(showline=True, linecolor="black", gridcolor="lightgrey", gridwidth=0.2)
    fig.update_xaxes(showline=True, linecolor="black")
    fig.write_image(output)    
    
    return    

LABEL2PLOT = { # choose the plotting method for each data
    "shortage" : plot_boxplot, 
    "bought"   : plot_boxplot,
    "waste"    : plot_boxplot, 
    "usable"   : plot_boxplot,
    "budget"   : plot_spent
}
def plot_all(folder, config, model, test, lstmethod, lstlabel):
    """ bulk plot for multiple labels and methods """

    budget = read_budget(folder, config)    
    df = load_result(folder, config, model, test, lstmethod)
    for label in lstlabel:
        func = LABEL2PLOT[label]
        output = "fig{}-{}-{}-{}-{}.pdf".format(config, model, test, "".join(lstmethod), label)
        
        if func==plot_boxplot:
            func(df, lstmethod, label, output)
        elif func==plot_spent:
            func(df, lstmethod, budget, output)
    
## -- Main program

if __name__ == "__main__":
    
    # ++ this snipet to avoid the current bug with kaleido package
    import time
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16]))
    fig.write_image("dummy.pdf", format="pdf") # dummy.pdf generated can be deleted
    time.sleep(2)
    # ++ 
    
    folder = "output"
    
    config, model, test = "", "realhist5", "realhist5"
    lstlabel = ["shortage", "bought", "waste", "usable", "budget"]
    lstmethod = ["pct100", "stoch", "affine"]
    plot_all(folder, config, model, test, lstmethod, lstlabel)
    
    config, model, test = "", "realhist5", "gentest100s0"
    plot_all(folder, config, model, test, lstmethod, lstlabel)
    
    config, model, test = "", "genhist100s0", "gentest100s0"
    lstmethod = ["pct100", "pct90"]
    plot_all(folder, config, model, test, lstmethod, lstlabel)
    lstmethod = ["stoch", "stochcvar95", "stochcvarw95"]
    plot_all(folder, config, model, test, lstmethod, lstlabel)
    lstmethod = ["pct90", "stochcvarw95", "affine"]
    plot_all(folder, config, model, test, lstmethod, lstlabel)    
    
    config, model, test = "n13r0", "genhist100s0", "gentest100s0"
    plot_all(folder, config, model, test, lstmethod, lstlabel)
    
