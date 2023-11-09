""" Utility functions """

## -- imported tools and functions

from sys import exit
from timeit import default_timer as timer

from calendar import monthrange
from datetime import datetime

from numpy import std, mean, percentile, \
    cumsum, searchsorted, isnan, \
    log, power, floor, ceil, round    
from numpy.random import seed, choice, randint, \
    lognormal, uniform, triangular, poisson, permutation

import json, csv


## -- constants

ENC, NLINE, SEP, INDT = "utf-8", "", ",", 4 # file encoding, newline, CSV delimiter, indentation

MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", # month names and conversion
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH2INDEX = {MONTH[i]:i for i in range(len(MONTH))}

EPS = 1e-4 # small error
OBJTOL = 1e-4 # objective tolerance to adjust when switching objectives
BUDGETCONSTRSCALE = 1e-3 # budget contraint scale factor to avoid numerical issues with large numbers

DATATYPE = { # know data types, this is used for conversion from string
    "refugee"  : int,
    "demand"   : int,
    "price"    : float,
    "avail"    : int,
    "bought"   : int,
    "usable"   : int,
    "consume"  : int,
    "shortage" : int,
    "waste"    : int, 
    "budget"   : float
}

POSTPROCESS = { # post-processing for data generated or computed
    "refugee" : "int(ceil({}))", 
    "demand"  : "int(ceil({}))",
    "price"   : "round({},2)",
    "avail"   : "int(round({}))",
    "bought"  : "int(ceil({}))"
}

# default configuration file for both generator and simulator
DEFAULT_CONF = """ 
{
    "generator_lowres" : {
        "start"         : "Jan", 
        "length"        : 120, 
        "num_scenarios" : 100,
        "refugee" : {
            "method" : "resample",
            "merge"  : [ ["Jun", "Jul", "Aug", "Sep"],
                         ["Nov", "Dec", "Jan", "Feb", "Mar"],
                         ["Apr", "May", "Oct"] ],
            "trend"  : [0, 0.0]
        },
        "price" : {
            "method" : "lognormal",
            "merge"  : [],
            "trend"  : [0, 0.0]
        }, 
        "avail" : {
            "method" : "lognormal",
            "merge"  : [],
            "trend"  : [0, 0.0]            
        }
    }, 
    "generator" : {
        "season" : {
            "s0" : [["Jan"], ["Feb"], ["Mar"], ["Apr"], ["May"], ["Jun"], 
                    ["Jul"], ["Aug"], ["Sep"], ["Oct"], ["Nov"], ["Dec"]],
            "s1" : [["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Sep"]],
            "s2" : [["Jul", "Aug", "Sep", "Oct"], 
                    ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]],
            "s3" : [["Jun", "Jul", "Sep"],
                    ["Nov", "Dec", "Jan", "Feb", "Mar"],
                    ["Apr", "May", "Oct"]]
        },
        "event" : {
            "seasonality" : {
                "Earthquake" : "s1",
                "Flood"      : "s3",
                "Landslide"  : "s3",
                "Tornado"    : "s3",
                "Tsunami"    : "s1",
                "Wildfire"   : "s2"
            },
            "seasonal_growth"    : 0,
            "demographic_growth" : 0,
            "sampling"           : "resample" 
        },
        "price" : {
            "seasonality"      : "s0",
            "inflation_rate"   : 0,
            "sampling"         : "lognormal"
        }
    },
    "simulator" : {
        "skip"       : 0,
        "length"     : 12,
        "budget"     : 14601312000.00,
        "storage_limit_age"      : 7,        
        "safety_stock"           : 270000,
        "safety_stock_limit_age" : 6,
        "refugee_to_demand"      : 5.6,
        "min_govt_buy_price"     : 5250.00,
        "optimiser" : {
            "trend" : {
                "refugee" : [0, 0.0], 
                "price"   : [0, 0.0]
            }, 
            "percentile" : {
                "refugee" : 90,
                "price"   : 50
            },
            "stochcvar" : {
                "alpha" : 0.95
            }
        }
    }
}
"""


## -- read-write for useful file formats

class Config:
    """ Configuration file loader """
    
    def __init__(self, filename=None):
        """ constructor """
        
        if filename:
            with open(filename, "rt", encoding=ENC, newline=NLINE) as f:
                load_attr_from_dict(self, json.load(f))
        else:
            load_attr_from_dict(self, json.loads(DEFAULT_CONF))


class STSRecording:
    """ Scenario-time series recording """

    def __init__(self, n=12, start="Jan", label=[]):
        """ constructor """
        
        self.n, self.start, self.label = n, start, label
        self.detail = [] 
    
    def read_csv(self, filename):    
        """ read data from file """

        with open(filename, "rt", encoding=ENC, newline=NLINE) as f:
            reader = csv.reader(f, delimiter=SEP)
            
            N = int(next(reader)[0])
            self.n = int(next(reader)[0]) 
            self.start = month_to_index(next(reader)[0].strip())
            self.label = next(reader)
            
            self.detail = []
            for i in range(N):
                data = [auto_typing(next(reader), label) 
                        for label in self.label]
                self.detail.append(data)
    
    def write_csv(self, filename):    
        """ write data to file """
                
        with open(filename, "wt", encoding=ENC, newline=NLINE) as f:
            writer = csv.writer(f, delimiter=SEP)
            
            writer.writerow([len(self.detail)])
            writer.writerow([self.n])
            writer.writerow([index_to_month(self.start)])
            writer.writerow(self.label)
            
            for data in self.detail:
                for row in data:
                    writer.writerow(row)

## -- useful functions

def exec_and_quit(msg, code, func, *arg, **kwarg):
    """ show the message, execute a function and quit """
    
    if msg: print(msg)
    if func: func(*arg, **kwarg)
    exit(code) 

def exec_timed(msg, func, *arg, **kwarg):
    """ execute a function and show elapsed time """
    
    print(msg, "..", end=" ")
    start = timer()
    result = func(*arg, **kwarg)
    print("done! {:.3f}s".format(timer() - start))
    
    return result
    
def load_attr_from_dict(obj, dic):
    """ add a set of attributes to object from a dictionary """
        
    for key, val in dic.items():
        setattr(obj, key, val) 

def is_dict_with_key(obj, key):
    """ check if the obj is a dictionary with specific key """
    
    return type(obj)==dict and key in obj

def month_to_index(month,shift=0): 
    """ convert month names to indices in modulo 12 """
    
    return MONTH2INDEX[month.strip()]+shift if type(month)==str \
           else [MONTH2INDEX[m.strip()]+shift for m in month]

def index_to_month(index):
    """ convert month indices to name in modulo 12 """

    return MONTH[index%12] if type(index)==int \
           else [MONTH[i%12] for i in index]

def random_seed(rseed):
    """ initiate the random seed """

    seed(rseed)

def sample_uniform_int(low=0, high=2):
    """ sample uniformly a number between low and high """
    
    return randint(low, high)

def sample_choice(data, size=1):
    """ sample uniformly from data with replacement """    
    
    return choice(data, size, replace=True) \
           if size>1 else [choice(data, replace=True)]

def sample_uniform_int(low=0, high=2):
    """ sample uniformly a number between low and high """
    
    return randint(low, high)
    
def sample_permutation(inp):
    """ sample a random permutation """
    
    return permutation(inp).tolist()    

def sample_choice(data, size=1):
    """ sample uniformly from data with replacement """    
    
    return choice(data, size, replace=True) \
           if size>1 else [choice(data, replace=True)]

def sample_custom(dic, size=1):
    """ sample a custom distribution based on provided weights """
    
    k, v = list(dic.keys()), list(dic.values())
    n = sum(v)
    if n<=0: return []
    
    return choice(k, p=[i/n for i in v], size=size).tolist() if size>1 \
           else [choice(k, p=[i/n for i in v])]
    
def sample_triangular(low, avg, high, size=1):
    """ sample a triangular distribution with given bounds and average """
    
    if low==high: return [low]        
    
    mode = low if avg<(2*low+high)/3 \
           else (high if avg>(low+2*high)/3 else 3*avg-low-high)
    
    return triangular(low, mode, high, size=size).tolist() if size>1 \
           else [triangular(low, mode, high)]
    
def sample_poisson(avg, size=1):
    """ sample a poisson distribution with given average """    
    
    return poisson(avg, size=size) if size>1 \
           else [poisson(avg)]
    

def sample_day(season, year=2021, size=1):
    """ sample a random date string within a set of months """

    cumlen = cumsum([monthrange(year,month_to_index(m,shift=1))[1] for m in season])    
    day = choice(range(cumlen[-1]), size)+1
    monthidx = searchsorted(cumlen, day)
    
    return [datetime(year,
                month_to_index(season[monthidx[i]],shift=1),
                d-(cumlen[monthidx[i]-1] if monthidx[i]>0 else 0))
            for i,d in enumerate(day.tolist())]
           
def sample_lognormal(theta, mu, sigma, size=1):
    """ sample from a delta-log-normal distribution """
    
    if theta==0:
        return lognormal(mu, sigma, size).tolist() if size>1 else [lognormal(mu, sigma)]
    
    if theta==1:
        return [0]*size
    
    res = []
    for i in range(size):
        if uniform()<theta:
            res.append(0)
        else: 
            res.append(lognormal(mu, sigma))
    return res

def estimate_param_lognormal(data):
    """ estimate parameters of a delta-log-normal distribution """
    
    nz = data.count(0)
    theta = nz/len(data)
    
    mu, sigma = .0, .0
    if theta<1:
        lndata = [log(s) for s in data if s>0]
        mu = mean(lndata)
        sigma = std(lndata, ddof=1 if len(lndata)>1 else 0)
        
    return theta, mu, sigma
    
def get_growth_factor(t, period, percent):
    """ get the periodic growth factor at the current time t """
    
    return power(1 + percent/100, t//period) if period>0 else 1.0
    
def apply_postprocess(data, label, factor=1.0):
    """ apply postprocess to a list of data """
    
    if factor>1.0 or factor<1.0: 
        data = [i*factor for i in data]
    
    return data if label not in POSTPROCESS \
           else [eval(POSTPROCESS[label].format("i")) for i in data]
           
def auto_typing(data, label):
    """ convert a list of strings to known type based on the label """
    
    if label not in DATATYPE:
        return data
    
    conv = DATATYPE[label]
    return [conv(i) for i in data]
    
