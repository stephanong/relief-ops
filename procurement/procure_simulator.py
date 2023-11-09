#!/usr/bin/env python3
""" Simulation loop for testing rice procurement models """

import csv
from copy import copy

from utils import exec_and_quit, exec_timed,  \
    floor, ceil, round, mean, percentile, month_to_index, index_to_month, \
    sample_uniform_int, \
    get_growth_factor, apply_postprocess
from utils import Config, STSRecording


from procure_model_det import initiate_model as initiate_det
from procure_model_stoch import initiate_model as initiate_stoch
from procure_model_stochcvar import initiate_model as initiate_stochcvar
from procure_model_stochcvarw import initiate_model as initiate_stochcvarw
from procure_model_affine import initiate_model as initiate_affine
from procure_model_affinecvar import initiate_model as initiate_affinecvar

from procure_model_det import optimise_all, get_solution
from procure_model_affine import get_solution as get_affine

## -- Optimisers (wrappers around the developed models)

class DummyOptimiser:
    """ Dummy optimiser that acts randomly, but balances the remained budget, 
        this is for testing purpose only
    """ 
    
    def __init__(self, conf, hist):
        """ constructor """
        
        self.conf, self.hist = conf, hist
        
    def decide(self, system): 
        """ decide and recommend the next buy quantity for a given system """
        
        return sample_uniform_int(0, 
            2*system.budget[-1]/((self.conf["length"]-system.itime)*system.price[-1]))

class ClairvoyantOptimiser(DummyOptimiser):
    """ Clairvoyant optimiser that has access to the future 
        this is for theoretical purpose only
    """ 

    leaked_info = { "demand": [], "price": [] }

    def __init__(self, conf):
        """ constructor """
        
        self.conf = conf
    
    def decide(self, system): 
        """ decide and recommend the next buy quantity for a given system """
        
        n, skip = conf["length"], conf["skip"]
        nleaked = len(ClairvoyantOptimiser.leaked_info["demand"])         
        
        # better done this in the simulator, also this will improve its code
        # here if we ask for running simulation longer than the available scenario data
        # the first year will be repeated periodically
        demand = [ClairvoyantOptimiser.leaked_info["demand"][skip+i if skip+i<nleaked else (skip+i)%12] for i in range(n)]
        price = [ClairvoyantOptimiser.leaked_info["price"][skip+i if skip+i<nleaked else (skip+i)%12] for i in range(n)]
        
        model = initiate_det(conf["length"] - system.itime, # remained time
                             conf["storage_limit_age"], 
                             system.budget[-1], # remained budget
                             conf["safety_stock"], 
                             conf["safety_stock_limit_age"], 
                             list(zip(demand[system.itime:], price[system.itime:])), 
                             l=system.inventory)
        optimise_all(model)
        
        return get_solution(model)[0]         

class BasicOptimiser(DummyOptimiser):
    """ Basic predictive optimiser uses average values of historical data """
    
    def __init__(self, conf, hist):
        """ constructor """
        
        super().__init__(conf, hist)
        
        # to recover index from label
        label_to_index = { l:i for i, l in enumerate(hist.label) }
        
        # build the prediction for each month
        np = min(hist.n, 12)
        periodic_predict = {
            label:[self.predict_func(label,
                        [hist.detail[j][label_to_index[label]][i] 
                         for j in range(len(hist.detail))])
                   for i in range(np)]
            for label in hist.label}
                
        # build the prediction for the whole running length
        n, start = conf["length"], conf["skip"]
        self.predict = {
            label:apply_postprocess([periodic_predict[label][(start+i)%12]
                                     *get_growth_factor(start+i,*conf["optimiser"]["trend"][label])                    
                                     for i in range(n)], label)
            for label in hist.label
        }
        
        # convert and adjust
        self.predict["demand"] = apply_postprocess([
            v*conf["refugee_to_demand"]
            for v in self.predict["refugee"]], "demand")
        self.predict["price"] = [max(v, conf["min_govt_buy_price"]) 
                                 for v in self.predict["price"]]

    def predict_func(self, label, data):
        """ predict function for a given labelled data, here using the mean """
        
        return mean(data)

    def decide(self, system):
        """ decide next buy quantity based on the prediction """
        
        model = initiate_det(conf["length"] - system.itime, # remained time
                             conf["storage_limit_age"], 
                             system.budget[-1], # remained budget
                             conf["safety_stock"], 
                             conf["safety_stock_limit_age"], 
                             list(zip(self.predict["demand"][system.itime:], 
                                      [system.price[-1]]+self.predict["price"][system.itime+1:])), 
                             l=system.inventory)
        optimise_all(model)
        
        return get_solution(model)[0] 
        
class PercentileOptimiser(BasicOptimiser):
    """ Optimiser that takes percentiles of historical data as predictive values """
    
    def predict_func(self, label, data):
        """ use percentile as prediction """
        
        return percentile(data, 
                          conf["optimiser"]["percentile"][label], 
                          interpolation="higher")
    

class StochasticOptimiser(DummyOptimiser):
    """ Optimiser based on stochastic programming """
    
    def __init__(self, conf, hist):
        """ constructor """
        
        super().__init__(conf, hist)
        
        # to recover index from label
        label_to_index = { l:i for i, l in enumerate(hist.label) }
        
        # perhaps, to see if it is better to do the trending in the simulator or in the generator
        # the reason is that the hi-res generator has its own trending
        
        # build the (demand,price)-scenarios for the whole running length 
        self.scen, n, start = [], conf["length"], conf["skip"]
        for j in range(len(hist.detail)):
            d = apply_postprocess([hist.detail[j][label_to_index["refugee"]][(start+i)%12]
                                   *get_growth_factor(start+i,*conf["optimiser"]["trend"]["refugee"])
                                   for i in range(n)],
                                   label="demand",factor=conf["refugee_to_demand"])
            p = apply_postprocess([max(hist.detail[j][label_to_index["price"]][(start+i)%12],conf["min_govt_buy_price"])
                                       *get_growth_factor(start+i,*conf["optimiser"]["trend"]["price"])  
                                   for i in range(n)], label="price")
            self.scen.append(list(zip(d, p)))
            
    def decide(self, system):
        """ decide next buy quantity """

        model = initiate_stoch(conf["length"] - system.itime, # remained time
                     conf["storage_limit_age"], 
                     system.budget[-1], # remained budget
                     conf["safety_stock"], 
                     conf["safety_stock_limit_age"], 
                     [[(s[system.itime][0],system.price[-1])]+s[system.itime+1:] 
                      for s in self.scen], 
                     l=system.inventory)
        
        optimise_all(model)
        
        return get_solution(model)[0] 

class StochasticCVaROptimiser(StochasticOptimiser):
    """ Optimiser based on stochastic programming but using CVaR for shortage """
        
    def decide(self, system):
        """ decide next buy quantity """

        model = initiate_stochcvar(conf["length"] - system.itime, # remained time
                     conf["storage_limit_age"], 
                     system.budget[-1], # remained budget
                     conf["safety_stock"], 
                     conf["safety_stock_limit_age"], 
                     [[(s[system.itime][0],system.price[-1])]+s[system.itime+1:] 
                      for s in self.scen], 
                     alpha=conf["optimiser"]["stochcvar"]["alpha"],
                     l=system.inventory)
        
        optimise_all(model)
        
        return get_solution(model)[0] 

class StochasticCVaRWOptimiser(StochasticOptimiser):
    """ Optimiser based on stochastic programming but using CVaR for waste """
        
    def decide(self, system):
        """ decide next buy quantity """
        
        model = initiate_stochcvarw(conf["length"] - system.itime, # remained time
                     conf["storage_limit_age"], 
                     system.budget[-1], # remained budget
                     conf["safety_stock"], 
                     conf["safety_stock_limit_age"], 
                     [[(s[system.itime][0],system.price[-1])]+s[system.itime+1:] 
                      for s in self.scen], 
                     alpha=conf["optimiser"]["stochcvar"]["alpha"],
                     l=system.inventory)
        
        optimise_all(model)
        
        return get_solution(model)[0] 

class AffineRobustOptimiser(StochasticOptimiser):
    """ Robust optimiser with affine rules """
    
    def __init__(self, conf, hist):
        """ constructor """
        
        super().__init__(conf, hist)
        
        self.v0, self.vd, self.vp, self.va = [], [], [], []
    
    def decide(self, system):
        """ decide next buy quantity """
        
        # solve the model once
        if not self.v0:
            model = initiate_affine(conf["length"] - system.itime, 
                         conf["storage_limit_age"], 
                         system.budget[-1], 
                         conf["safety_stock"], 
                         conf["safety_stock_limit_age"], 
                         self.scen, 
                         l=system.inventory)
            
            # here pick dual-simplex method to avoid numerical instabilities
            model.parameters.lpmethod.set(2)
            
            optimise_all(model)
            
            self.v0, self.vd, self.vp, self.va = get_affine(model)
         
        # compute the next buy qty
        buy = self.v0[0] + \
              sum(v*c for v,c in zip(system.demand,self.vd)) + \
              sum(v*c for v,c in zip(system.price,self.vp)) + \
              sum(v*c for v,c in zip(system.avail,self.va))
        
        return max(buy, 0)
        
## -- Simulator

class System:
    """ System states """
    
    def __init__(self, conf):
        """ constructor """
        
        self.conf = conf
        
        self.price, self.demand, self.avail = None, None, None
        self.bought, self.budget, self.itime = None, None, None
        self.shortage, self.waste, self.consume = None, None, None
        self.inventory = None
        
    def start(self, next_price, next_avail=None, inventory=None):
        """ start the system states """
        
        self.demand, self.price, self.avail = \
            [], [next_price], [next_avail] if next_avail else []
        self.bought, self.budget, self.itime = [], [conf["budget"]], 0
        self.shortage, self.waste, self.consume = [], [], []
        self.inventory = inventory if inventory \
            else [conf["safety_stock"]] + [0]*(conf["storage_limit_age"]-1) 
        self.usable = []
        
    def progress(self, to_buy, demand, next_price, next_avail=None):
        """ update the state of the system and move to the next time step """
                
        # check for sufficient budget and availability
        bought = int(min(to_buy, 
                         floor(self.budget[-1]/self.price[-1]), 
                         self.avail[-1] if self.avail else float("inf")))
        self.bought.append(bought)
        self.budget.append(round(self.budget[-1] - bought*self.price[-1],2)) 
        
        # record the new price
        self.price.append(next_price)
        if next_avail: self.avail.append(next_avail)
        self.waste.append(self.inventory[-1])
        
        # consume the inventory using FIFO
        shortage, consume = demand, 0
        self.inventory = [bought] + self.inventory[:-1]
        self.usable.append(sum(self.inventory[:-1]))        
        for i in range(len(self.inventory)-1,-1,-1):
            if shortage==0: break
            c = min(shortage, self.inventory[i])
            self.inventory[i] -= c
            shortage -= c
            consume += c        
        self.demand.append(demand)
        self.consume.append(consume)
        self.shortage.append(shortage)
        
        self.itime += 1    

class Simulator:
    """ Generation of scenario set """
    
    def __init__(self, conf, opt, test):
        """ constructor """
        
        self.conf, self.opt, self.test = conf, opt, test
    
    def run(self):
        """ run the simulation loop on each test scenario """
        
        # to recover the index
        label_to_index = { l:i for i, l in enumerate(self.test.label) }
        
        # prepare the system, output
        n, skip = self.conf["length"], self.conf["skip"]        
        system = System(conf)
        
        output = STSRecording(n+1, 
            (self.test.start + skip)%12,
            ["bought", "usable", "demand", "consume", 
             "shortage", "waste", "budget"])
        
        # simulate over test scenarios
        for scen in self.test.detail: 
                        
            # prepare demand, price, and availability
            demand = apply_postprocess(scen[label_to_index["refugee"]], 
                "demand", factor=self.conf["refugee_to_demand"])
            
            price = [max(i, self.conf["min_govt_buy_price"])
                      for i in scen[label_to_index["price"]]]
            
            avail = scen[label_to_index["avail"]] if "avail" in label_to_index else []
            
            # !! leak info to ClairvoyantOptimiser
            ClairvoyantOptimiser.leaked_info["demand"] = demand
            ClairvoyantOptimiser.leaked_info["price"] = price

            # initialise the system
            if n>0: 
                next_price = price[skip] 
                next_avail = avail[skip] if avail else None 
                system.start(next_price, next_avail)
            
            # for each time step
            for i in range(n):                 
                # a decision is made
                to_buy = apply_postprocess([self.opt.decide(system)], "bought")[0]
                
                # then new data are revealed 
                nexti = skip+i+1 if skip+i+1<self.test.n else (skip+i+1)%12 # assuming annually periodic for long run
                next_price = price[nexti] 
                next_avail = avail[nexti] if avail else None
                
                curi = skip+i if skip+i<self.test.n else (skip+i)%12
                cur_demand = demand[curi]
                
                system.progress(to_buy, cur_demand, next_price, next_avail)
            
            # unmet safety stock as the last shortage
            unmet_safety = max(self.conf["safety_stock"] - 
                               sum(system.inventory[
                                :self.conf["safety_stock_limit_age"]]), 0)
            
            output.detail.append(
                [ system.bought+[0], 
                  system.usable+[system.usable[-1]], 
                  system.demand+[self.conf["safety_stock"]], 
                  system.consume+[0],
                  system.shortage+[unmet_safety], 
                  system.waste+[0], 
                  system.budget[1:]+[system.budget[-1]] ])
        
#        print()
#        for res in output.detail:
#            print(sum(res[-3]),sum(res[-2]),self.conf["budget"]-res[-1][-1])
                
        return output

## -- Main program

if __name__ == "__main__": 

    # function wrappers

    def load_config(filename):
        return Config(filename).simulator
    
    def load_input(histfile, testfile=None):
        hist,test = STSRecording(), STSRecording()
        
        hist.read_csv(histfile)        
        if testfile is None:
            return hist, None
            
        test.read_csv(testfile)        
        return hist, test
        
    def create_optimiser(opt, conf, hist):        
        if opt=="basic":
            return BasicOptimiser(conf, hist)
        if opt=="percentile":
            return PercentileOptimiser(conf, hist)
        if opt=="stoch":
            return StochasticOptimiser(conf, hist)
        if opt=="stochcvar":
            return StochasticCVaROptimiser(conf, hist)            
        if opt=="stochcvarw":
            return StochasticCVaRWOptimiser(conf, hist)
        if opt=="affine":
            return AffineRobustOptimiser(conf, hist) 
        if opt=="clairvoyant":
            return ClairvoyantOptimiser(conf)            
        
        return DummyOptimiser(conf, hist) # otherwise
        
    from argparse import ArgumentParser, RawTextHelpFormatter 
    conf = Config().simulator
    
    parser = ArgumentParser(
        description="Simulator tool to evalute optimisation models. "
                    "This software is free and wild!\n"
                    "Relief-OpS project (c) 2021.\n",
        formatter_class=RawTextHelpFormatter,
        add_help=False)
        
    ag = parser.add_argument_group("required arguments") 
    ag.add_argument("-i", "--input", nargs="+", help="input historical and test scenarios, providing only one set will trigger the self-test mode") 
    ag.add_argument("-o", "--output", help="filename of the output")

    ag = parser.add_argument_group("optional arguments")
    known_opt = set(["basic", "percentile", "stoch", "stochcvar", "stochcvarw", "affine", "clairvoyant", "dummy"])
    ag.add_argument("-r", "--run", default="dummy",
        help="choose optimiser among {basic, percentile, stoch, stochcvar, stochcvarw, affine, clairvoyant, dummy} (default=dummy, for testing purpose)")
    ag.add_argument("-c", "--config", 
        help="a json configuration file to overwrite default options")    
    ag.add_argument("-n", "--duration", type=int, 
        help="duration of the simulation (default={})".format(conf["length"]))
    ag.add_argument("-s", "--skip", type=int, 
        help="number of months to skip at the beginning (default={})".format(conf["skip"]))
    ag.add_argument("-b", "--budget", type=float, 
        help="budget for the running simulation (default={})".format(conf["budget"]))
    ag.add_argument("-pr", "--prefugee", type=float, 
        help="percentile of #refugees used in percentile optimiser (default={})".format(conf["optimiser"]["percentile"]["refugee"]))
    ag.add_argument("-pa", "--palpha", type=float, 
        help="alpha used in stochcvar(w) optimiser(s) (default={})".format(conf["optimiser"]["stochcvar"]["alpha"]))
    ag.add_argument("-h", "--help", action="help", help="show this help message and exit") 
       
    args = parser.parse_args()

    if not (args.input and args.output):
        exec_and_quit("please specify the missing input(s) or output\n", 0, parser.print_help)
    
    if len(args.input) not in (1,2):
        exec_and_quit("the number of input files must be either 1 (self-test mode) or 2\n", 0, parser.print_help)

    if args.run not in known_opt:
        exec_and_quit("unsupported optimiser\n", 0, parser.print_help)        
        
    if args.config: # load user-defined configuration
        conf = exec_timed("Loading configuration file", load_config, args.config)
    
    # update with the input arguments
    if args.duration is not None: conf["length"] = args.duration
    if args.skip is not None: conf["skip"] = args.skip
    if args.budget is not None: conf["budget"] = args.budget
    if args.prefugee is not None: conf["optimiser"]["percentile"]["refugee"] = args.prefugee
    if args.palpha is not None: conf["optimiser"]["stochcvar"]["alpha"] = args.palpha
        
    # start simulating the rolling horizon
    hist, test = exec_timed("Loading input files", load_input, *args.input)
        
    res = STSRecording()
    if test: # normal mode
        opt = exec_timed("Preparing optimiser [{}]".format(args.run), create_optimiser, args.run, conf, hist)
        sim = Simulator(conf, opt, test)
        
        res = exec_timed("Running the simulation", sim.run)
    else: # self-test mode
        N = len(hist.detail)
        for i in range(N):
            ch, ct = STSRecording(), STSRecording()
            ch.n = ct.n = hist.n
            ch.label = ct.label = hist.label
            ch.start = ct.start = hist.start
            for j in range(N):
                if j!=i:
                    ch.detail.append(hist.detail[j])
                else:
                    ct.detail.append(hist.detail[j])
            
            opt = exec_timed("Preparing optimiser [{}]".format(args.run), create_optimiser, args.run, conf, ch)
            sim = Simulator(conf, opt, ct)
            
            resi = exec_timed("Running the simulation", sim.run)
            
            if i>0:
                res.detail.append(resi.detail[0])
            else:
                res.n, res.start, res.label = resi.n, resi.start, resi.label
                res.detail = resi.detail
    
    exec_timed("Writing the results", res.write_csv, args.output)
    
