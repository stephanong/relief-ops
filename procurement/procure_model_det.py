""" Deterministic model for rice procurement 
    
    provided functions: 
    initiate_model()   - construct the model
    optimise_all()     - solve the model optimizing all the objectives
    get_solution()     - get the solution, ie. buy/order quantities
    get_objective()    - get the objective values
"""

from numpy import round
from docplex.mp.model import Model

from utils import EPS, OBJTOL, BUDGETCONSTRSCALE

## -- modelling in DocPLEX

def initiate_model(n, tau, b, r, t, 
                   scen0, l=[], 
                   write="", fast_version=False): 
    """ initiate the procurement model with the first objective (shortage)
    
        :param int n: length of the planning horizon
        :param int tau: maximum age of the item
        :param float b: total available budget
        :param int r: quantity security stock required
        :param int t: maximum age of the security stock
        :param list scen0: scenario data of length n, 
                           each element is a triplets (demand, price, avail), 
                           if couples are provided instead, availability is ignored 
        :param list l: initial inventory level (use r and t if empty)
        :param str write: filename to write the model to (do nothing if empty)        
        :return: a docplex.mp.model.Model object
    """
    
    # create DocPLEX MIP model
    model = Model("budgetalloc-det")
    
    # store model parameters
    model.n, model.tau, model.b, model.r, model.t = n, tau, b, r, t

    # define y variables
    model.var_y = y = model.continuous_var_dict([i+1 for i in range(n)], name="y")    
    if len(scen0[0])>=3: # respect the availabilities 
        for i in range(n):
            y[i+1].set_ub(scen0[i][2])

    # define x variables
    model.var_x = x = model.continuous_var_dict([(i,j+1) for i in range(n+1) 
                                                      for j in range(tau)], name="x")
    for j in range(tau-1):
        val = l[j] if l else (r if j+1==t else 0)
        x[0,j+1].set_lb(val)
        x[0,j+1].set_ub(val)
        
    # define auxiliary variables on the shortage and consumption
    model.var_s = s = model.continuous_var_dict([i+1 for i in range(n+1)], name="s")
    model.var_c = c = model.continuous_var_dict([(i+1,j)
                                                 for i in range(n)
                                                 for j in range(tau)], name="c")

    # define and set the objectives
    model.obj_shortage = shortage = model.sum(s[i+1] for i in range(n+1))
    model.obj_waste = waste = model.sum(x[i+1,tau] for i in range(n))
    model.obj_budget = budget = model.sum(scen0[i][1]*y[i+1] for i in range(n))
    
    if fast_version:
        model.lstobj = [(model.minimize,shortage), 
                        (model.minimize,budget)]
    else:
        model.lstobj = [(model.minimize,shortage), 
                        (model.minimize,waste), 
                        (model.minimize,budget)]
        
    # set balance constraints
    model.add_constraints(
        (y[i+1] if j==0 else x[i,j]) == c[i+1,j] + x[i+1,j+1] 
        for i in range(n)
        for j in range(tau))
    
    # set demand-shortage constraints
    model.add_constraints(
        s[i+1] + model.sum(c[i+1,j] for j in range(tau)) == scen0[i][0] 
        for i in range(n))
    
    # set safety stock constraint
    model.add_constraint(s[n+1] + model.sum(x[n,j+1] for j in range(t)) >= r)
    
    # set the budget constraints
    model.add_constraint(budget*BUDGETCONSTRSCALE <= b*BUDGETCONSTRSCALE)
    
    # save the model to file
    if write: model.export_as_lp(write)
       
    return model

def optimise_all(model, write="", log_output=False):
    """ solve the problem by optimizing all objectives
        
        :param docplex.mp.model.Model model: the current model
        :param str write: filename to write the model to (do nothing if empty)        
        :param bool log_output: enable solving log
    """
    
    # to memorize the last constraint added
    lastconstr = None
        
    # iterate over the objectives
    for (i, obj) in enumerate(model.lstobj):
        
        # set and solve the current objective
        obj[0](obj[1])
        if write: model.export_as_lp(write + "-obj{}.lp".format(i+1))
        model.sol = model.solve(log_output=log_output)
        
        # check optimality
        if not is_optimal(model): 
            
            # if last fixed-objective is too tight, then we can undo it and go back to the previous one
            if lastconstr is not None:            
                model.lstobj[i-1][0](model.lstobj[i-1][1])
                model.remove_constraint(lastconstr)
                model.sol = model.solve(log_output=log_output)
            break
        
        # fix the solved objective as a new constraint
        lastconstr = model.add_constraint(
            (obj[1] <= obj[1].solution_value + OBJTOL) if obj[0]==model.minimize
            else (obj[1] >= obj[1].solution_value + OBJTOL) # OBJTOL is an additional measure to avoid the go-back above if possible
        )
    
def is_optimal(model):
    """ check if the model is solved to optimality
        
        :param docplex.mp.model.Model model: the current model        
        :return: True if it is solved to optimality else False
    """
    
    return (hasattr(model, "sol") 
            and (model.sol is not None) 
            and ("optimal" in model.sol.solve_details.status))

def get_solution(model):
    """ get the solution of a model after it has been optimally solved 
    
        :param docplex.mp.model.Model model: the current model
        :return: list of procurement order for each month
    """
    
    # return an empty solution if the model is not solved to optimality
    if not is_optimal(model): return []
    
    return [model.var_y[i+1].solution_value for i in range(model.n)]

def get_objective(model):
    """ get the objective values of a model after it has been optimally solved 
    
        :param docplex.mp.model.Model model: the current model
        :return: tuple of the three objectives shortage, waste and used budget
    """

    # return an empty list if the model is not solved to optimality
    if not is_optimal(model): return []
    
    return (model.obj_shortage.solution_value, 
            model.obj_waste.solution_value,
            model.obj_budget.solution_value)
    
## -- testing with pytest-3

def test_model_det():
    """ test the model and check the validity of skipping objective-2 """
    
    from numpy.random import randint, poisson
    
    n, tau, b = 6, 3, 100000
    r, t = 400, 1    
    
    ntest = 10
    
    for i in range(ntest):
        print()
    
        # create some data
        d = [randint(0,150), randint(0,250), randint(0,130), randint(0,540), randint(0,1060), randint(0,430)]
        p = [  poisson(375),   poisson(450),   poisson(125),   poisson(115),    poisson(300),   poisson(220)]
        a = [  poisson(200),   poisson(200),  poisson(1600),   poisson(600),    poisson(400),   poisson(160)]
        
        # slow solving
        slowmodel = initiate_model(n, tau, b, r, t, scen0=list(zip(d, p, a)), write="", fast_version=False)
        optimise_all(slowmodel, write="test_procure_model_det_slow")
        assert(is_optimal(slowmodel))
        
        slowsolution = get_solution(slowmodel)
        slowshortage, slowwaste, slowusage = get_objective(slowmodel)
        print("3-obj-solution = {}, shortage = {}, waste = {}, budget usage = {}".format(
               slowsolution, slowshortage, slowwaste, slowusage)) 

        # fast solving  
        fastmodel = initiate_model(n, tau, b, r, t, scen0=list(zip(d, p, a)), write="", fast_version=True)
        optimise_all(fastmodel, write="test_procure_model_det_fast")
        assert(is_optimal(fastmodel))
        
        solution = get_solution(fastmodel)
        shortage, waste, usage = get_objective(fastmodel)
        print("2-obj-solution = {}, shortage = {}, waste = {}, budget usage = {}".format(
               solution, shortage, waste, usage)) 
        
        # check if they are equivalent
        equiv = abs(slowshortage-shortage)<EPS and abs(slowwaste-waste)<EPS and abs(slowusage-usage)<EPS
        
        print("waste objective can be skipped =", "yes" if equiv else "no")    
        # assert(equiv==True) # not true if the safety stock is non-zero
    
        if not equiv: break
