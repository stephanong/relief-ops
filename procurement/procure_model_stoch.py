""" Stochastic programming model for rice procurement
    
    provided functions: 
    initiate_model()   - construct the model
    optimise_all()     - solve the model optimizing all the objectives
    get_solution()     - get the solution, ie. buy/order quantities
    get_objective()    - get the objective values
"""

from docplex.mp.model import Model
from procure_model_det import optimise_all, is_optimal, get_solution, get_objective

from utils import BUDGETCONSTRSCALE

## -- modelling with Docplex

def initiate_model(n, tau, b, r, t, 
                   scen, prob=[], l=[], 
                   write=""): 
    """ build the stochastic model for procurement
    
        :param int n: length of the planning horizon
        :param int tau: maximum age of the item
        :param float b: total available budget
        :param int r: quantity security stock required
        :param int t: maximum age of the security stock
        :param list scen: list of scenario data, 
                          each element of a scenario is a list of triplets (demand, price, avail), 
                          if couples are provided instead, availability is ignored
        :param list prob: probability of each scenario (assumed equiprobable if empty)
        :param list l: initial inventory level (use r and t if empty)
        :param str write: filename to write the model to (do nothing if empty)
        
        :return: a docplex.mp.model.Model object
    """
        
    # create docplex MIP  model
    model = Model("budgetalloc-stoch")
    
    # store model parameters
    N = len(scen)
    model.n, model.tau, model.N = n, tau, N
    
    # define y variables
    model.var_y = y = model.continuous_var_dict([i+1 for i in range(n)], name="y")    
    if len(scen[0][0])>=3: # respect the availabilities 
        for i in range(n):
            y[i+1].set_ub(min([scen[k][i][2] for k in range(N)]))
    
    # define x variables
    model.var_x = x = model.continuous_var_dict([(i,j+1,k+1) 
                                              for i in range(n+1)
                                              for j in range(tau)
                                              for k in range(N)], name="x")
    for k in range(N):
        for j in range(tau-1):
            val = l[j] if l else (r if j+1==t else 0)
            x[0,j+1,k+1].set_lb(val)
            x[0,j+1,k+1].set_ub(val)

    # define auxiliary variables on the shortage and consumption
    model.var_s = s = model.continuous_var_dict([(i+1,k+1)
                                                 for i in range(n+1)
                                                 for k in range(N)], name="s")
    model.var_c = c = model.continuous_var_dict([(i+1,j,k+1)
                                                 for i in range(n)
                                                 for j in range(tau)
                                                 for k in range(N)], name="c")
                                                
    # set objectives
    model.obj_shortage = expt_shortage = model.sum( (prob[k] if prob else 1/N)*model.sum(s[i+1,k+1] for i in range(n+1)) for k in range(N) )
    model.obj_waste = expt_waste = model.sum( (prob[k] if prob else 1/N)*model.sum(x[i+1,tau,k+1] for i in range(n)) for k in range(N) )
    model.obj_budget = expt_budget = model.sum( (prob[k] if prob else 1/N)*model.sum(scen[k][i][1]*y[i+1] for i in range(n)) for k in range(N) )
    
    model.lstobj = [(model.minimize, expt_shortage),
                    (model.minimize, expt_waste),
                    (model.minimize, expt_budget)]

    # set balance contraints
    model.add_constraints(
        (y[i+1] if j==0 else x[i,j,k+1]) == c[i+1,j,k+1] + x[i+1,j+1,k+1] 
        for i in range(n)
        for j in range(tau)
        for k in range(N) )
        
    # set demand-shortage constraints
    model.add_constraints(
        s[i+1,k+1] + model.sum(c[i+1,j,k+1] for j in range(tau)) == scen[k][i][0] 
        for i in range(n)
        for k in range(N) )

    # set the safety stock constraints
    model.add_constraints(
        s[n+1,k+1] + model.sum(x[n,j+1,k+1] for j in range(t)) >= r
        for k in range(N) )
        
    # set budget constraints
    model.add_constraints(
        model.sum((scen[k][i][1]*BUDGETCONSTRSCALE)*y[i+1] for i in range(n)) <= b*BUDGETCONSTRSCALE
        for k in range(N) )
    
    # save the model to file
    if write: model.export_as_lp(write)
    
    return model

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

def test_model_stoch():
    """ test the robust model """   
    
    from numpy.random import randint, poisson    

#-- to generate a minimal model that can cause numerical difficulty if decisions are non-integer
#    n, tau, b = 1, 2, 12593252445.75
#    r, t = 0, 1
#
#    ntest, N = 10, 2
#    
#    for i in range(ntest): 
#        print()
#        
#        # create the scenarios
#        scen = []
#        for i in range(N):
#            d = [randint(0,1000)]
#            p = [  6130.58]  
    
    n, tau, b = 6, 3, 100000
    r, t = 200, 1
    
    ntest, N = 10, 100
    
    for i in range(ntest): 
        print()
        
        # create the scenarios
        scen = []
        for i in range(N):
            d = [randint(0,150), randint(0,250), randint(0,130), randint(0,540), randint(0,1060), randint(0,430)]
            p = [  poisson(375),   poisson(450),   poisson(125),   poisson(115),    poisson(300),   poisson(220)]
            a = [  poisson(200),   poisson(200),  poisson(1600),   poisson(600),    poisson(400),   poisson(160)]

#            scen.append(list(zip(d, p, a)))
            scen.append(list(zip(d, p)))

        
        # test the robust model
        model = initiate_model(n, tau, b, r, t, scen, write="test_procure_model_stoch")
        optimise_all(model, write="test_procure_model_stoch")
        solution, objective = get_solution(model), get_objective(model)

        print("solution = {}, expected (shortage = {}, waste = {}, used budget = {})".format(
               solution, objective[0], objective[1], objective[2]))
        
        assert(is_optimal(model))
    
