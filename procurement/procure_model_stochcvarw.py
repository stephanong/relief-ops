""" Stochastic programming model for rice procurement with CVaR-shortage
    
    provided functions: 
    initiate_model()   - construct the model
    optimise_all()     - solve the model optimizing all the objectives
    get_solution()     - get the solution, ie. buy/order quantities
    get_objective()    - get the objective values
"""

from docplex.mp.model import Model
from procure_model_stoch import optimise_all, is_optimal, get_solution, get_objective

from utils import BUDGETCONSTRSCALE

## -- modelling with Docplex

def initiate_model(n, tau, b, r, t, 
                   scen, prob=[], alpha=0.95, l=[], 
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
        :param float alpha: confidence level of CVaR for the waste objective
        :param list l: initial inventory level (use r and t if empty)
        :param str write: filename to write the model to (do nothing if empty)
        
        :return: a docplex.mp.model.Model object
    """
    
    # create docplex MIP  model
    model = Model("budgetalloc-stochcvar")
    
    # store model parameters
    N = len(scen)
    model.n, model.tau, model.N, model.alpha = n, tau, N, alpha
    
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
            
    # define auxiliary variables to compute cvar
    model.vareta = eta = model.continuous_var(name="eta", lb=-model.infinity)
    model.varu = u = model.continuous_var_dict([k+1 for k in range(N)], name="u")
                                                       
    # set objectives
    model.obj_shortage = expt_shortage = model.sum( (prob[k] if prob else 1/N)*model.sum(s[i+1,k+1] for i in range(n+1)) for k in range(N) )
    model.obj_waste = cvar_waste = eta + (1/(1-alpha))*model.sum( (prob[k] if prob else 1/N)*u[k+1] for k in range(N) )
    model.obj_budget = expt_budget = model.sum( (prob[k] if prob else 1/N)*model.sum(scen[k][i][1]*y[i+1] for i in range(n)) for k in range(N) )
    
    model.lstobj = [(model.minimize, expt_shortage),
                    (model.minimize, cvar_waste),
                    (model.minimize, expt_budget)]


    # model parts that exceed VaR(alpha)
    model.add_constraints(
        u[k+1] >= model.sum(x[i+1,tau,k+1] for i in range(n)) - eta 
        for k in range(N) )

    # balance contraints
    model.add_constraints(
        (y[i+1] if j==0 else x[i,j,k+1]) == c[i+1,j,k+1] + x[i+1,j+1,k+1] 
        for i in range(n)
        for j in range(tau)
        for k in range(N) )
        
    # demand-shortage constraints
    model.add_constraints(
        s[i+1,k+1] + model.sum(c[i+1,j,k+1] for j in range(tau)) == scen[k][i][0] 
        for i in range(n)
        for k in range(N) )

    # safety stock constraints
    model.add_constraints(
        s[n+1,k+1] + model.sum(x[n,j+1,k+1] for j in range(t)) >= r
        for k in range(N) )
        
    # budget constraints
    model.add_constraints(
        model.sum((scen[k][i][1]*BUDGETCONSTRSCALE)*y[i+1] for i in range(n)) <= b*BUDGETCONSTRSCALE
        for k in range(N) )
    
    # save the model to file
    if write: model.export_as_lp(write)
    
    return model

## -- testing with pytest-3

def test_model_stochcvar():
    """ test the robust model """   
    
#    from model_base import initiate_model as initiate_base
    from procure_model_stoch import initiate_model as initiate_stoch
    from numpy.random import seed, randint, poisson

    n, tau, b = 6, 3, 1000000
    r, t = 200, 1
    
    ntest, N = 10, 100

    seed(0)    
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
        
        # test the robust models
        alpha = 0.95
        model = initiate_model(n, tau, b, r, t, scen, alpha=alpha)
        optimise_all(model, write="test_procure_model_stochcvar")
        assert(is_optimal(model))        
        solution, objective = get_solution(model), get_objective(model)
        print("solution = {}, expected shortage = {}, cvar-{} waste = {}, expected budget = {})".format(
               solution, objective[0], alpha, objective[1], objective[2]))
        
        alpha = 0.0
        model = initiate_model(n, tau, b, r, t, scen, alpha=alpha)
        optimise_all(model)
        assert(is_optimal(model))
        solution, objective = get_solution(model), get_objective(model)
        print("solution = {}, expected shortage = {}, cvar-{} waste = {}, expected budget = {})".format(
               solution, objective[0], alpha, objective[1], objective[2]))
        
        # test the average model (equivalent to alpha = 0)
        model = initiate_stoch(n, tau, b, r, t, scen)
        optimise_all(model)
        solution, objective = get_solution(model), get_objective(model)
        assert(is_optimal(model))
        print("solution = {}, expected (shortage = {}, waste = {}, budget = {})".format(
               solution, objective[0], objective[1], objective[2]))
        
        print()
