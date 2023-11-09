""" Ajustable robust model for rice procurement using affine rules
    
    provided functions: 
    initiate_model()   - construct the model
    optimise_all()     - solve the model optimizing all the objectives
    get_solution()     - get the solution, ie. buy/order quantities
    get_objective()    - get the objective values
"""

from docplex.mp.model import Model
from procure_model_stoch import optimise_all, is_optimal, get_objective

from utils import BUDGETCONSTRSCALE

## -- modelling with Docplex

def initiate_model(n, tau, b, r, t, 
                   scen, l=[], 
                   write=""): 
    """ build the approximation robust model for procurement using affine rules
    
        :param int n: length of the planning horizon
        :param int tau: maximum age of the item
        :param float b: total available budget
        :param int r: quantity security stock required
        :param int t: maximum age of the security stock
        :param list scen: list of scenario data, 
                          each element of a scenario is a list of triplets (demand, price, avail), 
                          if couples are provided instead, availability is ignored
        :param list l: initial inventory level (use r and t if empty)
        :param str write: filename to write the model to (do nothing if empty)
        
        :return: a docplex.mp.model.Model object
    """
    
    # create docplex MIP  model
    model = Model("budgetalloc-affine")
    
    # store model parameters
    N = len(scen)
    model.n, model.tau, model.N = n, tau, N
        
    # define variables for the affine rule
    model.var_v0 = v0 = model.continuous_var(lb=-model.infinity, name="v0")
    model.var_vd = vd = model.continuous_var_dict([i+1 for i in range(n-1)], lb=-model.infinity, name="vd")
    model.var_vp = vp = model.continuous_var_dict([i+1 for i in range(n)], lb=-model.infinity, name="vp")
    model.var_va = va = model.continuous_var_dict([i+1 for i in range(n)], lb=-model.infinity, name="va")
    
    # define y variables
    model.var_y = y = model.continuous_var_dict([(i+1,k+1) 
                                              for i in range(n)
                                              for k in range(N)], name="y")    
    if len(scen[0][0])>=3: # respect the availabilities 
        for i in range(n):
            for k in range(N):
                y[i+1,k+1].set_ub(min([scen[k][i][2] for k in range(N)]))
    
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

    # define auxiliary variables for the objectives
    model.obj_shortage = S = model.continuous_var(name="S")
    model.obj_waste = W = model.continuous_var(name="W")
    model.obj_budget = B = model.continuous_var(name="B")
                                                    
    # set the objectives    
    model.lstobj = [(model.minimize, S), 
                    (model.minimize, W), 
                    (model.minimize, B)]

    # affine-rule constraints
    model.add_constraints(
        y[i+1,k+1] == v0 + model.sum(scen[k][ii][0]*vd[ii+1] for ii in range(i))
                         + model.sum(scen[k][ii][1]*vp[ii+1] for ii in range(i+1))
                         + (model.sum(scen[k][ii][2]*va[ii+1] for ii in range(i+1)) if len(scen[0][0])>=3 else 0)
        for i in range(n)
        for k in range(N))
#    model.add_constraints(
#        y[i+1,k+1] <= 1 + v0 + model.sum(scen[k][ii][0]*vd[ii+1] for ii in range(i))
#                         + model.sum(scen[k][ii][1]*vp[ii+1] for ii in range(i+1))
#                         + (model.sum(scen[k][ii][2]*va[ii+1] for ii in range(i+1)) if len(scen[0][0])>=3 else 0)
#        for i in range(n)
#        for k in range(N))

    # define worst-scenario constraints
    model.add_constraints(
        model.sum(s[i+1,k+1] for i in range(n+1)) <= S
        for k in range(N) )
    model.add_constraints(
        model.sum(x[i+1,tau,k+1] for i in range(n)) <= W
        for k in range(N) )
    model.add_constraints(
        model.sum(scen[k][i][1]*y[i+1,k+1] for i in range(n)) <= B
        for k in range(N) )

    # set balance contraints
    model.add_constraints(
        (y[i+1,k+1] if j==0 else x[i,j,k+1]) == c[i+1,j,k+1] + x[i+1,j+1,k+1] 
        for i in range(n)
        for j in range(tau)
        for k in range(N) )
        
    # set demand-shortage constraints
    model.add_constraints(
        s[i+1,k+1] + model.sum(c[i+1,j,k+1] for j in range(tau)) == scen[k][i][0] 
        for i in range(n)
        for k in range(N) )

    # set safety stock constraints
    model.add_constraints(
        s[n+1,k+1] + model.sum(x[n,j+1,k+1] for j in range(t)) >= r        
        for k in range(N) )
        
    # set budget constraints
    model.add_constraints(
        model.sum((scen[k][i][1]*BUDGETCONSTRSCALE)*y[i+1,k+1] for i in range(n)) <= b*BUDGETCONSTRSCALE
        for k in range(N) )
    
    # save the model to file
    if write: model.export_as_lp(write)
#    if write: model.export_as_mps(write)
    
    return model

def get_solution(model):
    """ get the current solution which are the weights of the affine rule 
    
        :param docplex.mp.model.Model model: the current model
        :return: list of sequences of weights
    """
    
    if not is_optimal(model): return []
    
    return ([model.var_v0.solution_value], 
            [model.var_vd[i+1].solution_value for i in range(model.n-1)],
            [model.var_vp[i+1].solution_value for i in range(model.n)],
            [model.var_va[i+1].solution_value for i in range(model.n)])

## -- testing with pytest-3

def test_model_affine():
    """ test the robust model """   
        
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

            scen.append(list(zip(d, p, a)))
#            scen.append(list(zip(d, p)))    
    
        model = initiate_model(n, tau, b, r, t, scen)
        optimise_all(model, write="test_procure_model_affine") #, log_output=True)
        assert(is_optimal(model))
        solution, objective = get_solution(model), get_objective(model) 
        print("solution = {}, worst (shortage = {}, waste = {}, used budget = {})".format(
               solution, objective[0], objective[1], objective[2]))
    
    assert(is_optimal(model))
    
