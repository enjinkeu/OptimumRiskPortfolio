import numpy as np
import pandas as pd
import scipy.stats as sc
import pulp
import matplotlib.pyplot as plt


def get_parameterEstimate(series,alpha,mu,std):
    """ 
        This function tests for normality and returns VaR and cVaR 
        after testing for normality and skewness
    """
    test =sc.anderson(sc.norm.ppf((1-alpha),sc.norm.fit(series)))
    if alpha*100== 15:
        critical_value = test[1][0]
    elif alpha*100==10:
        critical_value = test[1][1]
    elif alpha*100==5:
        critical_value = test[1][2]
    elif alpha*100==2.5:
        critical_value = test[1][3]
    elif alpha*100==1:
        critical_value = test[1][4]

    if test[0]>critical_value :
        if sc.skewtest(series)[1]<=alpha:
            p = sc.skewnorm.fit(series)
            VaR = sc.skewnorm.ppf(1-alpha, *p)
            cVaR =(1/(1-alpha)) * sc.skewnorm.expect( lambda y: y, args = (p[0],), loc = p[1], scale = p[2], lb = VaR )
        else:
            VaR = abs(mu - sc.norm.ppf(alpha) * std)
            cVaR =(mu + alpha**-1 * sc.norm.pdf(sc.norm.ppf(alpha))* std)
        

    return VaR,cVaR







def getHistoricalPortfolios(efficientFrontierDF,HistoricaldailyreturnsDF):
    """
        This function returns historical portfolios daily returns for the efficient frontier
        scenarios
    """

    #get the number of assets
    assets = HistoricaldailyreturnsDF.columns.tolist()

    pf=pd.DataFrame([HistoricaldailyreturnsDF.dot(efficientFrontierDF[assets].loc[item])/100 for item in efficientFrontierDF.index]).T
    pf=pf.unstack().reset_index(name = 'returns')
    pf.rename(columns={'level_0':'portfolio'},inplace=True)

    return pf.set_index('portfolio')


def getHistoricalVaR(returnSeries,alpha=.05):
    """
        This function returns historical VaR for panda series of returns
    """
    return np.percentile(returnSeries,alpha*100)

def getHistoricalcVaR(returnSeries,alpha=.05):
    """
        This function returns historical cVaR for panda series of returns
    """
    booleanBelowVaRtest =returnSeries<=getHistoricalVaR(returnSeries,alpha)
    return returnSeries[booleanBelowVaRtest].mean()


def norm_VaR(alpha,mu,std):
    VaR = abs(mu - sc.norm.ppf(alpha) * std)
    cVaR =(mu + alpha**-1 * sc.norm.pdf(sc.norm.ppf(alpha))* std)
    return mu,std,VaR,cVaR
    

def var_parametric(mu, std, distribution='normal', alpha=.05, df=10):
    """
        this function returns parametric VaR for either normal or t-distribution
    """
    # because the distribution is symmetric
    if distribution == 'normal':
        VaR =  abs(mu - sc.norm.ppf(alpha) * std)
    elif distribution == 't-distribution':
        nu = df
        VaR = np.sqrt((nu-2)/nu) *  sc.t.ppf(1-alpha, nu) * std - mu
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return VaR

def cvar_parametric(mu, std, distribution='normal', alpha=.05, df=10):
    if distribution == 'normal':
        CVaR = (mu + alpha**-1 * sc.norm.pdf(sc.norm.ppf(alpha))* std)
    elif distribution == 't-distribution':
        nu = df
        xanu = sc.t.ppf(alpha, nu)
        CVaR = -1/(alpha) * (1-nu)**(-1) * (nu-2+xanu**2) * sc.t.pdf(xanu, nu) * std - mu
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return CVaR



def MeanCVaR(mu,mu_b,scen,scen_b,max_weight=1,min_weight=None,cvar_alpha=0.05):
        
    """ This function finds the optimal enhanced index portfolio according to some benchmark. The portfolio corresponds to the tangency portfolio where risk is evaluated according to the CVaR of the tracking error. The model is formulated using fractional programming.
    
    Parameters
    ----------
    mu : pandas.Series with float values
        asset point forecast
    mu_b : pandas.Series with float values
        Benchmark point forecast
    scen : pandas.DataFrame with float values
        Asset scenarios
    scen_b : pandas.Series with float values
        Benchmark scenarios
    max_weight : float
        Maximum allowed weight    
    cvar_alpha : float
        Alpha value used to evaluate Value-at-Risk one
    min_val : float    
        Values less than this are set to zero. This is used to eliminate floating points    
    
    
    Returns
    -------
    float
        Asset weights in an optimal portfolio
        
    Notes
    -----
    The tangency mean-CVaR portfolio is effectively optimized as:
        .. math:: STARR(w) = frac{w^T*mu - mu_b}{CVaR(w^T*scen - scen_b)}
    This procedure is described in [1]_
       
    References
    ----------
        .. [1] Stoyanov, Stoyan V., Svetlozar T. Rachev, and Frank J. Fabozzi. "Optimal financial portfolios." Applied Mathematical Finance 14.5 (2007): 401-436.
       
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> mean = (0.08, 0.09)
    >>> cov = [[0.19, 0.05], [0.05, 0.18]]
    >>> scen = pd.dataframe(np.random.multivariate_normal(mean, cov, 100))
    >>> portfolio = IndexTracker(pd.Series(mean), pd.Series(0), pd.dataframe(scen),pd.Series(0,index=scen.index) )
    """
    
    
    # define index
    i_idx = mu.index
    j_idx = scen.index
    
    # number of scenarios
    N = scen.shape[0]    
    M = 100000 # large number
    
    # define variables
    x = pulp.LpVariable.dicts("x", ( (i) for i in i_idx ),
                                         lowBound=0,
                                         cat='Continuous') 
    
    # loss deviation
    VarDev = pulp.LpVariable.dicts("VarDev", ( (t) for t in j_idx ),
                                         lowBound=0,
                                         cat='Continuous')
    
    # scaling variable used by fractional programming formulation
    Tau = pulp.LpVariable("Tau",   lowBound=0,
                                         cat='Continuous')
    
    # value at risk
    VaR = pulp.LpVariable("VaR",   lowBound=0,
                                         cat='Continuous')
            
    b_z = pulp.LpVariable.dicts("b_z",   ( (i) for i in i_idx ),
                                         cat='Binary')
    
    # auxiliary to make a non-linear formulation linear (see Linear Programming Models based on Omega Ratio for the Enhanced Index Tracking Problem) 
    Tau_i = pulp.LpVariable.dicts("Tau_i", ( (i) for i in i_idx ),
                                         lowBound=0,
                                         cat='Continuous')
    
    
    #####################################
    ## define model
    model = pulp.LpProblem("Enhanced Index Tracker (STAR ratio)", pulp.LpMaximize)
     
    #####################################
    ## Objective Function
             
    model += pulp.lpSum([mu[i] * x[i] for i in i_idx] ) - Tau*mu_b
                      
    #####################################
    # constraint
                      
    # calculate CVaR
    for t in j_idx:
        model += -pulp.lpSum([ scen.loc[t,i]  * x[i] for i in i_idx] ) + scen_b[t]*Tau - VaR <= VarDev[t]
    
    model += VaR + 1/(N*cvar_alpha)*pulp.lpSum([ VarDev[t] for t in j_idx]) <= 1    
    
    ### price*number of products cannot exceed budget
    model += pulp.lpSum([ x[i] for i in i_idx]) == Tau    
                                                
    ### Concentration limits
    # set max limits so it cannot not be larger than a fixed value  
    ###
    for i in i_idx:
        model += x[i] <= max_weight*Tau
    
    ### Add minimum weight constraint, either zero or atleast minimum weight
    if min_weight is not None:
    
        for i in i_idx:                           
            model += x[i] >= min_weight*Tau_i[i]
            model += x[i] <= max_weight*Tau_i[i]
            model += Tau_i[i] <= M*b_z[i]
            model += Tau_i[i] <= Tau
            model += Tau - Tau_i[i] + M*b_z[i] <= M
    
    
    # solve model
    model.solve(pulp.PULP_CBC_CMD(maxSeconds=60, msg=1, fracGap=0))
    
    # print an error if the model is not optimal
    if pulp.LpStatus[model.status] != 'Optimal':
        print("Whoops! There is an error! The model as error status:" + pulp.LpStatus[model.status] )
        
    
    #Get positions    
    if pulp.LpStatus[model.status] == 'Optimal':
     
        # print variables
        var_model = dict()
        for variable in model.variables():
            var_model[variable.name] = variable.varValue
         
        # solution with variable names   
        var_model = pd.Series(var_model,index=var_model.keys())
         
         
        long_pos = [i for i in var_model.keys() if i.startswith("x") ]
                     
        # total portfolio with negative values as short positions
        port_total = pd.Series(var_model[long_pos].values ,index=[t[2:] for t in var_model[long_pos].index])
        
        # value of tau
        tau_val = var_model[var_model.keys() == 'Tau']
    
        opt_port = port_total/tau_val[0]
        opt_port[opt_port < 0.00001] = 0 # get rid of floating numbers
            
    return opt_port

def PortfolioRiskTarget(mu,scen,CVaR_target=1,lamb=1,max_weight=1,min_weight=None,cvar_alpha=0.05):
    
    """ This function finds the optimal enhanced index portfolio according to some benchmark. The portfolio corresponds to the tangency portfolio where risk is evaluated according to the CVaR of the tracking error. The model is formulated using fractional programming.
    
    Parameters
    ----------
    mu : pandas.Series with float values
        asset point forecast
    mu_b : pandas.Series with float values
        Benchmark point forecast
    scen : pandas.DataFrame with float values
        Asset scenarios
    scen_b : pandas.Series with float values
        Benchmark scenarios
    max_weight : float
        Maximum allowed weight    
    cvar_alpha : float
        Alpha value used to evaluate Value-at-Risk one    
    
    Returns
    -------
    float
        Asset weights in an optimal portfolio
        
    """
     
    # define index
    i_idx = mu.index
    j_idx = scen.index
    
    # number of scenarios
    N = scen.shape[0]    
    
    # define variables
    x = pulp.LpVariable.dicts("x", ( (i) for i in i_idx ),
                                         lowBound=0,
                                         cat='Continuous') 
    
    # loss deviation
    VarDev = pulp.LpVariable.dicts("VarDev", ( (t) for t in j_idx ),
                                         lowBound=0,
                                         cat='Continuous')
        
    # value at risk
    VaR = pulp.LpVariable("VaR",   lowBound=0,
                                         cat='Continuous')
    CVaR = pulp.LpVariable("CVaR",   lowBound=0,
                                         cat='Continuous')
    
    # binary variable connected to cardinality constraints
    b_z = pulp.LpVariable.dicts("b_z",   ( (i) for i in i_idx ),
                                         cat='Binary')
        
    #####################################
    ## define model
    model = pulp.LpProblem("Mean-CVaR Optimization", pulp.LpMaximize)
     
    #####################################
    ## Objective Function
             
    model += lamb*(pulp.lpSum([mu[i] * x[i] for i in i_idx] )) - (1-lamb)*CVaR
                      
    #####################################
    # constraint
                      
    # calculate CVaR
    for t in j_idx:
        model += -pulp.lpSum([ scen.loc[t,i]  * x[i] for i in i_idx] ) - VaR <= VarDev[t]
    
    model += VaR + 1/(N*cvar_alpha)*pulp.lpSum([ VarDev[t] for t in j_idx]) == CVaR    
    
    model += CVaR <= CVaR_target        
    
    ### price*number of products cannot exceed budget
    model += pulp.lpSum([ x[i] for i in i_idx]) == 1    
                                                
    ### Concentration limits
    # set max limits so it cannot not be larger than a fixed value  
    ###
    for i in i_idx:
        model += x[i] <= max_weight
    
    ### Add minimum weight constraint, either zero or atleast minimum weight
    if min_weight is not None:
    
        for i in i_idx:           
            model += x[i] >= min_weight*b_z[i]
            model += x[i] <= b_z[i]
        
    # solve model
    model.solve()
    
    # print an error if the model is not optimal
    if pulp.LpStatus[model.status] != 'Optimal':
        print("Whoops! There is an error! The model has error status:" + pulp.LpStatus[model.status] )
        
    
    #Get positions    
    if pulp.LpStatus[model.status] == 'Optimal':
     
        # print variables
        var_model = dict()
        for variable in model.variables():
            var_model[variable.name] = variable.varValue
         
        # solution with variable names   
        var_model = pd.Series(var_model,index=var_model.keys())
         
         
        long_pos = [i for i in var_model.keys() if i.startswith("x") ]
             
        # total portfolio with negative values as short positions
        port_total = pd.Series(var_model[long_pos].values ,index=[t[2:] for t in var_model[long_pos].index])
    
        opt_port = port_total
    
    # set flooting data points to zero and normalize
    opt_port[opt_port < 0.000001] = 0
    opt_port = opt_port/sum(opt_port)
    
    # return portfolio, CVaR, and alpha
    return opt_port, var_model["CVaR"], (sum(mu * port_total) - mu_b)

def PortfolioLambda(mu,mu_b,scen,scen_b,max_weight=1,min_weight=None,cvar_alpha=0.05,ft_points=15):
    

    # asset names
    assets = mu.index

    # column names
    col_names = mu.index.values.tolist() 
    col_names.extend(["Mu","CVaR","STAR"])
    # number of frontier points 

    
    # store portfolios
    portfolio_ft = pd.DataFrame(columns=col_names,index=list(range(ft_points)))
    
    # maximum risk portfolio    
    lamb=0.99999
    max_risk_port, max_risk_CVaR, max_risk_mu = PortfolioRiskTarget(mu=mu,scen=scen,CVaR_target=100,lamb=lamb,max_weight=max_weight,min_weight=min_weight,cvar_alpha=cvar_alpha)
    portfolio_ft.loc[ft_points-1,assets] = max_risk_port
    portfolio_ft.loc[ft_points-1,"Mu"] = max_risk_mu
    portfolio_ft.loc[ft_points-1,"CVaR"] = max_risk_CVaR
    portfolio_ft.loc[ft_points-1,"STAR"] = max_risk_mu/max_risk_CVaR
    
    # minimum risk portfolio
    lamb=0.00001
    min_risk_port, min_risk_CVaR, min_risk_mu= PortfolioRiskTarget(mu=mu,scen=scen,CVaR_target=100,lamb=lamb,max_weight=max_weight,min_weight=min_weight,cvar_alpha=cvar_alpha)
    portfolio_ft.loc[0,assets] = min_risk_port
    portfolio_ft.loc[0,"Mu"] = min_risk_mu
    portfolio_ft.loc[0,"CVaR"] = min_risk_CVaR
    portfolio_ft.loc[0,"STAR"] = min_risk_mu/min_risk_CVaR
    
    # CVaR step size
    step_size = (max_risk_CVaR-min_risk_CVaR)/ft_points # CVaR step size
    
    # calculate all frontier portfolios
    for i in range(1,ft_points-1):
        CVaR_target = min_risk_CVaR + step_size*i
        i_risk_port, i_risk_CVaR, i_risk_mu= PortfolioRiskTarget(mu=mu,scen=scen,CVaR_target=CVaR_target,lamb=1,max_weight=max_weight,min_weight=min_weight,cvar_alpha=cvar_alpha)
        portfolio_ft.loc[i,assets] = i_risk_port
        portfolio_ft.loc[i,"Mu"] = i_risk_mu
        portfolio_ft.loc[i,"CVaR"] = i_risk_CVaR
        portfolio_ft.loc[i,"STAR"] = i_risk_mu/i_risk_CVaR
        
    return portfolio_ft


def getcVarEfficientEF(mu,mu_b,df, scen,scen_b,cvar_alpha,step):
    OptPort_zero = MeanCVaR(mu,mu_b,scen,scen_b,max_weight=1,min_weight=None,cvar_alpha=cvar_alpha)
    frontier = PortfolioLambda(mu=mu,mu_b=mu_b,scen= scen,scen_b=scen_b,max_weight=1,min_weight=None,cvar_alpha=cvar_alpha,ft_points=step)

    return {'optimum zero benchmark portfolio': OptPort_zero,'cVar_EF' : frontier}

def mcVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")

def mcCVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series.")


def PerformMCSim(nbr_of_sim,days_to_sim,mu,cov,weights, alpha ,initialPortfolio, isPlot=0):
    """
    This function allows us to perform Monte-Carlo simulation and
    outputs the correspon
    """

    mc_sims = nbr_of_sim # number of simulations
    T = days_to_sim #timeframe in days

    meanM = np.full(shape=(T, len(weights)), fill_value=mu)
    meanM = meanM.T

    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

    initialPortfolio = 10000

    for m in range(0, mc_sims):
        # MC loops
        Z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(cov)
        dailyReturns = meanM + np.inner(L, Z)
        portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

    if isPlot==1:
      plt.plot(portfolio_sims)
      plt.ylabel('Portfolio Value ($)')
      plt.xlabel('Days')
      plt.title('MC simulation of a stock portfolio')
      plt.show()  

      print('VaR ${}'.format(round(VaR,2)))
      print('CVaR ${}'.format(round(CVaR,2)))
    
    portResults = pd.Series(portfolio_sims[-1,:])

    VaR = initialPortfolio - mcVaR(portResults, alpha=5)
    CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

    

    return VaR,CVaR