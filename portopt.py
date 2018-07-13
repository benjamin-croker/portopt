import numpy as np
import scipy.optimize as scipy_opt


def portfolio_rets(rets, w):
    return np.matmul(rets, w)


def portfolio_var(cov, w):
    return np.matmul(w, np.matmul(cov, w))


def portfolio_sd(cov, w):
    return np.sqrt(portfolio_var(cov, w))


def even_w(rets):
    return np.ones(rets.shape)/rets.shape[0]


def make_fun(rets, cov, q):
    """ returns f(w) and f'(w)
    """
    def f(w):
        return np.matmul(w, np.matmul(cov, w)) - q * np.matmul(rets, w)

    def df(w):
        return 2*np.matmul(cov, w) - q * rets
    return f, df


def constraint_fun(w):
    "returns 0 when the sum of W == 1"
    return w.sum() - 1


def d_constraint_fun(w):
    return np.ones(w.shape)


def effecient_frontier(rets, cov, risk_free_rate,
                       min_q=0, max_q=10, n=30, stop_tol=1e-6):
    w0 = even_w(rets)
    results = []
    # bound weights to +ve numbers (including 0)
    bounds = scipy_opt.Bounds(np.zeros(w0.shape), np.full(w0.shape, np.inf))
    cons = [{'type': 'eq', 'fun': constraint_fun, 'jac': d_constraint_fun}]
    options = {'ftol': 1e-16, 'disp': False}

    for q in np.geomspace(min_q+1, max_q+1, num=n)-1:
        f, df = make_fun(rets, cov, q)
        res = scipy_opt.minimize(
            f, w0, jac=df,
            constraints=cons, bounds=bounds,
            method='SLSQP', options=options
        )

        if np.sum(np.abs(w0 - res['x'])) < stop_tol:
            break

        p_rets = portfolio_rets(rets, res['x'])
        p_sd = portfolio_sd(cov, res['x'])
        results += [
            {'w': res['x'],
             'portfolio_ret':  p_rets,
             'portfolio_sd': p_sd,
             'sharpe_ratio': (p_rets - risk_free_rate)/p_sd,
             'q': q}
        ]
        w0 = res['x']
    return results


def optim_sharpe(allocs, *args, **kwargs):
    sharpe = np.array([a['sharpe_ratio'] for a in allocs])
    i_min = np.argmax(sharpe)
    if i_min == 0:
        kwargs['min_q'] = 0
    else:
        kwargs['min_q'] = allocs[i_min - 1]['q']

    if i_min == len(allocs)-1:
        kwargs['max_q'] = 2 * allocs[i_min]['q']
    else:
        kwargs['max_q'] = allocs[i_min + 1]['q']

    new_allocs = effecient_frontier(*args, **kwargs)
    sharpe = np.array([a['sharpe_ratio'] for a in allocs])
    return new_allocs[np.argmax(sharpe)]
