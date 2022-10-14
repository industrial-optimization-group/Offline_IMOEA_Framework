import math

def probability_wrong(A, stdA, B, stdB):
    """
    Computes the probability of choosing A wrongly
    when minimizing using hyperbolic tangent approximation
    with approximated function values A and B 
    and their standard deviations stdA and stdB
    """
    if math.isclose(stdA, 0):
        stdA += 1e-7
    if math.isclose(stdB, 0):
        stdB += 1e-7

    m = (A-B) / stdB
    s = stdA / stdB
    inside = m / (0.8 * math.sqrt(2.0 + 2.0 * s**2.0))
    return 0.5 * (1.0 + math.tanh(inside))

def probability_wrong_right_eq(A, stdA, B, stdB):
    """
    Computes the probabilities of a being better than b
    a being worse than b and them beign equally good
    in terms of pareto-optimality
    Params:
        A: objective vector
        stdA: standard deviation of objective vector prediction A
        B: objective vector
        stdB: standard deviation of objective vector prediction B
    Returns:
        (P(A < B), P(A > B), P(A == B))
    """
    pright = 1.0
    pwrong = 1.0
    for a,sa, b, sb in zip(A, stdA, B, stdB):
        pwrongj = probability_wrong(a, sa, b, sb)
        pwrong *= pwrongj
        pright *= (1-pwrongj)
    pequal = 1 - pright - pwrong

    return pright, pwrong, pequal

def compute_ranking(F, stds):
    """
    Computes ranking of objective function values
    where best (smallest value gets smallest ranking)
    Params:
        F: ndarray (n,) of objective function values
        stds: ndarray (n,) of standard deviation values
        related to objective function predictions
    Returns:
        ndarray (n,) Ranking for each objective function value
    """
    R = []
    for i in range(len(F)):
        Ri = -0.5
        for j in range(len(F)):
            Ri += probability_wrong(F[i], stds[i], F[j], stds[j])
        R.append(Ri)
    return R

def compute_ranking_k(F, stds):
    """
    Ranks objective vectors
    Params:
        F: ndarray (n,k) of objective function values
        stds: ndarray (n,k) of standard deviation of predictions related to 
        objective function values in F
    Returns:
        Ranking ndarray (n,) for each objective function
    """
    R = []
    for i in range(len(F)):
        Ri = -0.5
        for j in range(len(F)):
            _, pwrong, pequal = probability_wrong_right_eq(F[i], stds[i], F[j], stds[j])
            Ri += pwrong + 0.5 * pequal
        R.append(Ri)
    return R


def probability_of_selection(F, stds):
    """
    Compute probability for each function value that it
    should be selected based on objective function
    values F and their corresponding
    standard deviations stds
    Params:
        F: ndarray (n,) of objective function values
        stds: ndarray (n,) of standard deviation values
        related to objective function predictions
    Returns:
        ndarray (n,) Ranking for each objective function value
    """
    R = compute_ranking(F, stds)
    P = []
    n = len(R)
    for i in range(len(R)):
        p = (2 * ((n-1) - R[i])) / (n*(n-1))
        P.append(p)
    return P

def probability_of_dominance(F, stds):
    """
    Compute probability for each function value that it
    should be selected based on objective function
    values F and their corresponding
    standard deviations stds
    Params:
        F: ndarray (n,k) of objective function values
        stds: ndarray (n,k) of standard deviation of predictions related to 
        objective function values in F
    Returns:
        probabilities ndarray (n,) for each objective function
    """
    R = compute_ranking_k(F, stds)
    P = []
    n = len(R)
    for i in range(len(R)):
        p = (2 * ((n-1) - R[i])) / (n*(n-1))
        P.append(p)
    return P