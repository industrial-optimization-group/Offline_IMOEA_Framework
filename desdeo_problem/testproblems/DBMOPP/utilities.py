import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


def get_2D_version(x, pi1, pi2):
    """
    Project n > 2 dimensional vector to 2-dimensional space

    Args:
        x (np.ndarray): A given vector to project to 2-dimensional space

    Returns:
        np.ndarray: A 2-dimensional vector
    """
    if (x.shape[1] <= 2):
        #print("Skipping projection, vector already 2 dimensional or less")
        return x
    l = np.divide(np.dot(x, pi1), np.sum(pi1))  # Left side of vector
    r = np.divide(np.dot(x, pi2), np.sum(pi2))  # Right side of vector
    #print(pi1, pi2)
    # should be [[]] still
    return np.hstack((l, r))


# X1 is a matrix or array, size n x number of design variables
# x2 is a array, size 1 x number of design variables

# currently x1 and x2 return float, and are both vectors?
def euclidean_distance(x1, x2):
    """
        Returns the euclidean distance between x1 and x2.
    """
    if x1 is None or x2 is None:
        print("euclidean distance supplied with nonetype")
        return None

    #dist = np.sqrt(np.sum((x1 - x2)**2, axis=1 ))
    dist = np.linalg.norm(x1-x2, axis=-1)
    # print(dist)
    return dist


def convhull(points):
    """
    Construct a convex hull of given set of points

    Args:
        points (np.ndarray): the points used to construct the convex hull

    Returns:
        np.ndarray: The indices of the simplices that form the convex hull
    """
    points = np.sort(points)
    try:
        return ConvexHull(points)
    except Exception:
        print("failed to construct the convex hull, using the points as is")
        return points


def in_hull(x: np.ndarray, points: np.ndarray):
    """
    Is a point inside a convex hull 

    Args:
        x (np.ndarray): The point that is checked
        points (np.ndarray): The point cloud of the convex hull

    Returns:
        bool: is x inside the convex hull given by points 
    """
    p = (np.concatenate(points))  # wont work for most cases?
    n_points = len(p)
    c = np.zeros(n_points)
    A = np.r_[p.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def get_random_angles(n):
    return np.random.rand(n, 1) * 2 * np.pi


def between_lines_rooted_at_pivot(x, pivot_loc, loc1, loc2) -> bool:
    """

    Args:
        x (np.ndarray): 2D point to check
        pivot_loc: attractor on boundary of circle
        loc1: another point on boundary of circle
        loc2: another point on boundary of circle

    Returns:
        bool: true if x on different side of line defined by pivot_loc and loc1, compared to the side of the line defined by pivot_loc and loc2.
        If x is also in the circle, then x is betweeen the two lines if return is true.
    """
    d1 = (x[0] - pivot_loc[0])*(loc1[1] - pivot_loc[1]) - \
        (x[1] - pivot_loc[1])*(loc1[0] - pivot_loc[0])
    d2 = (x[0] - pivot_loc[0])*(loc2[1] - pivot_loc[1]) - \
        (x[1] - pivot_loc[1])*(loc2[0] - pivot_loc[0])

    return d1 == 0 or d2 == 0 or np.sign(d1) != np.sign(d2)


def assign_design_dimension_projection(n_variables, vary_sol_density):
    """
    if more than two design dimensions in problem, need to assign
    the mapping down from this higher space to the 2D version
    which will be subsequantly evaluated
    """
    if n_variables <= 2:
        #print(
        #    "fNo need to assign dimension projections as number of variables is already {n_variables}")
        return None, None
    mask = np.random.permutation(n_variables-1)  # Test againt matlab
    if vary_sol_density:
        diff = np.random.randint(n_variables)
        mask = mask[:diff]  # Take the diff first elements
    else:
        half = int(np.ceil(n_variables))
        mask = mask[:half]  # Take half first elements
    pi1 = np.zeros(n_variables)
    pi1[mask] = True
    pi2 = pi1
    pi2 = np.ones(n_variables)
    pi2[mask] = False
    return pi1, pi2
