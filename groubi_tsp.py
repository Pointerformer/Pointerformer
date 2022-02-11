import sys
import math
import random
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
from env import readDataFile
import time

n = None


# Parse argument

# if len(sys.argv) < 2:
#     print('Usage: tsp.py npoints')
#     sys.exit(1)
# n = int(sys.argv[1])

# Create n random points

# random.seed(1)
# points = [(random.randint(0, 100), random.randint(0, 100)) for i in range(n)]

# Dictionary of Euclidean distance between each pair of points


def solver(data_path):
    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            # find the shortest cycle in the selected edge list
            tour = subtour(vals)
            if len(tour) < n:
                # add subtour elimination constr. for every pair of cities in tour
                model.cbLazy(
                    gp.quicksum(model._vars[i, j] for i, j in combinations(tour, 2))
                    <= len(tour) - 1
                )

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(vals):
        # make a list of edges selected in the solution
        edges = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, "*") if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    data = readDataFile(data_path)
    n = data.shape[1]
    all_res = []
    start = time.time()
    for points in data:

        dist = {
            (i, j): math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
            for i in range(n)
            for j in range(i)
        }

        m = gp.Model()

        # Create variables

        vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name="e")
        for i, j in vars.keys():
            vars[j, i] = vars[i, j]  # edge in opposite direction

        # You could use Python looping constructs and m.addVar() to create
        # these decision variables instead.  The following would be equivalent
        # to the preceding m.addVars() call...
        #
        # vars = tupledict()
        # for i,j in dist.keys():
        #   vars[i,j] = m.addVar(obj=dist[i,j], vtype=GRB.BINARY,
        #                        name='e[%d,%d]'%(i,j))

        # Add degree-2 constraint

        m.addConstrs(vars.sum(i, "*") == 2 for i in range(n))

        # Using Python looping constructs, the preceding would be...
        #
        # for i in range(n):
        #   m.addConstr(sum(vars[i,j] for j in range(n)) == 2)

        # Optimize model

        m._vars = vars
        m.Params.LazyConstraints = 1
        m.optimize(subtourelim)

        vals = m.getAttr("X", vars)
        tour = subtour(vals)

        assert len(tour) == n

        # print('')
        # print('Optimal tour: %s' % str(tour))
        # print('Optimal cost: %g' % m.ObjVal)
        # print('')

        all_res.append(m.ObjVal)
    end = time.time()
    mean_len = sum(all_res) / len(all_res)
    total_time = end - start
    with open("./data/res_partner_{}".format(n), "w") as fp:
        fp.write(str(mean_len) + "\n" + str(total_time))

    print(mean_len, total_time)


if __name__ == "__main__":
    ##
    for n in [500, 1000]:
        solver("./data/partner_{}.txt".format(n))
