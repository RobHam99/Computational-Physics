from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
import time

class Worker(Thread):
    """Computes x, v, and a of the ith body"""

    def __init__(self, *args, **kwargs):
        super(Worker, self).__init__(*args, **kwargs)
        self.inputs = [] # load work onto the thread
        self.results = [] # get return values from thread
        self.running = True # end life of thread cleanly
        self.daemon = True # tell python that this thread may be stopped when program exits
        self.start() # makes thread available for work

    def run(self):
        while self.running:
            if len(self.inputs) == 0:
                continue
            i, x0, v0, G, m, dt = self.inputs.pop(0)
            a_i0 = a(i, x0, G, m)
            v_i1 = a_i0 * dt + v0[i]
            x_i1 = a_i0 * dt**2 + v0[i] * dt + x0[i]
            result = (i, x_i1, v_i1)
            self.results.append(result)


class Pool(object):
    """A collection of P worker threads that distributes tasks
    evenly across them.
    """
    def __init__(self, size):
        self.size = size
        # create new workers based on size
        self.workers = [Worker() for p in range(size)]

    def do(self, tasks):
        # distribute tasks evenly among workers
        for p in range(self.size):
            self.workers[p].inputs += tasks[p::self.size]

        # wait for all workers to finish tasks
        while any([len(worker.inputs) != 0 for worker in self.workers]):
            pass

        # get back results from workers and clean out workers
        results = []
        for worker in self.workers:
            results += worker.results
            worker.results.clear()
        # return list of all results for all inputs
        return results

    # stop the workers when pool is shut down
    def __del__(self):
        for worker in self.workers:
            worker.running = False


def remove_i(x, i):
    """Drops the ith element of an array."""

    shape = (x.shape[0]-1,)+x.shape[1:] # get shape

    y = np.empty(shape, dtype=float) # create new y array with new shape
    y[:i] = x[:i] # take x up to index i
    y[i:] = x[i+1:] # skip index i and take the rest

    return y


def a(i, x, G, m):
    """The acceleration of the ith mass."""

    x_i = x[i]
    x_j = remove_i(x, i)
    m_j = remove_i(m, i)
    diff = x_j - x_i # part one of acceleration equation
    mag3 = np.sum(diff**2, axis=1)**1.5 # part two of acceleration equation

    result = G * np.sum(diff * (m_j / mag3)[:,np.newaxis], axis=0) # total acceleration due to each other mass j on mass i

    return result


def initial_cond(N, D):
    """Generates initial conditions for N unity masses at rest
    starting at random positions in D-dimensional space."""

    x0 = np.random.rand(N, D) # random initial positions
    v0 = np.zeros((N, D), dtype=float) # velocity array
    m = np.ones(N, dtype=float) # mass array, all masses = 1

    return x0, v0, m


def timestep(x0, v0, G, m, dt, pool):
    """Computes the next position and velocity for all masses given
    initial conditions and a time step size.
    """

    N = len(x0)
    tasks = [(i, x0, v0, G, m, dt) for i in range(N)] # create task for each body
    results = pool.do(tasks) # run tasks
    x1 = np.empty(x0.shape, dtype=float)
    v1 = np.empty(v0.shape, dtype=float)

    # put the results in order since they most likely aren't in order
    for i, x_i1, v_i1 in results:
        x1[i] = x_i1
        v1[i] = v_i1
    return x1, v1


def simulate(P, N, D, S, G, dt):
    """ Starts with initial conditions, finds x1, v1,
    then sets those as the initial conditions and calculates new x1, v1
    """

    x0, v0, m = initial_cond(N, D)
    pool = Pool(P)

    for s in range(S):
        x1, v1 = timestep(x0, v0, G, m, dt, pool)
        x0, v0 = x1, v1


Ps = [1, 2, 4, 8]
runtimes = []
for P in Ps:
    start = time.time()
    simulate(P, 64, 3, 300, 1.0, 1e-3)
    stop = time.time()
    runtimes.append(stop - start)

plt.figure()
for i in range(len(Ps)):
    plt.plot(Ps[i], runtimes[0]/runtimes[i], 'ro')
plt.show()
