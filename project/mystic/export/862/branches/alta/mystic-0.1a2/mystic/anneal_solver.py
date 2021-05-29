"""
Solvers
=======

This module contains an optimization routine based on the simulated
annealing algorithm.  

A minimal interface that mimics a scipy.optimize interface has been
implemented, and functionality from the mystic solver API has been added
with reasonable defaults.  

Minimal function interface to optimization routines::
    anneal -- Simulated Annealing solver

The corresponding solver built on mystic's AbstractSolver is::
    AnnealSolver -- a simulated annealing solver

Mystic solver behavior activated in anneal::
    - EvaluationMonitor = Sow()
    - StepMonitor = Sow()
    - enable_signal_handler()
    - termination = AveragePopulationImprovement(tolerance)

Usage
=====

See `mystic.examples.test_anneal` for an example of using
AnnealSolver.

All solvers included in this module provide the standard signal handling.
For more information, see `mystic.mystic.abstract_solver`.


History
==========
Based on anneal.py from scipy.optimize:
Original Author: Travis Oliphant 2002
Bug-fixes in 2006 by Tim Leslie

Adapted for Mystic, 2009
"""

__all__ = ['AnnealSolver','anneal']

# Mystic and numpy imports
from mystic.tools import Null, wrap_function
from mystic.tools import wrap_bounds
import numpy
from numpy import asarray, tan, exp, ones, squeeze, sign, \
     all, log, sqrt, pi, shape, array, minimum, where
from numpy import random

from abstract_solver import AbstractSolver

#############################################################
# Helper classes and methods

_double_min = numpy.finfo(float).min
_double_max = numpy.finfo(float).max

# Base class for annealing schedules
class base_schedule(object):
    def __init__(self):
        self.dwell = 20
        self.learn_rate = 0.5
        self.lower = -10
        self.upper = 10
        self.Ninit = 50
        self.accepted = 0
        self.tests = 0
        self.feval = 0
        self.k = 0
        self.T = None

    def init(self, **options):
        self.__dict__.update(options)
        self.lower = asarray(self.lower)
        self.lower = where(self.lower == numpy.NINF, -_double_max, self.lower)
        self.upper = asarray(self.upper)
        self.upper = where(self.upper == numpy.PINF, _double_max, self.upper)
        self.k = 0
        self.accepted = 0
        self.feval = 0
        self.tests = 0

    def getstart_temp(self):
        """ Find a matching starting temperature and starting parameters vector
        i.e. find x0 such that func(x0) = T0.

        Parameters
        ----------
        best_state : _state
            A _state object to store the function value and x0 found.

        Returns
        -------
        x0 : array
            The starting parameters vector.
        """

        assert(not self.dims is None)
        lrange = self.lower
        urange = self.upper
        fmax = _double_min
        fmin = _double_max
        for _ in range(self.Ninit):
            x0 = random.uniform(size=self.dims)*(urange-lrange) + lrange
            fval = self.func(x0, *self.args)
            self.feval += 1
            if fval > fmax:
                fmax = fval
            if fval < fmin:
                fmin = fval
                bestEnergy = fval
                bestSolution = array(x0)

        self.T0 = (fmax-fmin)*1.5
        return bestSolution, bestEnergy

    def accept_test(self, dE):
        T = self.T
        self.tests += 1
        if dE < 0:
            self.accepted += 1
            return 1
        p = exp(-dE*1.0/self.boltzmann/T)
        if (p > random.uniform(0.0, 1.0)):
            self.accepted += 1
            return 1
        return 0

    def update_guess(self, x0):
        pass

    def update_temp(self, x0):
        pass

# Simulated annealing schedules: 'fast', 'cauchy', and 'boltzmann'

#  A schedule due to Lester Ingber
class fast_sa(base_schedule):
    def init(self, **options):
        self.__dict__.update(options)
        if self.m is None:
            self.m = 1.0
        if self.n is None:
            self.n = 1.0
        self.c = self.m * exp(-self.n * self.quench)

    def update_guess(self, x0):
        x0 = asarray(x0)
        u = squeeze(random.uniform(0.0, 1.0, size=self.dims))
        T = self.T
        y = sign(u-0.5)*T*((1+1.0/T)**abs(2*u-1)-1.0)
        xc = y*(self.upper - self.lower)
        xnew = x0 + xc
        return xnew

    def update_temp(self):
        self.T = self.T0*exp(-self.c * self.k**(self.quench))
        self.k += 1
        return

class cauchy_sa(base_schedule):
    def update_guess(self, x0):
        x0 = asarray(x0)
        numbers = squeeze(random.uniform(-pi/2, pi/2, size=self.dims))
        xc = self.learn_rate * self.T * tan(numbers)
        xnew = x0 + xc
        return xnew

    def update_temp(self):
        self.T = self.T0/(1+self.k)
        self.k += 1
        return

class boltzmann_sa(base_schedule):
    def update_guess(self, x0):
        std = minimum(sqrt(self.T)*ones(self.dims), (self.upper-self.lower)/3.0/self.learn_rate)
        x0 = asarray(x0)
        xc = squeeze(random.normal(0, 1.0, size=self.dims))

        xnew = x0 + xc*std*self.learn_rate
        return xnew

    def update_temp(self):
        self.k += 1
        self.T = self.T0 / log(self.k+1.0)
        return

class _state(object):
    def __init__(self):
        self.x = None
        self.cost = None

################################################################################

class AnnealSolver(AbstractSolver):
    """Simulated annealing optimization."""
    
    def __init__(self, dim):
        """
Takes one initial input: 
    dim  -- dimensionality of the problem

All important class members are inherited from AbstractSolver.
        """
        AbstractSolver.__init__(self, dim)

    def Solve(self, func, termination, sigint_callback=None,
              EvaluationMonitor=Null, StepMonitor=Null, ExtraArgs=(), **kwds):
        """Minimize a function using simulated annealing.

Description:

    Uses a simulated annealing algorithm to find the minimum of
    a function of one or more variables.

Inputs:

    func -- the Python function or method to be minimized.
    termination -- callable object providing termination conditions.

Additional Inputs:

    sigint_callback -- callback function for signal handler.
    EvaluationMonitor -- a callable object that will be passed x, fval
        whenever the cost function is evaluated.
    StepMonitor -- a callable object that will be passed x, fval
        after the end of an iteration.
    ExtraArgs -- extra arguments for func.

Further Inputs:

    schedule     -- Annealing schedule to use: 'cauchy', 'fast', or 'boltzmann'
                    [default='fast']
    T0           -- Initial Temperature (estimated as 1.2 times the largest
                    cost-function deviation over random points in the range)
                    [default=None]
    learn_rate   -- scale constant for adjusting guesses
                    [default=0.5]
    boltzmann    -- Boltzmann constant in acceptance test
                     (increase for less stringent test at each temperature).
                    [default=1.0]
    quench, m, n -- Parameters to alter fast_sa schedule
                    [all default to 1.0]
    dwell        -- The number of times to search the space at each temperature.
                    [default=50]

    Optional Termination Conditions:
    Tf           -- Final goal temperature
                    [default=1e-12]
    maxaccept    -- Maximum changes to accept
                    [default=None]
        """
        #allow for inputs that don't conform to AbstractSolver interface
        args = ExtraArgs
        x0 = self.population[0]

        schedule = "fast"
        T0 = None
        boltzmann = 1.0
        learn_rate = 0.5
        dwell = 50
        quench = 1.0
        m = 1.0
        n = 1.0

        Tf = 1e-12 # or None?
        self._maxaccept = None

        self.disp = 0
        self.callback = None

        if kwds.has_key('schedule'): schedule = kwds['schedule']
        if kwds.has_key('T0'): T0 = kwds['T0']
        if kwds.has_key('boltzmann'): boltzmann = kwds['boltzmann']
        if kwds.has_key('learn_rate'): learn_rate = kwds['learn_rate']
        if kwds.has_key('dwell'): dwell = kwds['dwell']
        if kwds.has_key('quench'): quench = kwds['quench']
        if kwds.has_key('m'): m = kwds['m']
        if kwds.has_key('n'): n = kwds['n']    

        if kwds.has_key('Tf'): Tf = kwds['Tf']
        if kwds.has_key('maxaccept'): self._maxaccept = kwds['maxaccept']

        if kwds.has_key('disp'): self.disp = kwds['disp']
        if kwds.has_key('callback'): self.callback = kwds['callback']

        #-------------------------------------------------------------

        import signal
        self._EARLYEXIT = False

        fcalls, func = wrap_function(func, ExtraArgs, EvaluationMonitor)

        if self._useStrictRange:
            x0 = self._clipGuessWithinRangeBoundary(x0)
            # Note: wrap_bounds changes the results slightly from the original
            func = wrap_bounds(func, self._strictMin, self._strictMax)

        #generate signal_handler
        self._generateHandler(sigint_callback) 
        if self._handle_sigint: signal.signal(signal.SIGINT, self.signal_handler)
        #-------------------------------------------------------------

        schedule = eval(schedule+'_sa()')
        #   initialize the schedule
        schedule.init(dims=shape(x0),func=func,args=args,boltzmann=boltzmann,T0=T0,
                  learn_rate=learn_rate, lower=self._strictMin, upper=self._strictMax,
                  m=m, n=n, quench=quench, dwell=dwell)

        if self._maxiter is None:
            self._maxiter = 400

        current_state, last_state = _state(), _state()
        if T0 is None:
            x0, self.bestEnergy = schedule.getstart_temp()
            self.bestSolution = x0
        else:
            #self.bestSolution = None
            self.bestSolution = x0
            self.bestEnergy = 300e8

        retval = 0
        last_state.x = asarray(x0).copy()
        fval = func(x0,*args)
        schedule.feval += 1
        last_state.cost = fval
        if last_state.cost < self.bestEnergy:
            self.bestEnergy = fval
            self.bestSolution = asarray(x0).copy()
        schedule.T = schedule.T0
        fqueue = [100, 300, 500, 700]
        self.population = asarray(fqueue)*1.0
        iters = 0
        while 1:
            StepMonitor(self.bestSolution, self.bestEnergy)
            for n in range(dwell):
                current_state.x = schedule.update_guess(last_state.x)
                current_state.cost = func(current_state.x,*args)
                schedule.feval += 1

                dE = current_state.cost - last_state.cost
                if schedule.accept_test(dE):
                    last_state.x = current_state.x.copy()
                    last_state.cost = current_state.cost
                    if last_state.cost < self.bestEnergy:
                        self.bestSolution = last_state.x.copy()
                        self.bestEnergy = last_state.cost
            schedule.update_temp() 

            iters += 1
            fqueue.append(squeeze(last_state.cost))
            fqueue.pop(0)
            af = asarray(fqueue)*1.0

            # Update monitors/variables
            self.population = af
            self.energy_history.append(self.bestEnergy)

            if self.callback is not None:
                self.callback(self.bestSolution)

            # Stopping conditions
            # - last saved values of f from each cooling step
            #     are all very similar (effectively cooled)
            # - Tf is set and we are below it
            # - maxfun is set and we are past it
            # - maxiter is set and we are past it
            # - maxaccept is set and we are past it

            if self._EARLYEXIT or termination(self):
                # How to deal with the below warning? It uses feps, which is passed
                # to termination, so it would be repetitive to also pass it to Solve().
                #if abs(af[-1]-best_state.cost) > feps*10:
                #print "Warning: Cooled to %f at %s but this is not" \
                #      % (squeeze(last_state.cost), str(squeeze(last_state.x))) \
                #      + " the smallest point found."
                break
            if (Tf is not None) and (schedule.T < Tf):
                break
           # if (self._maxfun is not None) and (schedule.feval > self._maxfun):
           #     retval = 1
           #     break
            if (self._maxfun is not None) and (fcalls[0] > self._maxfun):
                retval = 1
                break
            if (iters > self._maxiter):
                retval = 2
                break
            if (self._maxaccept is not None) and (schedule.accepted > self._maxaccept):
                break

        signal.signal(signal.SIGINT,signal.default_int_handler)

        # Store some information. Is there a better way to do this?
        self.generations = iters        # Number of iterations
        self.T = schedule.T             # Final temperature
        self.accept = schedule.accepted # Number of tests accepted

        # code below here pushes output to scipy.optimize interface

        if self.disp:
            if retval == 1: 
                print "Warning: Maximum number of function evaluations has "\
                      "been exceeded."
            elif retval == 2:
                print "Warning: Maximum number of iterations has been exceeded"
            else:
                print "Optimization terminated successfully."
                print "         Current function value: %f" % self.bestEnergy
                print "         Iterations: %d" % iters
                print "         Function evaluations: %d" % fcalls[0]

        return 

##################################################################################
# function interface for using AnnealSolver

def anneal(func, x0, args=(), schedule='fast', full_output=0,
           T0=None, Tf=1e-12, maxaccept=None, lower= -100, upper= 100, maxiter=400,
           boltzmann=1.0, learn_rate=0.5, feps=1e-6, quench=1.0, m=1.0, n=1.0,
           maxfun=None, dwell=50, callback=None, disp=0, retall=0):
    """Minimize a function using simulated annealing.

    Inputs:

    func         -- Function to be optimized
    x0           -- Parameters to be optimized over. Should be a list.

    Optional Inputs:

    args         -- Extra parameters to function
    schedule     -- Annealing schedule to use.
                    Choices are: 'fast', 'cauchy', 'boltzmann'
    full_output  -- Non-zero to return optional outputs
    retall       -- Non-zero to return list of solutions at each iteration.
    disp         -- Non-zero to print convergence messages.
    T0           -- Initial Temperature (estimated as 1.2 times the largest
                    cost-function deviation over random points in the range)
    Tf           -- Final goal temperature
    maxfun       -- Maximum function evaluations
    maxaccept    -- Maximum changes to accept
    maxiter      -- Maximum cooling iterations
    learn_rate   -- scale constant for adjusting guesses
    boltzmann    -- Boltzmann constant in acceptance test
                     (increase for less stringent test at each temperature).
    feps         -- Stopping relative error tolerance for the function value in
                     last four coolings.
    quench, m, n -- Parameters to alter fast_sa schedule
    lower, upper -- lower and upper bounds on x0.
    dwell        -- The number of times to search the space at each temperature.

    Outputs: (xmin, {fopt, iters, feval, T, accept}, retval, {allvecs})

    xmin    -- Point giving smallest value found
    fopt    -- Minimum value of function found
    iters   -- Number of cooling iterations
    feval   -- Number of function evaluations
    T       -- final temperature
    accept  -- Number of tests accepted.
    retval  -- Flag indicating stopping condition:
                1 : Maximum function evaluations
                2 : Maximum iterations reached
                3 : Maximum accepted query locations reached
    allvecs -- a list of solutions at each iteration.
    """

    from mystic.tools import Sow
    from mystic.termination import AveragePopulationImprovement
    stepmon = Sow()
    evalmon = Sow()
    
    x0 = asarray(x0)
    solver = AnnealSolver(len(x0))
    solver.SetInitialPoints(x0)
    solver.enable_signal_handler()
    solver.SetEvaluationLimits(maxiter,maxfun)
    
    # Default bounds cause errors when function is not 1-dimensional
    solver.SetStrictRanges(lower,upper)

    solver.Solve(func,termination=AveragePopulationImprovement(tolerance=feps),\
                 EvaluationMonitor=evalmon,StepMonitor=stepmon,\
                 ExtraArgs=args, callback=callback,\
                 schedule=schedule, T0=T0, Tf=Tf,\
                 boltzmann=boltzmann, learn_rate=learn_rate,\
                 dwell=dwell, quench=quench, m=m, n=n,\
                 maxaccept=maxaccept, disp=disp)   
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize interface
    #x = list(solver.bestSolution)
    x = solver.bestSolution
    fval = solver.bestEnergy
    fcalls = len(evalmon.x)
    iterations = len(stepmon.x)
    allvecs = []
    for i in range(len(stepmon.x)):
       #allvecs.append(list(stepmon.x[i]))
        allvecs.append(stepmon.x[i])

    # to be fixed
    T = solver.T
    accept = solver.accept

    retval = 0
    if (maxfun is not None) and (fcalls > maxfun):
        retval = 1
    if (iterations > maxiter):
        retval = 2
    if (maxaccept is not None) and (accept > maxaccept):
        retval = 3

    if full_output:
        retlist = x, fval, iterations, fcalls, T, accept, retval
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x, retval
        if retall:
            retlist = (x, allvecs)

    return retlist

if __name__=='__main__':
    help(__name__)

# end of file
