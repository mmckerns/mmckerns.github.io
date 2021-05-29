"""
Solvers
=======

This module contains an optimization routine adapted
from scipy.optimize.  The minimal scipy interface has been preserved,
and functionality from the mystic solver API has been added with
reasonable defaults.

Minimal function interface to optimization routines::
   fmin_ncg -- Newton-CG optimization

The corresponding solvers built on mystic's AbstractSolver are::
   NCGSolver -- Solver that uses the Newton-CG algorithm to minimize a function.

Mystic solver behavior activated in fmin_ncg::
   - EvaluationMonitor = Sow()
   - StepMonitor = CustomSow() with 4 columns
   - enable_signal_handler()
   - termination = SolutionImprovement(tolerance)

Usage
=====

See `mystic.examples.test_ncg` for an example of using
NCGSolver. 

All solvers included in this module provide the standard signal handling.
For more information, see `mystic.mystic.abstract_solver`.

References
==========
See Wright, and Nocedal 'Numerical Optimization', 1999,
      pg. 140.

Adapted to Mystic, 2009
"""
__all__ = ['NCGSolver','fmin_ncg']

import numpy
from numpy import atleast_1d, eye, mgrid, argmin, zeros, shape, empty, \
     squeeze, vectorize, asarray, absolute, sqrt, Inf, asfarray, isinf

from mystic.tools import Null, wrap_function
from mystic.tools import wrap_bounds
from abstract_solver import AbstractSolver

#############################################################################
# Helper methods and classes

def line_search_BFGS(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=1):
    """Minimize over alpha, the function ``f(xk+alpha pk)``.

    Uses the interpolation algorithm (Armiijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pg. 56-57

    :Returns: (alpha, fc, gc)

    """

    xk = atleast_1d(xk)
    fc = 0
    phi0 = old_fval # compute f(xk) -- done in past loop
    phi_a0 = f(*((xk+alpha0*pk,)+args))
    fc = fc + 1
    derphi0 = numpy.dot(gfk,pk)

    if (phi_a0 <= phi0 + c1*alpha0*derphi0):
        return alpha0, fc, 0, phi_a0

    # Otherwise compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = f(*((xk+alpha1*pk,)+args))
    fc = fc + 1

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, fc, 0, phi_a1

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satifies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    while 1:       # we are assuming pk is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + numpy.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = f(*((xk+alpha2*pk,)+args))
        fc = fc + 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, fc, 0, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

def approx_fhess_p(x0,p,fprime,epsilon,*args):
    f2 = fprime(*((x0+epsilon*p,)+args))
    f1 = fprime(*((x0,)+args))
    return (f2 - f1)/epsilon

_epsilon = sqrt(numpy.finfo(float).eps)

#########################################################################

class NCGSolver(AbstractSolver):
    """
Newton-CG Optimization.
    """
    
    def __init__(self, dim):
        """
Takes one initial input: 
    dim  -- dimensionality of the problem

All important class members are inherited from AbstractSolver.
        """
        AbstractSolver.__init__(self,dim)


    def Solve(self, func, termination, sigint_callback=None,
              EvaluationMonitor=Null, StepMonitor=Null, #GradientMonitor=Null,
              ExtraArgs=(), **kwds):
        """Minimize a function using NCG.

Description:

    Uses a Newton-CG algorithm to find the minimum of
    a function of one or more variables.

Inputs:

    func -- the Python function or method to be minimized.
    termination -- callable object providing termination conditions.

Additional Inputs:

    sigint_callback -- callback function for signal handler.
    EvaluationMonitor -- a callable object that will be passed x, fval
        whenever the cost function is evaluated.
    StepMonitor -- a callable object that will be passed x, fval, the gradient, and
                   the Hessian after the end of an iteration.
    ExtraArgs -- extra arguments for func and fprime (same for both).

Further Inputs:

    fprime -- callable f'(x,*args)
            Gradient of f.
    fhess_p : callable fhess_p(x,p,*args)
        Function which computes the Hessian of f times an
        arbitrary vector, p.
    fhess : callable fhess(x,*args)
        Function to compute the Hessian matrix of f.
    epsilon : float or ndarray
        If fhess is approximated, use this value for the step size.
    callback : callable
        An optional user-supplied function which is called after
        each iteration.  Called as callback(xk), where xk is the
        current parameter vector.
    disp : bool
        If True, print convergence message.

    Notes:
      Only one of `fhess_p` or `fhess` need to be given.  If `fhess`
      is provided, then `fhess_p` will be ignored.  If neither `fhess`
      nor `fhess_p` is provided, then the hessian product will be
      approximated using finite differences on `fprime`. `fhess_p`
      must compute the hessian times an arbitrary vector. If it is not
      given, finite-differences on `fprime` are used to compute
      it. 

        """
        # allow for inputs that don't conform to AbstractSolver interface
        args = ExtraArgs
        x0 = self.population[0]
        x0 = asarray(x0).flatten()

        epsilon = _epsilon
        self.disp = 1
        self.callback = None
        fhess_p = None
        fhess = None

        if kwds.has_key('epsilon'): epsilon = kwds['epsilon']
        if kwds.has_key('callback'): self.callback = kwds['callback']
        if kwds.has_key('disp'): self.disp = kwds['disp']
        if kwds.has_key('fhess'): fhess = kwds['fhess']
        if kwds.has_key('fhess_p'): fhess_p = kwds['fhess_p']

        # fprime is actually required. Temporary fix?:
        if kwds.has_key('fprime'): fprime = kwds['fprime']
        #-------------------------------------------------------------

        import signal
        self._EARLYEXIT = False

        fcalls, func = wrap_function(func, args, EvaluationMonitor)
        if self._useStrictRange:
            x0 = self._clipGuessWithinRangeBoundary(x0)
            func = wrap_bounds(func, self._strictMin, self._strictMax)

        #generate signal_handler
        self._generateHandler(sigint_callback) 
        if self._handle_sigint: signal.signal(signal.SIGINT, self.signal_handler)

        #--------------------------------------------------------------

        if self._maxiter is None:
            self._maxiter = len(x0)*200

        # Wrap gradient function? 
        # gcalls, fprime = wrap_function(fprime, args, GradientMonitor)   
        gcalls, fprime = wrap_function(fprime, args, Null)

        # Wrap hessian monitor?
        # But wrap_function assumes the function takes one parameter...
        #if fhess is not None:
        #    hcalls2, fhess = wrap_function(fhess, args, HessianMonitor)
        #else:
        #    if fhess_p is not None:
        #        hcalls2, fhess_p = wrap_function(fhess_p, args, HessianMonitor)

        #xtol = len(x0)*avextol
        #update = [2*xtol]
        xk = x0
        k = 0
        hcalls = 0
        old_fval = func(x0)
        abs = absolute
        while k < self._maxiter:
            # Compute a search direction pk by applying the CG method to
            #  del2 f(xk) p = - grad f(xk) starting from 0.
            b = -fprime(xk)
            maggrad = numpy.add.reduce(abs(b))
            eta = min([0.5,numpy.sqrt(maggrad)])
            termcond = eta * maggrad
            xsupi = zeros(len(x0), dtype=x0.dtype)
            ri = -b
            psupi = -ri
            i = 0
            dri0 = numpy.dot(ri,ri)
    
            if fhess is not None:             # you want to compute hessian once.
                A = fhess(*(xk,)+args)
                hcalls = hcalls + 1
    
            while numpy.add.reduce(abs(ri)) > termcond:
                if fhess is None:
                    if fhess_p is None:
                        Ap = approx_fhess_p(xk,psupi,fprime,epsilon)
                    else:
                        Ap = fhess_p(xk,psupi, *args)
                        hcalls = hcalls + 1
                else:
                    Ap = numpy.dot(A,psupi)
                # check curvature
                Ap = asarray(Ap).squeeze() # get rid of matrices...
                curv = numpy.dot(psupi,Ap)
                if curv == 0.0:
                    break
                elif curv < 0:
                    if (i > 0):
                        break
                    else:
                        xsupi = xsupi + dri0/curv * psupi
                        break
                alphai = dri0 / curv
                xsupi = xsupi + alphai * psupi
                ri = ri + alphai * Ap
                dri1 = numpy.dot(ri,ri)
                betai = dri1 / dri0
                psupi = -ri + betai * psupi
                i = i + 1
                dri0 = dri1          # update numpy.dot(ri,ri) for next time.
    
            pk = xsupi  # search direction is solution to system.
            gfk = -b    # gradient at xk
            alphak, fc, gc, old_fval = line_search_BFGS(func,xk,pk,gfk,old_fval)
    
            update = alphak * pk

            # Put last solution in trialSolution for termination()
            self.trialSolution = xk

            xk = xk + update        # upcast if necessary
            if self.callback is not None:
                self.callback(xk)
            k += 1

            # Update variables/monitors
            self.bestSolution = xk
            self.bestEnergy = old_fval
            StepMonitor(self.bestSolution,self.bestEnergy, gfk, Ap)
            self.energy_history.append(self.bestEnergy)

            if self._EARLYEXIT or termination(self): 
                break

        self.generations = k

        # Fix me?
        self.hcalls = hcalls
        self.gcalls = gcalls[0]

        signal.signal(signal.SIGINT,signal.default_int_handler)
    
        if self.disp:
            fval = old_fval
        if k >= self._maxiter:
            if self.disp:
                print "Warning: Maximum number of iterations has been exceeded"
                print "         Current function value: %f" % fval
                print "         Iterations: %d" % k
                print "         Function evaluations: %d" % fcalls[0]
                print "         Gradient evaluations: %d" % gcalls[0]
                print "         Hessian evaluations: %d" % hcalls
        else:
            if self.disp:
                print "Optimization terminated successfully."
                print "         Current function value: %f" % fval
                print "         Iterations: %d" % k
                print "         Function evaluations: %d" % fcalls[0]
                print "         Gradient evaluations: %d" % gcalls[0]
                print "         Hessian evaluations: %d" % hcalls
    
#############################################################################
# Interface for using NCGSolver

def fmin_ncg(func, x0, fprime, fhess_p=None, fhess=None, args=(), xtol=1e-5,
             epsilon=_epsilon, maxiter=None, full_output=0, disp=1, retall=0,
             callback=None):
    """Minimize a function using the Newton-CG method.

    Input Parameters:

        f : callable f(x,*args)
            Objective function to be minimized.
        x0 : ndarray
            Initial guess.
        fprime : callable f'(x,*args)
            Gradient of f.

    Optional Input Parameters:

        fhess_p : callable fhess_p(x,p,*args)
            Function which computes the Hessian of f times an
            arbitrary vector, p.
        fhess : callable fhess(x,*args)
            Function to compute the Hessian matrix of f.
        args : tuple
            Extra arguments passed to f, fprime, fhess_p, and fhess
            (the same set of extra arguments is supplied to all of
            these functions).
        epsilon : float or ndarray
            If fhess is approximated, use this value for the step size.
        callback : callable
            An optional user-supplied function which is called after
            each iteration.  Called as callback(xk), where xk is the
            current parameter vector.
        xtol : float
            Convergence is assumed when the relative error in
            the minimizer falls below this amount.
        maxiter : int
            Maximum number of iterations to perform.
        full_output : bool
            If True, return the optional outputs.
        disp : bool
            If True, print convergence message.
        retall : bool
            If True, return a list of results at each iteration.

    Returns: (xopt, {fopt, fcalls, gcalls, hcalls, warnflag},{allvecs})

        xopt : ndarray
            Parameters which minimizer f, i.e. ``f(xopt) == fopt``.
        fopt : float
            Value of the function at xopt, i.e. ``fopt = f(xopt)``.
        fcalls : int
            Number of function calls made.
        gcalls : int
            Number of gradient calls made.
        hcalls : int
            Number of hessian calls made.
        warnflag : int
            Warnings generated by the algorithm.
            1 : Maximum number of iterations exceeded.
        allvecs : list
            The result at each iteration, if retall is True (see below).

    Notes:
      Only one of `fhess_p` or `fhess` need to be given.  If `fhess`
      is provided, then `fhess_p` will be ignored.  If neither `fhess`
      nor `fhess_p` is provided, then the hessian product will be
      approximated using finite differences on `fprime`. `fhess_p`
      must compute the hessian times an arbitrary vector. If it is not
      given, finite-differences on `fprime` are used to compute
      it. 
    """
 
    from mystic.tools import Sow, CustomSow
    from mystic.termination import SolutionImprovement
    #stepmon = Sow()
    stepmon = CustomSow('x','y','g','h', x='x', y='fval', \
                         g='Gradient', h='InverseHessian')
    evalmon = Sow()

    solver = NCGSolver(len(x0))
    solver.SetInitialPoints(x0)
    solver.enable_signal_handler()
    solver.SetEvaluationLimits(maxiter,None)
    # Does requiring fprime break abstract_solver interface?
    solver.Solve(func, SolutionImprovement(tolerance=xtol),\
                 EvaluationMonitor=evalmon,StepMonitor=stepmon,\
                 disp=disp, ExtraArgs=args, callback=callback,\
                 epsilon=epsilon, fhess_p=fhess_p,\
                 fhess=fhess, fprime=fprime)
    solution = solver.Solution()

    # code below here pushes output to scipy.optimize interface
    #x = list(solver.bestSolution)
    x = solver.bestSolution
    fval = solver.bestEnergy
    warnflag = 0
    fcalls = len(evalmon.x)
    iterations = len(stepmon.x)

    # Fix me?
    gcalls = solver.gcalls  
    hcalls = solver.hcalls

    allvecs = []
    for i in range(iterations):
       #allvecs.append(list(stepmon.x[i]))
        allvecs.append(stepmon.x[i])

    if iterations >= solver._maxiter:
        warnflag = 1

    if full_output:
        retlist = x, fval, fcalls, gcalls, hcalls, warnflag
        if retall:
            retlist += (allvecs,)
    else:
        retlist = x
        if retall:
            retlist = (x, allvecs)

    return retlist

if __name__=='__main__':
    help(__name__)

# End of file
