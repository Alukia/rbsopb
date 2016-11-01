# Resolution of large scale block-structured robust optimization problems

 This archive provides a simple generic implementation of the (regularized) algorithm described in the paper 

> *"Regularized decomposition of large scale block-structured robust optimization problems"*
>
> by **Wim van Ackooij**, **Nicolas Lebbe** and **Jérôme Malick**.


Robust block-structured optimization
--

We consider block-structured optimization problems of the following form

```
  (1)  minₓ ∑ᵢ fᵢ(xᵢ) + Ψ(x)
       s.t. xᵢ ∈ Xᵢ, for all i = 1,…,m
       
  x ∈ X ⊂ Rⁿ, n = ∑ᵢ nᵢ,
  Πᵢ Xᵢ = X with Xᵢ ⊂ Rⁿⁱ compact,
   Ψ : Rⁿ  → R U {+∞} convex,
  fᵢ : Rⁿⁱ → R convex
```

Where the coupling function `Ψ` is subject to uncertainties.  We could for example assume that it depends on a parameter `d` lying in a given uncertainty set `D ⊂ Rᵖ` which leads to

```
  Ψ(x) = sup(d ∈ D) φ(d,x)
```
The algorithm use a bundle-based decomposition methods to tackle large scale instances and only need to solve multiple times each block with an additional linear and quadratic terms, that is to say
```
  minₓᵢ fᵢ(xᵢ) + bᵀx + 1/2 xᵀAx
```

Getting Started
--

### Prerequisites

 This program use the `CPLEX` and `Eigen` libraries and must be linked in the given `Makefile` by modifying respectively the `CPLEXPATH` and `EIGENPATH` variables.
 
### Installing

The tests are then compiled using a simple
```Shell
 $ make
```
 command in the root directory of the archive.

Running the tests
--


Two examples are available in the `tests` directory.

### Quadratic sum

A simple example is given in `tests/test_simple.cpp` where we solve problem (1) for `m` quadratic functions `fᵢ = aᵢxᵢ² + bᵢxᵢ + cᵢ` with `Xᵢ = [-M,M]` and a two-branch penalization function `Ψ = sup(d ∈ [μ-k,μ+k]) φ(d,x)` with `φ(d,x) = max(c₁(∑ᵢxᵢ-d),c₂(∑ᵢxᵢ-d))`.

To run this example :
```Shell
 $ make
 $ ./bin/test_simple
```

### Unit commitment

A more usefull example is given in `tests/test_uc.cpp` solving a basic unit commitment problem containing thermal units described below.
Each unit `Uᵢ (i = 0,…,8)` use `T=24` (the next 24 hours) variables `xᵢₜ` which correspond to the production of the unit at time `t`

**Functions `fᵢ` :**

The global cost for the schedule `xᵢₜ` of unit `Uᵢ` is given using a simple linear formula 
```
fᵢ(xᵢₜ) = bᵢᵀxᵢ
```

**Domain `Xᵢ ⊂ R²⁴` :**

For each unit the production is bounded from 0 to `Mᵢ`, so
```
0 ≤ xᵢₜ ≤ Mᵢ
```
and between two consecutive time steps t, t+1 the production might be modified to at most `Kᵢ`.
```
|xᵢₜ-xᵢₜ₊₁| ≤ Kᵢ,  t > 1
```
    
Lastly, each unit start with a given production `xᵢ₀`
```
|xᵢ₀-xᵢ₁| ≤ Kᵢ
```

**Oracle Ψ :**

We decide to modelize the  uncertainty set `D` by a cubic set around the average load `μ` and a width of `k`.

For a given demand `d ∈ D` the cost `φ` penalize underproduction linearly by a factor `c₁ ≤ 0` and surproduction by a factor `c₂ ≥ 0` using the following formula
```
φ(d,y) = max(c₁(y-d),c₂(y-d)),
```
the `Ψ` function is then
```
Ψ(x) = ∑ₜ sup(dₜ ∈ Dₜ) φ(dₜ,(Ax)ₜ)
```
with `(Ax)ₜ = ∑ᵢ xᵢₜ` the production at time step `t`.

To run this example :
```Shell
 $ make
 $ ./bin/test_uc
```

Documentation
--

Including the file `rrbsopb.h` you have access to the class `rbsopb` and `rrbsopb` whose unique constructors requires 3 parameters :
```C++
rrbsopb(VectorXi &ni, double(*fi)(int,VectorXd&,VectorXd&,VectorXd&), double(*psi)(VectorXd&,VectorXd&));
//  ni : vector of m integers containing the number of variables of each block
//  fi : pointer to a function of 4 variables : int i, VectorXd& x, VectorXd& b, VectorXd& A
//        i : the number of a block
//        x : the variable x which minimize fᵢ(xᵢ) + bᵀx + 1/2 xᵀdiag(A)x
//        b : vector corresponding to an additional linear term
//        A : vector corresponding to the diagonal of the matrix of an additional quadratic term
//       the function fi should return the value of minₓᵢ fᵢ(xᵢ) + bᵀx + 1/2 xᵀdiag(A)x
// psi : pointer to a function of 2 variables : VectorXd& x, VectorXd& g
//        x : point of evaluation of the function Ψ
//        g : a subgradient of Ψ(x)
//       the function psi should return the value of Ψ(x)
```
*The files in the `tests` directory gives two simple examples using this constructor.*

Once a `rrbsopb` (or `rbsopb`) is defined you can specify some parameters using the following methods :

```C++
// set type of primal recovery
// 0 : best-iterate, 1 : Dantzig-Wolfe-like
rbsopb* setPrimalRecovery(bool b);
```

```C++
// set if warm start is on or off
rbsopb* setWarmStart(bool b);
```

```C++
// set max number of cuts for the warm start
rbsopb* setMaxCutsWS(int nb);
```

```C++
// set maximum number of iterations for the bundle
rbsopb* setMaxInternIt(int nb);
```

```C++
// set maximum number of iterations for the algorithm
rbsopb* setMaxOuterIt(int nb);
```

```C++
// set precision for the bundle maximizing Θₖ(μ)
rbsopb* setInternPrec(double gap);
```

```C++
// set precision for the algorithm
rbsopb* setOuterPrec(double gap);
```

```C++
// set a time limit for the solver of each bundle iteration
rbsopb* setTimeLimit(double time);
```

```C++
// set a time limit for the dantzig-wolfe recovery
rbsopb* setPrimalTimeLimit(double time);
```

```C++
// set verbosity level
// 0 : no text displayed
// 1 : general information for each steps
// 2 : complete information for each inner and outer steps
rbsopb* setVerbosity(int n);
```

Then using the method `double solve(VectorXd& x)` you can solve the optimization problem, returning the final objective with the corresponding variable `x`.
