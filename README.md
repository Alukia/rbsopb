# Resolution of large scale block-structured robust optimization problems

 This archive provides a simple generic implementation of the (regularized) algorithm described in the paper 

> *"Regularized decomposition of large scale block-structured robust optimization problems"*
> --- by **Wim van Ackooij**, **Nicolas Lebbe** and **Jérôme Malick**.


### Robust block-structured optimization


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

### Getting Started

##### Prerequisites

 This program use the `CPLEX` and `Eigen` libraries and must be linked in the given `Makefile` by modifying respectively the `CPLEXPATH` and `EIGENPATH` variables.
 
##### Installing

The tests are then compiled using a simple
```Shell
 $ make
```
 command in the root directory of the archive.

### Running the tests


Two examples are available in the `tests` directory.

##### Quadratic sum

A simple example is given in `tests/test_simple.cpp` where we solve problem (1) for `m` quadratic functions `fᵢ = aᵢxᵢ² + bᵢxᵢ + cᵢ` with `Xᵢ = [-M,M]` and a two-branch penalization function `Ψ = sup(d ∈ [μ-k,μ+k]) φ(d,x)` with `φ(d,x) = max(c₁(∑ᵢxᵢ-d),c₂(∑ᵢxᵢ-d))`.

```Shell
 $ make
 $ ./bin/test_simple
```

##### Unit commitment

A more usefull example is given in `tests/test_uc.cpp` solving a basic unit commitment problem containing thermal units described below :
 
```
Each unit Uᵢ (i = 0,…,8) use T=24 variables xᵢₜ which correspond to the production of the unit at time t

Functions fᵢ :
the global cost for the schedule xᵢₜ of unit Uᵢ is given using a simple linear formula 

    fᵢ(xᵢₜ) = bᵢᵀxᵢ


Domain Xᵢ ⊂ R²⁴ :
for each unit the production is bounded from 0 to Mᵢ, so

    0 ≤ xᵢₜ ≤ Mᵢ

and between two consecutive time steps ₜ, ₜ₊₁ the production might be modified to at most Kᵢ

    |xᵢₜ-xᵢₜ₊₁| ≤ Kᵢ,  t > 1
    
lastly, each unit start with a given production xᵢ₀

    |xᵢ₀-xᵢ₁| ≤ Kᵢ


Oracle Ψ :
we decide to modelize the  uncertainty set D by a cubic set around the average load μ and a width of k.

for a given demand d ∈ D the cost φ penalize underproduction linearly by a factor c₁ ≤ 0 and surproduction by a factor c₂ ≥ 0 using the following formula

    φ(d,y) = max(c₁(y-d),c₂(y-d))

the Ψ function is then

    Ψ(x) = ∑ₜ sup(dₜ ∈ Dₜ) φ(dₜ,(Ax)ₜ)

with (Ax)ₜ = ∑ᵢ xᵢₜ the production at time step t.
```

```Shell
 $ make
 $ ./bin/test_uc
```

### Documentation

Including the file `rrbsopb.h` you have access to the class `rrbsopb`