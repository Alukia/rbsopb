#pragma once

#include <fstream>
#include "rbundle.h"

/*********************************************
 * solve a robust block-structured           *
 * optimization problem of the form          *
 *                                           *
 * (1)   minₓ ∑ᵢ fᵢ(xᵢ) + Ψ(x)               *
 *       s.t. xᵢ ∈ Xᵢ, for all i = 1,…,m     *
 *                                           *
 * xᵢ ∈ ℝⁿⁱ, ∑ᵢ nᵢ = n,                      *
 *  Ψ : ℝⁿ  → ℝ U {+∞} convex,               *
 * fᵢ : ℝⁿⁱ → ℝ convex,                      *
 * Xᵢ ⊂ ℝⁿⁱ compact, Πᵢ Xᵢ = X               *
 *********************************************
 * To solve (1) we apply a cutting-plane     *
 * model to Ψ(x) named Φ(x)                  *
 *                                           *
 *    Φ(x) = maxᵢ(aᵢᵀx + bᵢ), i = 1,…,k      *
 *                                           *
 * We then move to the Lagrangian giving     *
 *                                           *
 *    minμ Θₖ(μ)                             *
 *    s.t μ ∈ [0,1]ᵏ and ∑ₗ μₗ = 1           *
 *                                           *
 * with Θₖ(μ) solution of                    *
 *                                           *
 * (2)   minₓ f(x) + <∑ₗ μₗgˡ, x> + ⋯        *
 *            ⋯ ∑ₗ μₗ(Ψ(xˡ) - <gˡ, xˡ>)      *
 *       s.t. x ∈ X                          *
 *                                           *
 * thus solving the m sub-problems           *
 *                                           *
 * (3)   minₓᵢ fᵢ(xᵢ) + <∑ₗ μₗgᵢˡ, xᵢ>       *
 *       s.t. xᵢ ∈ Xᵢ                        *
 *                                           *
 * gives Θₖ(μ), a sub-gradient               *
 *                                           *
 *    (Ψ(xˡ) + <gˡ, x(μ)-xˡ>)ₗᵀ ∈ ∂Θₖ(μ)     *
 *                                           *
 * and a primal solution x(μ) ∈ X            *
 *********************************************
 * The numerical algorithm is a follows      *
 *                                           *
 * step 0 : Initialization                   *
 *   generate k (at least 3) initial cuts    *
 *   for the cutting-plane model of Ψ.       *
 *                                           *
 * step 1 : Lagrangian dual                  *
 *   use a bundle method to solve (2) by     *
 *   solving the m sub-problems (3).         *
 *                                           *
 * step 2 : Primal Recovery                  *
 *   exploit the dual information generated  *
 *   to recover a solution xᵏ⁺¹ ∈ X.         *
 *                                           *
 * step 3 : Oracle call                      *
 *   call Ψ at xᵏ⁺¹ to compute a value and   *
 *   a subgradient to enrich the model of Ψ. *
 *                                           *
 * step 4 : Stopping test                    *
 *   if Ψ(xᵏ⁺¹) − Φ(xᵏ⁺¹) ≤ δₛₜₒₚ then stop  *
 *   otherwise k ← k+1 and return in step 2  *
 *********************************************/

// primal recovery types
enum prType {PSEUDO_SCHEDULE, BEST_ITERATE, DANTZIG_WOLFE};

#define debug(n,s) if(verbosity >= n) std::cout << s << std::endl

class rbsopb {

protected:

	int maxInternIt; // max number of iterations for the bundle
	int maxOuterIt;  // max number of iterations for the algorithm
	double tolIntern; // tolerance for convergence of the bundle
	double tolOuter;  // tolerance for convergence of the algorithm
	double maxTime; // max time for each steps of the bundle
	int verbosity; // verbosity level
	double primalMaxTime; // primal recovery max time to solve
	prType primal; // pseudo-schedule, best_iterate or dantzig-wolfe
	bool ws;  // warm start ?
	int ws_n; // number of warm-start cuts

	int n; // number of variables for x
	int m; // number of blocks

	VectorXi ni;     // ni(i) = number of dimensions for xᵢ
	VectorXi sumpni; // sumpni(i) = n₀ + … + nᵢ

	int nbOracleCall; // number of intern bundle steps

	// the function f which for each integer 0 ≤ i ≤m-1 
	// solve minₓᵢ fᵢ(xᵢ) + bᵀxᵢ + 1/2 xᵢᵀ A xᵢ 
	double(*fi)(int,VectorXd&,VectorXd&,VectorXd&);

	// the function Ψ which give f(x) and g ∈ ∂f(x)
	double (*psi)(VectorXd&,VectorXd&);

	// cutting-plane model of psi
	bundle psihat;

	// compute ∑ᵢ fᵢ(xᵢ) + bᵀxᵢ + 1/2 xᵢᵀ diag(A) xᵢ
	double f(VectorXd& x, VectorXd& b, VectorXd& A);

	// compute Θₖ(μ)
	virtual double thetak(VectorXd& mu, VectorXd& gmu,
		VectorXd& x, VectorXd&);

	// step 0 : initialization
	// initialize the cutting-plane of Ψ with N random cuts
	double initialize(int N, VectorXd&);

	// variables for warm-start
	std::deque<VectorXd> ws_mu;
	std::deque<VectorXd> ws_x;
	std::deque<double> ws_theta;

	// warm-start given bundle using previous iterations
	virtual void warmstart(bundle* bdl);

	// storage of informations for warmstart
	virtual void storageForWS(VectorXd&, VectorXd&, double);

	// variables for primal recovery
	std::deque<VectorXd> pr_xj;
	std::deque<VectorXd> pr_fj;
	VectorXd pr_alp;

	// step 1 : lagrangian dual
	double maximizeThetak();

	// step 2 : Primal recovery (pseudo schedule)
	// (!) only if all the Xᵢ are convex
	// use a convex combinations of the xᵢ found during
	// dual maximization with coefficients given by the
	// dual multipliers
	virtual double pseudoSchedule(VectorXd& x);

	// step 2 : Primal recovery (best iterate)
	// return the best iterate (least value of f+Φ) from
	// the L admissible xₗ found during dual maximization
	virtual double bestIterate(VectorXd& x);

	// step 2 : Primal recovery (Dantzig-Wolfe like)
	// knowing L admissible xˡ found during dual maximization
	// we search the best xᵢˡ for each block, this imply to
	// solve the following MILP problem
	// 
	//    minᵤₓᵣ ∑ₗ ∑ᵢ uᵢₗ fᵢ(xᵢˡ) + r
	//    s.t. xᵢ = ∑ₗ uᵢₗ xᵢˡ
	//         ∑ₗ uᵢₗ = 1
	//         uᵢₗ ∈ {0,1}
	//         ∀i, (aᵢ,-1)ᵀ (x,r) ≤ -bᵢ
	//         
	// for which very efficient greedy heuristics exist.
	//virtual double dantzigWolfe(VectorXd& x);

	// step 3 : oracle call
	// return Ψ(x) and add a new cut
	double oracleCall(VectorXd& x);

	// step 4 : stopping test
	// params : Ψ(xₖ₊₁), Φ(xₖ₊₁), f(xₖ₊₁), Θₖ(μₖ₊₁)
	virtual bool stoppingTest(double, double, double, double);

	std::deque<double> log_f, log_psi, log_gap, log_delta;
	std::deque<int> log_nb;

public:

	rbsopb(VectorXi&,
		double(*)(int,VectorXd&,VectorXd&,VectorXd&),
		double(*)(VectorXd&,VectorXd&));

	~rbsopb();

	// set type of primal recovery
	// 0 : best-iterate, 1 : Dantzig-Wolfe-like
	rbsopb* setPrimalRecovery(prType b);

	// set if warm start is on or off
	rbsopb* setWarmStart(bool b);

	// set max number of cuts for the warm start
	rbsopb* setMaxCutsWS(int nb);

	// set maximum number of iterations for the bundle
	rbsopb* setMaxInternIt(int nb);

	// set maximum number of iterations for the algorithm
	rbsopb* setMaxOuterIt(int nb);

	// set precision for the bundle maximizing Θₖ(μ)
	rbsopb* setInternPrec(double gap);

	// set precision for the algorithm
	rbsopb* setOuterPrec(double gap);

	// set a time limit for the solver of each bundle iteration
	rbsopb* setTimeLimit(double time);

	// set a time limit for the dantzig-wolfe recovery
	rbsopb* setPrimalTimeLimit(double time);

	// set if verbose or not
	// 0 : no text displayed
	// 1 : general information for each steps
	// 2 : complete information for each inner and outer steps
	rbsopb* setVerbosity(int n);

	virtual double solve(VectorXd&);

	// write logs about values of f, Ψ, the duality gap,
	// the number of inner iterations and the delta for
	// the stopping test
	void writeLogs(std::string name);

};
