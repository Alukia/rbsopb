#pragma once

#include "bundle.h"

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

class rbsopb {

protected:

	int maxIter; // max number of iterations
	double tol;  // tolerance for convergence
	bool verb;   // verbose ?
	bool primal; // true : dantzig-wolfe, false : best iterate
	bool ws;     // warm start ?
	int ws_nb;   // max number of warm-start cuts 
	int ws_k;    // current index to add cuts for ws
	int ws_n;    // current number of warm-start cuts

	int n; // number of variables for x
	int m; // number of blocks

	VectorXi ni;     // ni(i) = number of dimensions for xᵢ
	VectorXi sumpni; // sumpni(i) = n₀ + … + nᵢ

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
	MatrixXd ws_mu;
	MatrixXd ws_x;
	VectorXd ws_theta;

	// warm-start given bundle using previous iterations
	virtual void warmstart(bundle& bdl);

	// storage of informations for warmstart
	virtual void storageForWS(VectorXd&, VectorXd&, double);

	// step 1 : lagrangian dual
	// xj contain all the admissible x obtained during
	// the maximization of the dual with fj(j) = f(xj.row(j))
	double maximizeThetak(MatrixXd& xj, MatrixXd& fj);

	// step 2 : Primal recovery (best iterate)
	// return the best iterate (least value of f+Φ) from
	// the L admissible xₗ found during dual maximization
	virtual double bestIterate(MatrixXd& xj,
		MatrixXd& fj, VectorXd& x);

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
	virtual double dantzigWolfe(MatrixXd& xj,
		MatrixXd& fj, VectorXd& x);

	// step 3 : oracle call
	// return Ψ(x) and add a new cut
	double oracleCall(VectorXd& x);

	// step 4 : stopping test
	// params : Ψ(xₖ₊₁), Φ(xₖ₊₁), f(xₖ₊₁), Θₖ(μₖ₊₁)
	virtual bool stoppingTest(double, double, double, double);

public:

	rbsopb(VectorXi&,
		double(*)(int,VectorXd&,VectorXd&,VectorXd&),
		double(*)(VectorXd&,VectorXd&));

	~rbsopb();

	void setPrimalRecovery(bool b) { primal = b; };
	void setWarmStart(bool b) { ws = b; };

	virtual double solve(VectorXd&);

};
