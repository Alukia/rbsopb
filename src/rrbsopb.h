#pragma once

#include "rbsopb.h"

/*********************************************
 * solve a robust block-structured           *
 * optimization problem using a regularizion *
 *********************************************
 * replace the Lagrangian Θₖ(μ) with the     *
 * solution of the problem                   *
 *                                           *
 * (1)   minₓ f(x) + 1/(2tₖ)||x-yₖ||^2 + ⋯   *
 *            ⋯ <∑ₗ μₗgˡ, x> + ⋯             *
 *            ⋯ ∑ₗ μₗ(Ψ(xˡ) - <gˡ, xˡ>)      *
 *       s.t. x ∈ X                          *
 *                                           *
 * thus solving the m sub-problems           *
 *                                           *
 * (3)   minₓᵢ fᵢ(xᵢ) + ⋯                    *
 *             ⋯ <∑ₗ μₗgᵢˡ-1/tₖyₖ, xᵢ> + ⋯   *
 *             ⋯ xᵀ(1/(2tₖ)I)x               *
 *       s.t. xᵢ ∈ Xᵢ                        *
 *                                           *
 * gives Θₖ(μ) and as before, a sub-gradient *
 *********************************************
 * The numerical algorithm is therefore      *
 * almost identical except for a new step    *
 * which appear between step 3 and 4         *
 *                                           *
 *  step 3bis : Descent condition            *
 *    let m > 0 a constant and δₖ by         *
 *         δₖ = f(yₖ)+Ψ(yₖ) - f(x)+Φ(x)      *
 *    if f(yₖ)+Ψ(yₖ) - f(x)+Ψ(x) ≥ m δₖ      *
 *    then let yₖ₊₁ = x or else yₖ₊₁ = yₖ    *
 *********************************************/

class rrbsopb : public rbsopb {

protected:

	double frac; // fraction for predicted decrease
	double t; // the proximal parameter
	VectorXd y; // the stability center yₖ
	double objy; // (f+Ψ)(yₖ)

	// variables for warm-start
	std::deque<VectorXd> ws_y;

	// compute Θₖ(μ)
	virtual double thetak(VectorXd& mu, VectorXd& gmu,
		VectorXd& x, VectorXd& fx);

	// warm-start
	virtual void warmstart(bundle* bdl);
	// storage for warmstart
	virtual void storageForWS(VectorXd&, VectorXd&, double);

	// step 2 : return convex combination using dual
	// multipliers in case of convex admissible sets
	//virtual double pseudoSchedule(VectorXd& x);

	// // step 2 : return best iterate (least value of f+Φ+ 1/(2tₖ)||x-yₖ||^2)
	virtual double bestIterate(VectorXd& x);

	// // step 2 : Primal recovery (Dantzig-Wolfe like)
	// virtual double dantzigWolfe(VectorXd& x);

	// step 3bis & 4 : descent condition & stopping test
	virtual bool stoppingTest(double psi, double psihat,
		double f, double theta, VectorXd& x);

public:

	rrbsopb(VectorXi &ni,
		double(*fi)(int,VectorXd&,VectorXd&,VectorXd&),
		double(*psi)(VectorXd&,VectorXd&));

	~rrbsopb();

	// set decreased fraction for update of stability center
	rrbsopb* setDecreaseFraction(double);

	// set proximal
	rrbsopb* setProximalParameter(double);


	virtual double solve(VectorXd&);

};
