#pragma once

#include "bundle.h"

/*****************************************
 * stabilization of the classic linear   *
 * cutting-plane model using a           *
 * quadratic term with a stability       *
 * center yₖ and a proximal parameter tₖ *
 *                                       *
 * let Φₖ(x) be the cutting-plane model  *
 * of f(x), we now solve                 *
 *                                       *
 * minₓ Φₖ(x) + 1/(2tₖ) ||x-yₖ||^2       *
 *     ⟺                                 *
 * minₓᵣ + 1/(2tₖ)yₖᵀyₖ             (1)  *
 *       + (-(1/tₖ)yₖ,1)ᵀ (x,r)          *
 *               ⎧ 1/(2tₖ)I  0 ⎫         *
 *       + (x,r)ᵀ⎩    0      0 ⎭(x,r)    *
 * s.t. ∀i, (aᵢ,-1)ᵀ (x,r) ≤ -bᵢ         *
 *                                       *
 *****************************************
 * to obtain an ε-solution the next 4    *
 * steps are done iteratively            *
 *                                       *
 *    1. solve (1) and store x           *
 *    2. recover f(x) and g ∈ ∂f(x)      *
 *    3. add a new cut aᵀx + b with      *
 *       a = g                           *
 *       b = f(x) - g*x                  *
 *    4. if δₖ = f(yₖ)-Φₖ(x) small then  *
 *       let yₖ₊₁ = x or else yₖ₊₁ = yₖ  *
 *                                       *
 *****************************************/

class rbundle : public bundle {

private:

	// δ for change of stability center
	double delta;

	double t; // the proximal parameter
	double objy; // Φₖ(yₖ)
	VectorXd y; // the stability center yₖ

	// change the linear and quadratic part
	// of the optimization problem after
	// a modification of t or y
	void updatePB();

	void init(int n);

public:

	~rbundle();

	// initialize a regularized cutting-plane
	// model for a function of n variables
	rbundle(int n);

	// initialize a regularized cutting-plane
	// model with additionnal lower and upper
	// bound constraints for x 
	rbundle(int n, VectorXd&, VectorXd&);

	virtual void setConvex();
	virtual void setConcave();

	// now also change the stability center
	virtual void addCut(VectorXd&, double, VectorXd&);

	// find the minimum of the regularized
	// cutting-plane model
	virtual double solve(VectorXd& x);

	// set the proximal paramater
	void setT(double tk);

};
