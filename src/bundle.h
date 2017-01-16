#pragma once

#include <deque>
#include "cplex_pb.h"

/**************************************
 * solve minₓ f(x) for f convex and   *
 * x ∈ ℝⁿ using a cutting-plane model * 
 *                                    *
 * f(x) ⇒ maxᵢ(aᵢᵀx + bᵢ), i = 1,…,m  *
 *                                    *
 **************************************
 * minₓ maxᵢ(aᵢᵀx + bᵢ)               *
 *   ⟺                                *
 * minₓᵣ r                            *
 * s.t. ∀i, r ≥ aᵢᵀx + bᵢ             *
 *   ⟺                                *
 * minₓᵣ (0,…,0,1)ᵀ (x,r)         (1) *
 * s.t. ∀i, (aᵢ,-1)ᵀ (x,r) ≤ -bᵢ      *
 **************************************
 * to obtain an ε-solution the next   *
 * 3 steps are done iteratively       *
 *                                    *
 *   1. solve (1) and store x         *
 *   2. recover f(x) and g ∈ ∂f(x)    *
 *   3. add a new cut aᵀx + b with    *
 *      a = g                         *
 *      b = f(x) - g*x                *
 *                                    *
 * knowing that for all y             *
 *   f(y) ≥ g*(y-x) + f(x), g ∈ ∂f(x) *
 **************************************/

class bundle {

protected:

	int n; // number of variables for x
	int m; // number of cuts
	bool convex; // is f(x) convex or concave
	cplexPB pb;
	std::deque<VectorXd> cutsA; // cutsA(i) = (aᵢ,-1)ᵀ
	std::deque<double> cutsb; // cutsb(i) = -bᵢ

	void init(int N);

public:

	~bundle();

	// initialize a cutting-plane model
	// for a function of n variables
	bundle(int n);

	// initialize a cutting plane model
	// with additionnal lower and upper
	// bound constraints for x 
	bundle(int n, VectorXd&, VectorXd&);

	// change to concave or convex function
	virtual void setConvex();
	virtual void setConcave();

	// save the bundle problem to a file
	void saveProblem(std::string name);

	// add a new cut at position x where
	// f(x) = fx and g ∈ ∂f(x)
	virtual void addCut(VectorXd& x, double fx, VectorXd& g);

	// evaluate the cutting-plane model at x
	double eval(VectorXd& x);

	// find the minimum of the cutting-plane model
	virtual double solve(VectorXd& x);
	// find the minimum of the cutting-plane model
	// and gives dual multipliers
	virtual double solveWithDual(VectorXd& x, VectorXd& xDual);

	// add a new linear constraint on x
	void addConstraint(VectorXd &a, double b, char zsense = 'L');

	// coefficients of the subgradients
	std::deque<VectorXd>* constraints() { return &cutsA; }
	// constant value of the constraints
	std::deque<double>* subgradients() { return &cutsb; }

	// // return the coefficients [deb,deb+nb] of the i-th subgradient
	// void getSubgradient(int i, int deb, int nb, VectorXd& g);
	// // return the constant value of the i-th constraint
	// double getConstant(int i);

	// set verbosity of the solver
	void setVerbose(bool b) { pb.setVerbose(b); }
	// set time limit for each iteration
	void setTimeLimit(double t) { pb.setTimeLimit(t); }

	// return current number of cuts
	int numberOfCuts() { return m; };

};
