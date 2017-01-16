#pragma once

/***********************************
 * definition of an optimization   *
 * problem of the following form   *
 *                                 *
 *   minₓ cᵀx + 1/2 xᵀQx           *
 *   s.t. Ax ≤ b                   *
 *                                 *
 * with xᵢ either discrete or real *
 ***********************************/

#include <iostream>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#pragma GCC diagnostic pop
#include "ilcplex/cplex.h"

typedef Eigen::SparseMatrix<double> SpMat;

using namespace Eigen;

class cplexPB {

private:

	int cpx_status; // status of CPLEX
	CPXENVptr env;  // CPLEX env pointer
	CPXLPptr lp;    // CPLEX prb pointer

	int n; // number of variables for x
	int m; // number of linear constraints
	bool isQuadratic; // Q == 0 ?

	void initFromLbUb(int, VectorXd&, VectorXd&);

public:

	cplexPB(std::string);
	cplexPB(int);
	cplexPB(int, double, double);
	cplexPB(int, VectorXd&, VectorXd&);
	cplexPB(MatrixXd& A, VectorXd& b, VectorXd& c,
		VectorXd& xlb, VectorXd& xub);
	~cplexPB();

	int getn() { return n; }

	// change objective type
	void chgobjsen(int sen);

	// save the problem to a file
	void saveProblem(std::string name);

	// set to continuous, binary or integer
	void setType(int i, char t);
	void setToContinuous(int i) { setType(i, 'C'); }
	void setToBinary(int i) { setType(i, 'B'); }
	void setToInteger(int i) { setType(i, 'I'); }

	// set the linear part of the objective
	void setLinearObjective(VectorXd& c);
	void getLinearObjective(VectorXd& c);

	// set the quadratic part of the objective
	// (note that there is a 0.5 factor before Q)
	void setQuadraticObjective(SpMat& Q);
	void setQuadraticObjective(VectorXd& Q); // diagonal

	// add a linear constraint
	void addConstraint(VectorXd& a, double b, char zsense = 'L');

	// set tolerance for convergence
	void setTolerance(double gap);

	// set a time limit for the solver
	void setTimeLimit(double t);

	// set verbose on or off
	void setVerbose(bool b);

	// solve the optimization problem
	double solve(VectorXd& x);
	// solve the optimization problem and recover dual variables
	double solveWithDual(VectorXd& x, VectorXd& xDual);

};
