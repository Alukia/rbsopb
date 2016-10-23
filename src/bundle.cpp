#include "bundle.h"

/******************************
 * CONSTRUCTORS / DESTRUCTORS *
 ******************************/

void bundle::init(int N) {
	n = N;
	m = 0;
	convex = true;

	VectorXd c(n+1);
	c << VectorXd::Zero(n), 1.0;
	pb.setLinearObjective(c);
}

bundle::bundle(int n, VectorXd& xlb, VectorXd& xub)
	: pb(cplexPB(n+1,xlb,xub)) {
	init(n);
}
bundle::bundle(int n) : pb(cplexPB(n+1)) {
	init(n);
}
bundle::~bundle() {}

/***********
 * METHODS *
 **********/

// save the problem to a file
void bundle::saveProblem(std::string name) {
	pb.saveProblem(name);
}

void bundle::addCut(VectorXd& x, double fx, VectorXd& g) {
	// one new constraint
	m++;
	// compute the linear constraint
	VectorXd A(n+1);
	A << g, -1;
	double b = -(fx - g.dot(x));
	// store this new constraint
	cutsA.conservativeResize(m, n+1);
	cutsA.row(m-1) = A;
	cutsb.conservativeResize(m);
	cutsb(m-1) = b;
	// add it to the optimization program
	pb.addConstraint(A, b, convex ? 'L' : 'G');
}

double bundle::eval(VectorXd& x) {
	// maxᵢ(aᵢᵀx + bᵢ) ⟺ max coeff of (Ax + b)
	VectorXd prod = cutsA.block(0,0,m,n)*x - cutsb;
	return convex ? prod.maxCoeff() : prod.minCoeff();
}

double bundle::solve(VectorXd& x) {
	VectorXd xr(n+1);
	double sol = pb.solve(xr);
	x = xr.segment(0,n);
	return sol;
}

void bundle::addConstraint(VectorXd &a, double b, char zsense) {
	VectorXd ap0(a.size()+1); ap0 << a, 0.;
	pb.addConstraint(ap0, b, zsense);
}

void bundle::getSubgradient(int i, int deb, int nb, VectorXd& g) {
	g = cutsA.block(i,deb,1,nb).transpose();
}

double bundle::getConstant(int i) {
	return cutsb(i);
}
