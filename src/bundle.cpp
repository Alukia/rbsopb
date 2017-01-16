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

void bundle::setConvex() {
	convex = true;
	pb.chgobjsen(CPX_MIN);
};
void bundle::setConcave() {
	convex = false;
	pb.chgobjsen(CPX_MAX);
};

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
	cutsA.push_back(A);
	cutsb.push_back(b);
	// add it to the optimization program
	pb.addConstraint(A, b, convex ? 'L' : 'G');
}

double bundle::eval(VectorXd& x) {
	// maxᵢ(aᵢᵀx + bᵢ) ⟺ max coeff of (Ax + b)
	std::deque<VectorXd>::iterator it_A = cutsA.begin();
	std::deque<double>::iterator it_b = cutsb.begin();
	VectorXd prod(m);
	for(int i = 0; i < m; ++i) {
		prod(i) = (it_A++)->segment(0,n).dot(x) - *(it_b++);
	}
	return convex ? prod.maxCoeff() : prod.minCoeff();
}

double bundle::solve(VectorXd& x) {
	VectorXd xr = VectorXd::Zero(n+1);
	VectorXd xDual = VectorXd::Zero(m+1);
	double sol = pb.solve(xr);
	x = xr.segment(0,n);
	return sol;
}
double bundle::solveWithDual(VectorXd& x, VectorXd& xDual) {
	VectorXd xr = VectorXd::Zero(n+1);
	VectorXd xDualr = VectorXd::Zero(m+1);
	double sol = pb.solveWithDual(xr, xDualr);
	xDual = -xDualr.segment(1,m);
	x = xr.segment(0,n);
	return sol;
}

void bundle::addConstraint(VectorXd &a, double b, char zsense) {
	VectorXd ap0(a.size()+1); ap0 << a, 0.;
	pb.addConstraint(ap0, b, zsense);
}

// void bundle::getSubgradient(int i, int deb, int nb, VectorXd& g) {
// 	g = cutsA.block(i,deb,1,nb).transpose();
// }

// double bundle::getConstant(int i) {
// 	return cutsb(i);
// }
