#include "rbundle.h"

/******************************
 * CONSTRUCTORS / DESTRUCTORS *
 ******************************/

void rbundle::init(int n) {
	 delta = .1;
	 t = 1.;
	 y = VectorXd(n);
}

rbundle::rbundle(int n) : bundle(n) { init(n); };

rbundle::rbundle(int n, VectorXd& xlb, VectorXd& xub)
	: bundle(n, xlb, xub) { init(n); }

rbundle::~rbundle() {}

/***********
 * METHODS *
 **********/

void rbundle::updatePB() {
	VectorXd c(n+1);
	c << -1./t*y, 1.;
	pb.setLinearObjective(c);

	VectorXd A(n+1);
	A << (convex?1.:-1.)*VectorXd::Constant(n, 1./t), 0.;
	pb.setQuadraticObjective(A);
}

void rbundle::setConvex() {
	bundle::setConvex();
	updatePB();
}
void rbundle::setConcave() {
	bundle::setConcave();
	updatePB();
}

void rbundle::addCut(VectorXd& x, double fx, VectorXd& g) {
	// test relative difference with the model
	// (f(y)-Φₖ(x)) / |f(y)| ≤ δ
	if( m == 0 || fx < objy - delta*(objy - eval(x)) ) {
		y = x;
		objy = fx;
		updatePB();
	}
	bundle::addCut(x, fx, g);
}

double rbundle::solve(VectorXd& x) {
	VectorXd xr(n+1);
	double sol = pb.solve(xr);
	x = xr.segment(0,n);
	return sol + (convex?1.:-1.)*(1./t*y.dot(x)-.5/t*x.squaredNorm());
}

void rbundle::setT(double tk) {
	t = tk;
	updatePB();
}
