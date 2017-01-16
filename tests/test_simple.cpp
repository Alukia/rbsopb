#include "../src/rrbsopb.h"

/*****************************************
 * solving                               *
 *   minₓ ∑ᵢ aᵢxᵢ^2 + bᵢx + cᵢ + Ψ(x)    *
 *   s.t. xᵢ ∈ [-M,M]                  *
 * with                                  *
 *                                       *
 *   Ψ(x) = sup(d ∈ [μ-k,μ+k]) φ(d,x)    *
 * φ(d,x) = max(c₁(∑ᵢxᵢ-d),c₂(∑ᵢxᵢ-d))   *
 *****************************************/

int N = 8;
double ax[] = {.1,.2,.3,.4,.5,.6,.7,.8};
double bx[] = {-.1,.2,.3,.4,.3,.2,.2,.3};
double cx[] = {.0,.1,.3,.5,.8,.9,.0,.0};
int maxx = 10., minx = -10.;
double f(int i, VectorXd& x, VectorXd& b, VectorXd& A)
{
	double a0 = ax[i]+.5*A(0,0), b0 = bx[i]+b(0), c0 = cx[i];
	x(0) = -.5*b0/a0;
	if(a0 < .0 || x(0) < minx || x(0) > maxx) {
		double v1 = a0*minx*minx + b0*minx + c0;
		double v2 = a0*maxx*maxx + b0*maxx + c0;
		if(v1 < v2) {
			x(0) = minx;
			return v1;
		} else {
			x(0) = maxx;
			return v2;
		}
	}

	return a0*x(0)*x(0) + b0*x(0) + c0;
}

double mu = 10., k = 5.;
double ca1 = -10., ca2 = 2.;
double psi(VectorXd& x, VectorXd& g) {
	double v1 = ca1*(x.sum()-(mu+k)),
	       v2 = ca2*(x.sum()-(mu-k));
	if(v1 > v2) {
		g << VectorXd::Constant(x.size(), ca1);
		return v1;
	} else {
		g << VectorXd::Constant(x.size(), ca2);
		return v2;
	}
}


int main()
{
	VectorXi ni = VectorXi::Constant(N, 1);

	rbsopb pb(ni, f, psi);
	pb.setPrimalRecovery(PSEUDO_SCHEDULE)->setVerbosity(2);

	VectorXd sol(N);
	pb.solve(sol);
	std::cout << "\nFinal result :" << std::endl;
	std::cout << "   x = (" << sol.transpose() << ")" << std::endl;
	double sumfi = 0.;
	for(int i = 0; i < N; ++i) {
		sumfi += ax[i]*sol(i)*sol(i) + bx[i]*sol(i) + cx[i];
	}
	std::cout << "f(x) = " << sumfi << std::endl;
	std::cout << "prod = " << sol.sum() << " vs. μ = " << mu << " & k = " << k << std::endl;
	VectorXd tmp(N);
	double psival = psi(sol, tmp);
	std::cout << "Ψ(x) = " << psival << std::endl;
	std::cout << "f+Ψ  = " << sumfi+psival << std::endl;

	return 0;
}
