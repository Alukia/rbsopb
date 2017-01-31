#include "rrbsopb.h"

/******************************
 * CONSTRUCTORS / DESTRUCTORS *
 ******************************/

rrbsopb::rrbsopb(VectorXi &ni,
	double(*fi)(int,VectorXd&,VectorXd&,VectorXd&),
	double(*psi)(VectorXd&,VectorXd&)) :
	rbsopb(ni, fi, psi), frac(.1), t(1e5) { }

rrbsopb::~rrbsopb() { }

/***********
 * METHODS *
 ***********/

double rrbsopb::thetak(VectorXd& mu, VectorXd& gmu,
	VectorXd& x, VectorXd& fx)
{
	// value : ∑ᵢ(minₓᵢ fᵢ(xᵢ) + <∑ₗ μₗgᵢˡ-1/tₖyₖ, xᵢ>) + xᵀ(1/(2tₖ)I)x
	int k = mu.size();
	fx = VectorXd::Zero(m); // fx(i) = minₓᵢ fᵢ(xᵢ)
	double minpart = 0.;
	std::deque<VectorXd>* grads = psihat.constraints();
	std::deque<double>* csts = psihat.subgradients();
	for(int i = 0; i < m; ++i) {
		int deb = sumpni(i), nb = ni(i);
		VectorXd xi(nb);
		// reminder : cplex automatically add a .5 factor
		VectorXd A = VectorXd::Constant(nb, 1./t);
		VectorXd b = VectorXd::Zero(nb); // ∑ₗ μₗgᵢˡ
		std::deque<VectorXd>::iterator it_g = grads->begin();
		for(int l = 0; l < k; ++l) {
			VectorXd g = it_g++->segment(deb,nb);
			b += mu(l)*g;
		}
		b -= 1./t*y.segment(deb,nb);
		double tmp = fi(i, xi, b, A);
		minpart += tmp;
		fx(i) = tmp - b.dot(xi) - .5/t*xi.squaredNorm(); // + minₓᵢ fᵢ(xᵢ)
		x.segment(deb,nb) = xi;
	}
	// value : constant part ∑ₗ μₗ(Ψ(xˡ) - <gˡ, xˡ>) + 1/(2tₖ)yₖᵀyₖ
	double cstpart = .5/t*y.squaredNorm();
	// subgradient : (Ψ(xˡ) + <gˡ, x(μ)-xˡ>)ₗᵀ ∈ ∂Θₖ(μ)
	std::deque<VectorXd>::iterator it_g = grads->begin();
	std::deque<double>::iterator it_c = csts->begin();
	for(int l = 0; l < k; ++l) {
		double cst = -*(it_c++); // Ψ(xˡ) - <gˡ, xˡ>
		VectorXd g = (it_g++)->segment(0,n);
		gmu(l) = cst + x.dot(g);
		cstpart += mu(l)*cst;
	}
	// end
	return minpart + cstpart;
}

void rrbsopb::warmstart(bundle* bdl) {
	int k = psihat.numberOfCuts();
	std::deque<VectorXd>* grads = psihat.constraints();
	std::deque<double>* csts = psihat.subgradients();
	std::deque<VectorXd>::iterator it_mu = ws_mu.begin();
	std::deque<VectorXd>::iterator it_x = ws_x.begin();
	std::deque<VectorXd>::iterator it_y = ws_y.begin();
	std::deque<double>::iterator it_theta = ws_theta.begin();
	// for each intern steps
	for(int l = 0; l < ws_n; ++l) {
		VectorXd mul = VectorXd::Zero(k);
		for(int i = 0; i < it_mu->rows(); ++i)
			mul(i) = (*it_mu)(i);
		it_mu++;
		VectorXd xmul = *(it_x++);
		VectorXd g(k);
		std::deque<VectorXd>::iterator it_g = grads->begin();
		std::deque<double>::iterator it_c = csts->begin();
		// for each outer steps
		for(int i = 0; i < k; ++i) {
			VectorXd gl = (it_g++)->segment(0,n);
			g(i) = -*(it_c++) + xmul.dot(gl);
		}
		VectorXd yl = *(it_y++);
		double theta = *(it_theta++) + .5/t*(
			(xmul-y).squaredNorm()-(xmul-yl).squaredNorm()
		);
		bdl->addCut(mul, theta, g);
	}
}

void rrbsopb::storageForWS(VectorXd& mu,
	VectorXd& x, double theta)
{
	ws_y.push_back(y);
	rbsopb::storageForWS(mu, x, theta);
}

double rrbsopb::bestIterate(VectorXd& x)
{
	debug(1, "step 2 : Primal recovery (best iterate)");

	std::deque<VectorXd>::iterator xj = pr_xj.begin();
	std::deque<VectorXd>::iterator fj = pr_fj.begin();
	int L = pr_alp.rows();

	x = *(xj++);
	double best_obj = (fj++)->sum() + psihat.eval(x)
		+ .5/t*(x-y).squaredNorm();
	for(int j = 1; j < L; ++j) {
		VectorXd& xtmp = *(xj++);
		double ftmp = (fj++)->sum() + psihat.eval(xtmp)
			+ .5/t*(xtmp-y).squaredNorm();
		if(ftmp < best_obj) {
			x = xtmp;
			best_obj = ftmp;
		}
	}
	double tmp = best_obj - .5/t*(x-y).squaredNorm();
	return best_obj - psihat.eval(x);
}

// double rrbsopb::dantzigWolfe(VectorXd& x)
// {
// 	debug(1, "step 2 : Primal recovery (Dantzig-Wolfe like)");
// 	int L = xj.rows();
// 	cplexPB pr(L*m+n+1);
// 	pr.setTimeLimit(primalMaxTime);
// 	for(int i = 0; i < L*m; ++i)
// 		pr.setToBinary(i);
// 	// linear objective
// 	VectorXd c(L*m+n+1);
// 	MatrixXd f = fj;
// 	f.resize(L*m,1);
// 	c << f, -1./t*y, 1.;
// 	pr.setLinearObjective(c);
// 	// quadratic objective
// 	VectorXd A(L*m+n+1);
// 	A << VectorXd::Zero(L*m), VectorXd::Constant(n, 1./t), 0.;
// 	pr.setQuadraticObjective(A);
// 	// constraint sum for each group
// 	for(int i = 0; i < m; ++i) {
// 		VectorXd b = VectorXd::Zero(L*m+n+1);
// 		b.segment(i*L,L) = VectorXd::Constant(L, 1.);
// 		pr.addConstraint(b, 1., 'E');
// 	}
// 	// constraint for xi
// 	for(int i = 0; i < m; ++i)
// 		for(int j = 0; j < ni(i); ++j) {
// 			int k = sumpni(i)+j;
// 			VectorXd b = VectorXd::Zero(L*m+n+1);
// 			b(L*m+k) = -1.;
// 			b.segment(i*L,L) = xj.col(k);
// 			pr.addConstraint(b, 0., 'E');
// 		}
// 	// copy constraint of psihat bundle
// 	int K = psihat.numberOfCuts();
// 	std::deque<VectorXd>::iterator it_g = psihat.constraints()->begin();
// 	std::deque<double>::iterator it_c = psihat.subgradients()->begin();
// 	for(int i = 0; i < K; ++i) {
// 		VectorXd b = VectorXd::Zero(L*m+n+1);
// 		b.segment(L*m,n) = it_g++->segment(0,n);
// 		b(L*m+n) = -1.;
// 		pr.addConstraint(b, *it_c++, 'L');
// 	}
// 	// solve
// 	VectorXd sol(L*m+n+1);
// 	double tmp = pr.solve(sol);
// 	x = sol.segment(L*m, n);
// 	tmp = tmp + 1./t*y.dot(x) - .5/t*x.squaredNorm();
// 	return tmp - psihat.eval(x);
// }


bool rrbsopb::stoppingTest(double psi,
	double psihat, double f, double theta, VectorXd& x)
{
	debug(1, "step 4 : Stopping test");
	double diff = objy - (f+psihat);
	double diffrel = diff / abs(objy);
	double quad = .5*1./t *(x-y).squaredNorm();
	double gap = f + psihat + quad - theta;
	double delta = f + psi + quad - theta;
	debug(0, "     Θₖ(μ) = " << theta);
	debug(0, "      f(x) = " << f);
	debug(0, "    psi(x) = " << psi);
	debug(0, " psihat(x) = " << psihat);
	debug(0, " jump dual = " << gap);
	debug(0, "    deltak = " << diffrel);
	debug(0, " err.  Δₖᴬ = " << delta);
	if( f+psi <= objy - frac*diff) {
		debug(1, " update of the stability center");
		y = x;
		objy = f+psi;
	}

	// for logs
	log_f.push_back(f);
	log_psi.push_back(psi);
	log_gap.push_back(gap);
	log_delta.push_back(delta);

	return diffrel < tolOuter;
}

double rrbsopb::solve(VectorXd& x) {
	VectorXd xkp1(n);
	double obj, psixkp1, f;
	// step 0 : Initialization
	obj = initialize(3, xkp1);
	y = xkp1;
	objy = obj;
	// principal loop
	for(int i = 0; i < maxOuterIt; ++i) {
		debug(0, "\nIteration n°" << i);
		// step 1 : Lagrangian dual
		double theta = maximizeThetak();
		// step 2 : Primal recovery
		switch(primal) {
			case BEST_ITERATE:
				f = bestIterate(xkp1);
			break;
			case PSEUDO_SCHEDULE:
				f = pseudoSchedule(xkp1);
			break;
		}
		//f = bestIterate(xj, fj, xkp1);
		// step 3 : Oracle call
		double psihatskp1 = psihat.eval(xkp1);
		psixkp1 = oracleCall(xkp1);
		// step 4 : Stopping test
		if(stoppingTest(psixkp1, psihatskp1, f, theta, xkp1))
			break;
	}
	debug(0, "Number of oracle calls : " << nbOracleCall);
	x = xkp1;
	obj = f + psixkp1;
	return obj;
}

/***********
 * SETTERS *
 **********/

#define SETTER(a,b,c) rrbsopb* rrbsopb::set##a(b tmp) { \
    c = tmp; \
    return this; \
}
SETTER(DecreaseFraction, double, frac)
SETTER(ProximalParameter, double, t)
