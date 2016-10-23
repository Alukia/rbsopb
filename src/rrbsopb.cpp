#include "rrbsopb.h"

/******************************
 * CONSTRUCTORS / DESTRUCTORS *
 ******************************/

rrbsopb::rrbsopb(VectorXi &ni,
	double(*fi)(int,VectorXd&,VectorXd&,VectorXd&),
	double(*psi)(VectorXd&,VectorXd&)) :
	rbsopb(ni, fi, psi), frac(.1), t(1000.)
{
	ws_y = MatrixXd::Zero(ws_nb,n);
}

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
	for(int i = 0; i < m; ++i) {
		int deb = sumpni(i), nb = ni(i);
		VectorXd xi(nb);
		// note : cplex automatically add a .5 factor
		VectorXd A = VectorXd::Constant(nb, 1./t);
		VectorXd b = VectorXd::Zero(nb); // ∑ₗ μₗgᵢˡ
		for(int l = 0; l < k; ++l) {
			VectorXd g(nb);
			psihat.getSubgradient(l,deb,nb,g);
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
	for(int l = 0; l < k; ++l) {
		double cst = -psihat.getConstant(l); // Ψ(xˡ) - <gˡ, xˡ>
		VectorXd g(n);
		psihat.getSubgradient(l,0,n,g);
		gmu(l) = cst + x.dot(g);
		cstpart += mu(l)*cst;
	}
	// end
	return minpart + cstpart;
}

void rrbsopb::warmstart(bundle& bdl) {
	int k = psihat.numberOfCuts();
	for(int l = 0; l < ws_n && l < ws_nb; ++l) {
		VectorXd mul = ws_mu.block(l,0,1,k).transpose();
		VectorXd xmul = ws_x.row(l);
		VectorXd g(k);
		for(int i = 0; i < k; ++i) {
			double cst = -psihat.getConstant(i);
			VectorXd gl(n);
			psihat.getSubgradient(i,0,n,gl);
			g(i) = cst + xmul.dot(gl);
		}
		VectorXd yl = ws_y.row(l);
		double theta = ws_theta(l) + .5/t*(
			(xmul-y).squaredNorm()-(xmul-yl).squaredNorm()
		);
		bdl.addCut(mul, theta, g);
	}
}

void rrbsopb::storageForWS(VectorXd& mu,
	VectorXd& x, double theta)
{
	int k = psihat.numberOfCuts();
	ws_mu.block(ws_k,0,1,k) = mu.transpose();
	ws_x.row(ws_k) = x;
	ws_y.row(ws_k) = y;
	ws_theta(ws_k) = theta;
	ws_k = (ws_k+1)%ws_nb;
	ws_n++;
}

double rrbsopb::bestIterate(MatrixXd& xj,
	MatrixXd& fj, VectorXd& x)
{
	debug("step 2 : Primal recovery (best iterate)");
	int nbCuts = fj.rows();
	x = xj.row(0);
	double best_obj = fj.row(0).sum() + psihat.eval(x)
		+ .5/t*x.squaredNorm();
	for(int j = 1; j < nbCuts; ++j) {
		VectorXd xtmp = xj.row(j);
		double ftmp = fj.row(j).sum() + psihat.eval(xtmp)
			+ .5/t*xtmp.squaredNorm();
		if(ftmp < best_obj) {
			x = xtmp;
			best_obj = ftmp;
		}
	}
	return best_obj - psihat.eval(x);
}

double rrbsopb::dantzigWolfe(MatrixXd& xj,
	MatrixXd& fj, VectorXd& x)
{
	debug("step 2 : Primal recovery (Dantzig-Wolfe like)");
	int L = xj.rows();
	cplexPB pr(L*m+n+1);
	for(int i = 0; i < L*m; ++i)
		pr.setToBinary(i);
	// linear objective
	VectorXd c(L*m+n+1);
	MatrixXd f = fj;
	f.resize(L*m,1);
	c << f, -1./t*y, 1.;
	pr.setLinearObjective(c);
	// quadratic objective
	VectorXd A(L*m+n+1);
	A << VectorXd::Zero(L*m), VectorXd::Constant(n, 1./t), 0.;
	pr.setQuadraticObjective(A);
	// constraint sum for each group
	for(int i = 0; i < m; ++i) {
		VectorXd b = VectorXd::Zero(L*m+n+1);
		b.segment(i*L,L) = VectorXd::Constant(L, 1.);
		pr.addConstraint(b, 1., 'E');
	}
	// constraint for xi
	for(int i = 0; i < m; ++i)
		for(int j = 0; j < ni(i); ++j) {
			int k = sumpni(i)+j;
			VectorXd b = VectorXd::Zero(L*m+n+1);
			b(L*m+k) = -1.;
			b.segment(i*L,L) = xj.col(k);
			pr.addConstraint(b, 0., 'E');
		}
	// copy constraint of psihat bundle
	int K = psihat.numberOfCuts();
	for(int i = 0; i < K; ++i) {
		VectorXd b = VectorXd::Zero(L*m+n+1);
		VectorXd g(n);
		psihat.getSubgradient(i,0,n,g);
		b.segment(L*m,n) = g;
		b(L*m+n) = -1.;
		pr.addConstraint(b, psihat.getConstant(i), 'L');
	}
	// solve
	VectorXd sol(L*m+n+1);
	double tmp = pr.solve(sol);
	x = sol.segment(L*m, n);
	return tmp - psihat.eval(x);
}


bool rrbsopb::stoppingTest(double psi,
	double psihat, double f, double theta, VectorXd& x)
{
	debug("step 4 : Stopping test");
	double diff = objy - (f+psihat);
	double diffrel = diff / abs(objy);
	double quad = .5*1./t *(x-y).squaredNorm();
	debug("     Θₖ(μ) = " << theta);
	debug("      f(x) = " << f);
	debug("    psi(x) = " << psi);
	debug(" psihat(x) = " << psihat);
	debug(" jump dual = " << f + psihat + quad - theta);
	debug("    deltak = " << diffrel);
	debug(" err.  Δₖᴬ = " << f + psi + quad - theta);
	if(objy - (f+psi) >= frac*diff) {
		debug(" update of the stability center");
		y = x;
		objy = f+psi;
	}
	return diffrel < tol;
}

double rrbsopb::solve(VectorXd& x) {
	VectorXd xkp1(n);
	double obj, psixkp1, f;
	// step 0 : Initialization
	obj = initialize(3, xkp1);
	y = xkp1;
	objy = obj;
	// principal loop
	for(int i = 0; i < maxIter; ++i) {
		debug("\nIteration n°" << i);
		// step 1 : Lagrangian dual
		MatrixXd xj;
		MatrixXd fj;
		double theta = maximizeThetak(xj, fj);
		// step 2 : Primal recovery
		if(primal) {
			f = dantzigWolfe(xj, fj, xkp1);
		} else {
			f = bestIterate(xj, fj, xkp1);
		}
		// step 3 : Oracle call
		double psihatskp1 = psihat.eval(xkp1);
		psixkp1 = oracleCall(xkp1);
		// step 4 : Stopping test
		if(stoppingTest(psixkp1, psihatskp1, f, theta, xkp1))
			break;
	}
	x = xkp1;
	obj = f + psixkp1;
	return obj;
}
