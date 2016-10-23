#include "rbsopb.h"

/******************************
 * CONSTRUCTORS / DESTRUCTORS *
 ******************************/

rbsopb::rbsopb(VectorXi &ni,
	double(*fi)(int,VectorXd&,VectorXd&,VectorXd&),
	double(*psi)(VectorXd&,VectorXd&)) :
	maxIter(1000), tol(1e-5), verb(false), primal(true),
	ws(true), ws_nb(100), ws_k(0), ws_n(0),
	ni(ni), fi(fi), psi(psi), psihat(ni.sum())
{
	m = ni.size();
	sumpni = VectorXi(m+1);
	sumpni(0) = 0;
	n = 0;
	for(int i = 0; i < m; ++i) {
		n += ni(i);
		sumpni(i+1) = n;
	}
	ws_mu = MatrixXd::Zero(ws_nb, maxIter);
	ws_x = MatrixXd::Zero(ws_nb,n);
	ws_theta = VectorXd::Zero(ws_nb);
}

rbsopb::~rbsopb() { }

/***********
 * METHODS *
 **********/

double rbsopb::f(VectorXd& x, VectorXd& b, VectorXd& A) {
	double sum = 0.;
	// #pragma omp parallel for num_threads(8) shared(x)
	for(int i = 0; i < m; ++i) {
		int deb = sumpni(i), nb = ni(i);
		VectorXd xi(nb);
		VectorXd bi(nb); bi << b.segment(deb,nb);
		VectorXd Ai(nb); Ai << A.segment(deb,nb);
		sum += fi(i, xi, bi, Ai);
		x.segment(deb,nb) = xi;
	}
	return sum;
}

double rbsopb::initialize(int N, VectorXd& x) {
	debug("step 0 : Initialization");
	// initialize the cutting-plane model of Ψ
	// using N feasible points given by fᵢ(xᵢ+bᵢ)
	// with bᵢ random vectors
	VectorXd A = VectorXd::Zero(n);
	double obj;
	for(int i = 0; i < N; ++i) {
		// we recover an x ∈ X
		VectorXd b = VectorXd::Random(n);
		obj = f(x, b, A);
		// we compute Ψ(x) and g ∈ ∂Ψ(x)
		VectorXd gpsix(n);
		double psix = psi(x, gpsix);
		// we enrich the model of Ψ
		psihat.addCut(x, psix, gpsix);
	}
	debug(" " << N << " cut(s) added");
	return obj + psihat.eval(x);
}

double rbsopb::thetak(VectorXd& mu, VectorXd& gmu,
	VectorXd& x, VectorXd& fx)
{
	// value : minimizing part ∑ᵢ(minₓᵢ fᵢ(xᵢ) + <∑ₗ μₗgᵢˡ, xᵢ>)
	int k = mu.size();
	fx = VectorXd::Zero(m); // fx(i) = minₓᵢ fᵢ(xᵢ)
	double minpart = 0.;
	for(int i = 0; i < m; ++i) {
		int deb = sumpni(i), nb = ni(i);
		VectorXd xi(nb);
		VectorXd A = VectorXd::Zero(nb);
		VectorXd b = VectorXd::Zero(nb); // ∑ₗ μₗgᵢˡ
		for(int l = 0; l < k; ++l) {
			VectorXd g(nb);
			psihat.getSubgradient(l,deb,nb,g);
			b += mu(l)*g;
		}
		double tmp = fi(i, xi, b, A);
		minpart += tmp;
		fx(i) = tmp - b.dot(xi); // + minₓᵢ fᵢ(xᵢ)
		x.segment(deb,nb) = xi;
	}
	// value : constant part ∑ₗ μₗ(Ψ(xˡ) - <gˡ, xˡ>)
	double cstpart = 0;
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

void rbsopb::warmstart(bundle& bdl) {
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
		bdl.addCut(mul, ws_theta(l), g);
	}
}

void rbsopb::storageForWS(VectorXd& mu,
	VectorXd& x, double theta)
{
	int k = psihat.numberOfCuts();
	ws_mu.block(ws_k,0,1,k) = mu.transpose();
	ws_x.row(ws_k) = x;
	ws_theta(ws_k) = theta;
	ws_k = (ws_k+1)%ws_nb;
	ws_n++;
}

double rbsopb::maximizeThetak(MatrixXd& xj, MatrixXd& fj)
{
	debug("step 1 : Lagrangian dual");
	int k = psihat.numberOfCuts();
	VectorXd z = VectorXd::Zero(k);
	VectorXd o = VectorXd::Constant(k,  1.);
	bundle bundleThetaK(k, z, o);
	bundleThetaK.setConcave();
	bundleThetaK.addConstraint(o, 1., 'E');

	xj = MatrixXd::Zero(maxIter,n);
	fj = MatrixXd::Zero(maxIter,m);
	int nbj = 0;

	if(ws && ws_n != 0) {
		// We want to warm-start the bundle with
		// informations of the last bundle.
		// For each previous μₗ we have
		// 
		//    Θₖ(μₗ) = Θₖ₊₁([μₗ,0])
		//
		// (Ψ(xⁱ) + <gⁱ, x(μˡ)-xⁱ>)ᵢ ∈ ∂Θₖ₊₁([μₗ,0])
		// with i = 1,…,k+1
		warmstart(bundleThetaK);
	} else {
		// we can initialize the cutting-plane model of
		// Θₖ(μ) with only one cut because μ bounded due
		// to the constraints μ ∈ [0,1]ᵏ so obj = r > cst
		VectorXd mu0 = VectorXd::Constant(k, 1.0/k);
		VectorXd g0(k), x0(n);
		VectorXd fx0;
		double theta0 = thetak(mu0, g0, x0, fx0);

		xj.row(nbj) = x0;
		fj.row(nbj) = fx0;
		nbj++;

		debug(" " << "(0)" << " Θₖ(μ) = " << theta0);
		bundleThetaK.addCut(mu0, theta0, g0);
	}

	VectorXd mujp1(k), gjp1(k), xjp1(n);
	double theta;
	for(int j = 1; j < maxIter; ++j) {
		bundleThetaK.solve(mujp1);
		VectorXd fxjp1;
		theta = thetak(mujp1, gjp1, xjp1, fxjp1);

		xj.row(nbj) = xjp1;
		fj.row(nbj) = fxjp1;
		nbj++;

		double thetahat = bundleThetaK.eval(mujp1);
		double rel = (thetahat-theta)/abs(thetahat);
		debug(" "<< "("<<j<<")" <<
			" Θₖ(μ) = " << theta << ", hat(Θₖ)(μ) = "
			<< thetahat << ", rel. dist = " << rel);
		if(rel < tol) break;
		bundleThetaK.addCut(mujp1, theta, gjp1);

		// storage of value for warm-start
		if(ws) {
			storageForWS(mujp1, xjp1, theta);
		}
	}
	xj.conservativeResize(nbj+1, n);
	fj.conservativeResize(nbj+1, m);
	return theta;
}

double rbsopb::bestIterate(MatrixXd& xj,
	MatrixXd& fj, VectorXd& x)
{
	debug("step 2 : Primal recovery (best iterate)");
	int nbCuts = fj.rows();
	x = xj.row(0);
	double best_obj = fj.row(0).sum() + psihat.eval(x);
	for(int j = 1; j < nbCuts; ++j) {
		VectorXd xtmp = xj.row(j);
		double ftmp = fj.row(j).sum() + psihat.eval(xtmp);
		if(ftmp < best_obj) {
			x = xtmp;
			best_obj = ftmp;
		}
	}
	return best_obj - psihat.eval(x);
}

double rbsopb::dantzigWolfe(MatrixXd& xj,
	MatrixXd& fj, VectorXd& x)
{
	debug("step 2 : Primal recovery (Dantzig-Wolfe like)");
	int L = xj.rows();
	cplexPB pr(L*m+n+1);
	for(int i = 0; i < L*m; ++i)
		pr.setToBinary(i);
	VectorXd c(L*m+n+1);
	MatrixXd f = fj;
	f.resize(L*m,1);
	c << f, VectorXd::Zero(n), 1.;
	pr.setLinearObjective(c);
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

double rbsopb::oracleCall(VectorXd& x) {
	debug("step 3 : Oracle call");
	VectorXd g(n);
	double psix = psi(x, g);
	psihat.addCut(x, psix, g);
	return psix;
}

bool rbsopb::stoppingTest(double psi,
	double psihat, double f, double theta)
{
	debug("step 4 : Stopping test");
	double diff = psi - psihat;
	double diffrel = diff / abs(f+psi);
	debug("     Θₖ(μ) = " << theta);
	debug("      f(x) = " << f);
	debug("    psi(x) = " << psi);
	debug(" psihat(x) = " << psihat);
	debug(" jump dual = " << f + psihat - theta);
	debug(" rel. dist = " << diffrel);
	debug(" err.  Δₖᴬ = " << f + psi - theta);
	return diffrel < tol;
}

double rbsopb::solve(VectorXd& x) {
	VectorXd xkp1(n);
	double obj, psixkp1, f;
	// step 0 : Initialization
	obj = initialize(3, x);
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
		if(stoppingTest(psixkp1, psihatskp1, f, theta))
			break;
	}
	x = xkp1;
	obj = f + psixkp1;
	return obj;
}
