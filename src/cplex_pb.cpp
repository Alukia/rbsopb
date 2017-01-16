#include "cplex_pb.h"

/******************************
 * CONSTRUCTORS / DESTRUCTORS *
 ******************************/

void cplexPB::initFromLbUb(int N, VectorXd& xlb, VectorXd& xub) {
	n = N;
	m = 0;
	isQuadratic = false;
	env = CPXopenCPLEX(&cpx_status);
	lp = CPXcreateprob(env, &cpx_status, "");
	VectorXd c = VectorXd::Constant(n, 0.);
	cpx_status = CPXcopylp(env, lp, n, 0, CPX_MIN, c.data(),
		NULL, NULL, NULL, NULL, NULL, NULL,
		xlb.data(), xub.data(), NULL);
	cpx_status = CPXsetdblparam(env, CPX_PARAM_EPRHS, 1e-6);
	cpx_status = CPXchgprobtype(env, lp, CPXPROB_LP);
}

cplexPB::cplexPB(std::string file)
	: isQuadratic(false)
{
	env = CPXopenCPLEX(&cpx_status);
	lp = CPXcreateprob(env, &cpx_status, file.c_str());
	cpx_status = CPXreadcopyprob(env, lp, file.c_str(), NULL);
	n = CPXgetnumcols(env, lp);
	m = CPXgetnumrows(env, lp);
}

cplexPB::cplexPB(int n, double xlb, double xub) {
	VectorXd lb = VectorXd::Constant(n, xlb);
	VectorXd ub = VectorXd::Constant(n, xub);
	initFromLbUb(n, lb, ub);
}

cplexPB::cplexPB(int n, VectorXd& xlb, VectorXd& xub) {
	VectorXd xlbr(n);
	if(xlb.size() < n) {
		xlbr << xlb, VectorXd::Constant(n-xlb.size(), -CPX_INFBOUND);
	} else xlbr = xlb;
	VectorXd xubr(n);
	if(xub.size() < n) {
		xubr << xub, VectorXd::Constant(n-xub.size(),  CPX_INFBOUND);
	} else xubr = xub;
	initFromLbUb(n, xlbr, xubr);
}

cplexPB::cplexPB(int n) {
	VectorXd xlb = VectorXd::Constant(n, -CPX_INFBOUND);
	VectorXd xub = VectorXd::Constant(n,  CPX_INFBOUND);
	initFromLbUb(n, xlb, xub);
}

cplexPB::cplexPB(MatrixXd &A, VectorXd &b,
	VectorXd &c, VectorXd &xlb, VectorXd &xub)
	: isQuadratic(false)
{
	n = c.rows();
	m = A.rows();
	env = CPXopenCPLEX(&cpx_status);
	lp = CPXcreateprob(env, &cpx_status, "");

	// indice du premier élément de la i-ème ligne
	VectorXi Amat_beg(m);
	for(int i = 0; i < m; ++i) Amat_beg(i) = n*i;

	// nombre éléments non nuls dans la ligne i
	VectorXi Amat_cnt(m);
	Amat_cnt.fill(n);

	// type de contrainte, 'L' pour <=
	Matrix<char,Dynamic,1> zsense(m);
	zsense.fill('L');

	// numéro de ligne de l'élément i
	VectorXi Amat_ind(m*n);
	for(int i = 0, k = 0; i < m; ++i)
		for(int j = 0; j < n; ++j)
			Amat_ind(k++) = i;

	cpx_status = CPXcopylp(env, lp, n, m, CPX_MIN, c.data(), b.data(),
		zsense.data(), Amat_beg.data(), Amat_cnt.data(),
		Amat_ind.data(), A.data(),
		xlb.data(), xub.data(), NULL);
	cpx_status = CPXsetdblparam(env, CPX_PARAM_EPRHS, 1e-6);
	cpx_status = CPXchgprobtype(env, lp, CPXPROB_LP);
}

cplexPB::~cplexPB() {
	if(n != 0) {
		cpx_status = CPXfreeprob(env, &lp);
		cpx_status = CPXcloseCPLEX(&env);
	}
}

/***********
 * METHODS *
 **********/

void cplexPB::chgobjsen(int sen) {
	CPXchgobjsen(env, lp, sen);
}

void cplexPB::saveProblem(std::string name) {
	cpx_status = CPXwriteprob(env, lp, name.c_str(), "LP");
}

void cplexPB::setType(int i, char t) {
	cpx_status = CPXchgctype(env, lp, 1, &i, &t);
}

void cplexPB::setLinearObjective(VectorXd &b) {
	VectorXi ind(n);
	for(int i = 0; i < n; ++i) ind(i) = i;
	cpx_status = CPXchgobj(env, lp, n, ind.data(), b.data());
}
void cplexPB::getLinearObjective(VectorXd& c) {
	c.resize(n);
	cpx_status = CPXgetobj(env, lp, c.data(), 0, n-1);
}

void cplexPB::setQuadraticObjective(SpMat& Q) {
	int nnz = Q.nonZeros();
	isQuadratic = nnz != 0;
	if(!isQuadratic) return;
	VectorXd Qval(nnz);
	VectorXi Qind(nnz), Qbeg(nnz);
	VectorXi Qcnt = VectorXi::Zero(nnz);

	for(int i = 0, k = 0; i < Q.outerSize(); ++i)
	for(SpMat::InnerIterator it(Q,k); it; ++it) {
		Qval(k) = it.value();
		Qbeg(k) = it.col();
		Qind(k) = it.row();
		Qcnt(i)++;
		k++;
	}

	cpx_status = CPXcopyquad(env, lp, Qbeg.data(),
		Qcnt.data(), Qind.data(), Qval.data());
}
void cplexPB::setQuadraticObjective(VectorXd& Q) {
	isQuadratic = !Q.isZero();
	if(!isQuadratic) return;
	cpx_status = CPXcopyqpsep(env, lp, Q.data());
}

void cplexPB::addConstraint(VectorXd &a, double b, char zsense) {
	int rmatbeg = 0;
	int rcnt = 1;
	int ccnt = 0;
	VectorXi rmatind(n);
	for(int i = 0; i < n; ++i) rmatind(i) = i;
	cpx_status = CPXaddrows(env, lp, ccnt, rcnt, n, &b, &zsense,
		&rmatbeg, rmatind.data(), a.data(), NULL, NULL);
	m++;
}

double cplexPB::solve(VectorXd& x) {
	int nbBinary = CPXgetnumbin(env, lp);
	int nbInteger = CPXgetnumint(env, lp);
	int nbMixedInteger = nbBinary + nbInteger;
	if(isQuadratic) {
		if(nbMixedInteger != 0) {
			cpx_status = CPXchgprobtype(env, lp, CPXPROB_MIQP);
			cpx_status = CPXmipopt(env, lp);
		} else {
			cpx_status = CPXchgprobtype(env, lp, CPXPROB_QP);
			cpx_status = CPXqpopt(env, lp);
		}
	} else {
		if(nbMixedInteger != 0) {
			cpx_status = CPXchgprobtype(env, lp, CPXPROB_MILP);
			cpx_status = CPXmipopt(env, lp);
		} else {
			cpx_status = CPXchgprobtype(env, lp, CPXPROB_LP);
			cpx_status = CPXlpopt(env, lp);
		}
	}
	if(cpx_status != 0) std::cerr << "ERROR (c="<<cpx_status<<
		") while solving cplex problem." << std::endl;
	cpx_status = CPXgetx(env, lp, x.data(), 0, n-1);
	double obj;
	cpx_status = CPXgetobjval(env, lp, &obj);
	return obj;
}
double cplexPB::solveWithDual(VectorXd& x, VectorXd& xDual) {
	double obj = solve(x);
	cpx_status = CPXgetpi(env, lp, xDual.data(), 0, m-1);
	return obj;
}

void cplexPB::setTolerance(double gap) {
	cpx_status = CPXsetdblparam(env, CPX_PARAM_EPGAP, gap);
}

void cplexPB::setTimeLimit(double t) {
	cpx_status = CPXsetdblparam(env, CPX_PARAM_TILIM, t);
}

void cplexPB::setVerbose(bool b) {
	cpx_status = CPXsetintparam(env, CPX_PARAM_SCRIND, b ? CPX_ON : CPX_OFF);
}
