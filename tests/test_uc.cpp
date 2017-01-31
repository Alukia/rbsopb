#include "../src/rrbsopb.h"
#include <cstdlib>

/****************************************
 * Example of a basic unit commitment   *
 * problem in which we have N=9 thermal *
 * units and we want to schedule the    *
 * production for the next T=24 hours   *
 * knowing a given average load between *
 * μ-k and μ+k.                         *
 ****************************************
 * Each unit Uᵢ (i = 0,…,8) use T       *
 * variables xᵢₜ which correspond to    *
 * the production of the unit at time ₜ *
 ****************************************
 * Functions fᵢ :                       *
 * the global cost for the schedule xᵢₜ *
 * of unit Uᵢ is given using a simple   *
 * linear formula                       *
 *                                      *
 *    fᵢ(xᵢₜ) = bᵢᵀxᵢ                   *
 *                                      *
 ****************************************
 * Domain Xᵢ :                          *
 * for each unit the production is      *
 * bounded from 0 to Mᵢ, so             *
 *                                      *
 *        0 ≤ xᵢₜ ≤ Mᵢ                  *
 *                                      *
 * and between two consecutive time     *
 * steps ₜ, ₜ₊₁ the production might be *
 * modified to at most Kᵢ               *
 *                                      *
 *        |xᵢₜ-xᵢₜ₊₁| ≤ Kᵢ,  t > 1      *
 *                                      *
 * lastly, each unit start with a       *
 * given production xᵢ₀                 *
 *                                      *
 *        |xᵢ₀-xᵢ₁| ≤ Kᵢ                *
 *                                      *
 * see tests/uc/themal_i.lp to see the  *
 * actual linear problem solved.        *
 ****************************************
 * Oracle Ψ :                           *
 * we decide to modelize the            *
 * uncertainty set D by a cubic set     *
 * around the average load μ and a      *
 * width of k.                          *
 *                                      *
 * for a given demand d ∈ D the cost φ  *
 * penalize underproduction linearly by *
 * a factor c₁ ≤ 0 and surproduction    *
 * by a factor c₂ ≥ 0 using the         *
 * following formula                    *
 *                                      *
 *     φ(d,y) = max(c₁(y-d),c₂(y-d))    *
 *                                      *
 * the Ψ function is then               *
 *                                      *
 *  Ψ(x) = ∑ₜ sup(dₜ ∈ Dₜ) φ(dₜ,(Ax)ₜ)  *
 *                                      *
 * with (Ax)ₜ = ∑ᵢ xᵢₜ the production   *
 * at time step t.                      *
 *                                      *
 * We can reformulate Ψ explicitly      *
 *                                      *
 *        Ψ(x) = ∑ₜ max(v₁ₜ, v₂ₜ)       *
 * with                                 *
 * v₁ₜ = c₁((Ax)ₜ-(μ+k))                *
 * v₂ₜ = c₂((Ax)ₜ-(μ-k)).               *
 *                                      *
 * and we have a subgradient            *
 *                                      *
 *       (gᵢₜ)ᵢₜ ∈ ∂Ψ(x)                *
 * with                                 *
 * gᵢₜ = c₁ if v₁ₜ > v₂ₜ, c₂ otherwise. *
 ****************************************/


//////////////////
// Functions fi //
//////////////////

class unit {
private:
	int n; // number of variables
	cplexPB pb;
	VectorXd c;
public:
	unit(std::string file) : pb(file) {
		n = pb.getn();
		pb.getLinearObjective(c);
	}
	unit(int T, double Mi, double Ki, double xi0, double cost)
		: n(T), pb(T, 0., Mi)
	{
		VectorXd x0c(T); x0c << 1., VectorXd::Zero(T-1);
		pb.addConstraint(x0c, xi0+Ki, 'L');
		pb.addConstraint(x0c, xi0-Ki, 'G');
		for(int i = 1; i < T; ++i) {
			VectorXd xic = VectorXd::Zero(T);
			xic(i-1) = 1.; xic(i) = -1.;
			pb.addConstraint(xic, Ki, 'L');
			pb.addConstraint(xic, -Ki, 'G');
		}
		c = VectorXd::Constant(T, cost);
		pb.setLinearObjective(c);

	}
	~unit() {}

	double costForSchedule(VectorXd& x) { return x.dot(c); }

	double solve(VectorXd& x, VectorXd& b, VectorXd& A) {
		//std::cout << b.transpose() << "\n\n";
		VectorXd c_tmp = c + b;
		pb.setLinearObjective(c_tmp);
		pb.setQuadraticObjective(A);
		return pb.solve(x);
	}
};

unit* units;
int nbUnits;
double f(int i, VectorXd& x, VectorXd& b, VectorXd& A) {
	return units[i].solve(x, b, A);
}


////////////////
// Oracle psi //
////////////////

int T = 24;
double k = 100.0;
double mu[] = {
	2288.4, 2590.1, 3002.8, 3165.6, 3313.9, 3454.0, 3314.7, 3073.9,
	2518.0, 2034.4, 1610.9, 1010.0, 1613.6, 2034.7, 2515.4, 3066.5,
	3300.5, 3451.8, 3300.0, 3066.2, 2521.4, 2047.9, 1627.2, 1029.1
};
// simple two lines cost
double c1 = -150., c2 = 13.;
double psi(VectorXd& x, VectorXd& g) {
	// y : ℝⁿᵀ → ℝᵀ
	// y = Ax
	VectorXd y = VectorXd::Zero(T);
	for(int i = 0; i < nbUnits; ++i)
		y += x.segment(i*T,T);

	double sum = 0.;
	VectorXd v(T);
	for(int t = 0; t < T; ++t) {
		double v1 = c1 * (y(t) - (mu[t]+k)),
		       v2 = c2 * (y(t) - (mu[t]-k));
		if(v1 > v2) {
			v(t) = c1;
			sum += v1;
		} else {
			v(t) = c2;
			sum += v2;
		}
	}

	// g = Aᵀv
	g = VectorXd::Zero(nbUnits*T);
	for(int i = 0; i < nbUnits; ++i)
	for(int t = 0; t < T; ++t) {
		g(i*T+t) = v(t);
	}

	return sum;
}

//////////////////
// Main program //
//////////////////

int main(int argc, char const *argv[])
{
	bool ws = true, reg = false;
	prType primal = PSEUDO_SCHEDULE;
	for(int i = 1; i < argc; ++i) {
		std::string str = argv[i];
		if(str == "-h") {
			std::cout <<
			"Example of a basic unit-commitment problem\n"
			" Primal recovery\n"
			"  -ps  pseudo-schedule [default]\n"
			"  -bi  best-iterate\n"
			//"  -dw  dantzig-wolfe-like\n"
			" Warm-start\n"
			"  -ws  on [default]\n"
			"  -nws off\n"
			" Regularized algorithm\n"
			"  -r   yes [default]\n"
			"  -nr  no"
			<< std::endl;
			exit(0);
		}
		//else if(str == "-dw")  primal = DANTZIG_WOLFE;
		else if(str == "-bi")  primal = BEST_ITERATE;
		else if(str == "-ps")  primal = PSEUDO_SCHEDULE;
		else if(str == "-ws")  ws = true;
		else if(str == "-nws") ws = false;
		else if(str == "-nr")  reg = false;
		else if(str == "-r")   reg = true;
	}

	//initialization

	unit U[] = { // create units
		unit(T, 900., 200., 700., 20.),
		unit(T, 900., 200., 700., 40.),
		unit(T, 900., 200., 700., 40.),
		unit(T, 300., 60., 150., 30.),
		unit(T, 300., 60., 150., 30.),
		unit(T, 200., 40., 0., 20.),
		unit(T, 200., 40., 0., 20.),
		unit(T, 200., 40., 0., 20.),
		unit(T, 100., 20., 0., 10.),
	};
	nbUnits = sizeof(U)/sizeof(unit);
	units = U;
	VectorXi ni = VectorXi::Constant(nbUnits, T);

	// solving

	rbsopb* pb = reg ? new rrbsopb(ni, f, psi) : new rbsopb(ni, f, psi);
	pb->setPrimalRecovery(primal)->setWarmStart(ws)->setVerbosity(2);

	VectorXd sol(nbUnits*T);
	pb->solve(sol);

	// display results

	std::cout << "\nFinal production schedule variables :" << std::endl;
	for(int i = 0; i < nbUnits; ++i) {
		std::cout << "unit " << i << " : ";
		for(int t = 0; t < T; ++t) {
			std::cout << sol(i*T+t) << (t == T-1 ? "\n" : " ");
		}
	}

	VectorXd prod = VectorXd::Zero(T);
	for(int i = 0; i < nbUnits; ++i) {
		prod += sol.segment(i*T, T);
	}

	std::cout << "\nProduction for t = 1..T :" << std::endl;
	for(int t = 0; t < T; ++t)
		std::cout << prod(t) << (t == T-1 ? "\n" : " ");

	std::cout << "\nDistance to average load for t = 1..T :" << std::endl;
	for(int t = 0; t < T; ++t)
		std::cout << prod(t)-mu[t] << (t == T-1 ? "\n" : " ");

	double sumfi = 0.;
	for(int i = 0; i < nbUnits; ++i) {
		VectorXd xi = sol.segment(i*T,T);
		sumfi += units[i].costForSchedule(xi);
	}
	std::cout << "\nf(x) = " << sumfi << std::endl;
	VectorXd g(nbUnits*T);
	double psival = psi(sol, g);
	std::cout << "Ψ(x) = " << psival << std::endl;
	std::cout << "f+Ψ  = " << sumfi+psival << std::endl;

	pb->writeLogs("log_ps_"
		+ std::string(ws  ? "ws" : "nws") + "_"
		+ std::string(reg ? "r"  : "nr" ) +
	".txt");

	return 0;
}
