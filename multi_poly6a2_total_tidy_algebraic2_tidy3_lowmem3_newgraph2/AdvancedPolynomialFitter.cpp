// AdvancedPolynomialFitter.cpp
// Eigen-powered solver (ArduinoEigen/Eigen 0.3.2 compatible)
// Contains full-memory solver and an orthogonal-basis (Chebyshev) low-memory solver
// with double-double accumulation for numerical robustness.

#include "AdvancedPolynomialFitter.hpp"

#if __has_include(<ArduinoEigenDense.h>)
  #include <ArduinoEigenDense.h>
#else
  #include <Eigen/Dense>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <numeric>

// ----------------- Numeric helpers -----------------
static double combinations(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k > n / 2) k = n - k;
    double res = 1.0;
    for (int i = 1; i <= k; ++i) res = res * (n - i + 1) / i;
    return res;
}

using VecMatrix = std::vector<std::vector<double>>;
static VecMatrix getShiftedLegendreMonomialCoeffs(int degree) {
    VecMatrix L(degree + 1);
    L[0] = {1.0};
    if (degree > 0) L[1] = {-1.0, 2.0};
    for (int j = 1; j < degree; ++j) {
        int j1 = j + 1;
        int twoj1 = 2 * j + 1;
        L[j1].assign(j1 + 1, 0.0);
        for (int k = 0; k <= j; ++k) L[j1][k+1] = twoj1 * 2.0 * L[j][k];
        for (int k = 0; k <= j; ++k) L[j1][k] -= twoj1 * L[j][k];
        for (int k = 0; k < j; ++k) L[j1][k] -= j * L[j-1][k];
        for (int k = 0; k <= j1; ++k) L[j1][k] /= double(j1);
    }
    return L;
}

// ----------------- MSE calculation (double-x version) -----------------
double AdvancedPolynomialFitter::calculateMSED(const std::vector<float>& coeffs,
                                               const std::vector<double>& x,
                                               const std::vector<float>& y) {
    double mean = 0.0;
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        double xi_pow = 1.0;
        double pred = 0.0;
        for (size_t j = 0; j < coeffs.size(); ++j) {
            pred += double(coeffs[j]) * xi_pow;
            xi_pow *= x[i];
        }
        double e = pred - double(y[i]);
        double sq = e * e;
        mean += (sq - mean) / double(i + 1);
    }
    return mean;
}

// ----------------- Eigen conversions -----------------
static Eigen::VectorXd vecToEigen(const std::vector<float>& v) {
    Eigen::VectorXd out((Eigen::Index)v.size());
    for (Eigen::Index i = 0; i < (Eigen::Index)v.size(); ++i) out[i] = double(v[(size_t)i]);
    return out;
}

// Robust symmetric solve: LDLT with small ridge fallback, then QR fallback
static Eigen::VectorXd robustSymmetricSolve(const Eigen::MatrixXd& ATA, const Eigen::VectorXd& ATy) {
    const double ridge_init = 1e-12;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(ATA);
    if (ldlt.info() == Eigen::Success) {
        Eigen::VectorXd x = ldlt.solve(ATy);
        if (std::isfinite(x.norm())) return x;
    }
    // try small ridge scaling by trace
    double trace = ATA.trace();
    double lambda = (trace == 0.0) ? ridge_init : ridge_init * trace;
    for (int attempt = 0; attempt < 8; ++attempt) {
        Eigen::MatrixXd M = ATA;
        for (Eigen::Index i = 0; i < M.rows(); ++i) M(i, i) += lambda;
        Eigen::LDLT<Eigen::MatrixXd> ldlt2(M);
        if (ldlt2.info() == Eigen::Success) {
            Eigen::VectorXd x = ldlt2.solve(ATy);
            if (std::isfinite(x.norm())) return x;
        }
        lambda *= 100.0;
    }
    // last resort: QR on ATA
    return ATA.colPivHouseholderQr().solve(ATy);
}

// ----------------- Primary: fitPolynomialD (full-memory) -----------------
std::vector<float> AdvancedPolynomialFitter::fitPolynomialD(const std::vector<double>& x,
                                                            const std::vector<float>& y,
                                                            int degree,
                                                            OptimizationMethod method) {
    if (x.size() != y.size() || x.empty() || degree < 1) return {};
    const Eigen::Index n = (Eigen::Index)x.size();
    const int m = degree + 1;

    // Build Vandermonde (full) using raw double x
    Eigen::MatrixXd A(n, m);
    for (Eigen::Index i = 0; i < n; ++i) {
        double xi_pow = 1.0;
        for (int j = 0; j < m; ++j) {
            A(i, j) = xi_pow;
            xi_pow *= x[(size_t)i];
        }
    }

    Eigen::VectorXd yv = vecToEigen(y);
    Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(yv);

    std::vector<float> out(m);
    for (int i = 0; i < m; ++i) out[i] = static_cast<float>(coeffs[i]);

    if (method == LEVENBERG_MARQUARDT) out = levenbergMarquardtD(out, x, y, degree);
    return out;
}

// ----------------- fitPolynomialD_superpos5c (uses precomputed powers) with Kahan -----------------
std::vector<float> AdvancedPolynomialFitter::fitPolynomialD_superpos5c(const std::vector<double>& x,
                                                                       const std::vector<float>& y,
                                                                       int degree,
                                                                       OptimizationMethod /*method*/) {
    const size_t n = x.size();
    const int m = degree + 1;
    if (n == 0) return std::vector<float>();

    // We'll accumulate ATA and ATy with Kahan compensation to reduce rounding error.
    Eigen::MatrixXd ATA = Eigen::MatrixXd::Zero(m, m);
    Eigen::MatrixXd ATA_comp = Eigen::MatrixXd::Zero(m, m); // Kahan compensations
    Eigen::VectorXd ATy = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd ATy_comp = Eigen::VectorXd::Zero(m);

    std::vector<std::vector<double>> xPowers(n, std::vector<double>(m, 1.0));
    for (size_t i = 0; i < n; ++i) for (int j = 1; j < m; ++j) xPowers[i][j] = xPowers[i][j-1] * x[i];

    for (size_t i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            // Kahan add for ATy[j] += xPowers[i][j] * y[i]
            double val_aty = xPowers[i][j] * double(y[i]);
            double yv = val_aty - ATy_comp(j);
            double t = ATy(j) + yv;
            ATy_comp(j) = (t - ATy(j)) - yv;
            ATy(j) = t;

            for (int k = 0; k <= j; ++k) {
                double val = xPowers[i][j] * xPowers[i][k];
                double y2 = val - ATA_comp(j,k);
                double t2 = ATA(j,k) + y2;
                ATA_comp(j,k) = (t2 - ATA(j,k)) - y2;
                ATA(j,k) = t2;
            }
        }
    }
    // symmetrize ATA
    for (int j = 0; j < m; ++j) for (int k = j+1; k < m; ++k) ATA(j,k) = ATA(k,j);

    Eigen::VectorXd coeffs = robustSymmetricSolve(ATA, ATy);
    std::vector<float> out(m);
    for (int i = 0; i < m; ++i) out[i] = static_cast<float>(coeffs[i]);
    return out;
}

// ----------------- double-double (dd) exact-ish arithmetic helpers -----------------
struct dd { double hi; double lo; dd(): hi(0.0), lo(0.0) {} dd(double h): hi(h), lo(0.0) {} dd(double h, double l): hi(h), lo(l) {} };

static inline void two_sum(double a, double b, double &sum, double &err) {
#if defined(__GNUC__)
    sum = a + b;
    double bv = sum - a;
    err = (a - (sum - bv)) + (b - bv);
#else
    sum = a + b;
    double bv = sum - a;
    err = (a - (sum - bv)) + (b - bv);
#endif
}

static inline void two_prod(double a, double b, double &prod, double &err) {
#if defined(__GNUC__)
    prod = a * b;
    err = std::fma(a, b, -prod);
#else
    prod = a * b;
    // fallback Dekker-style split
    const double splitter = 134217729.0; // 2^27 + 1
    double a_high = (a * splitter);
    a_high = a_high - (a_high - a);
    double a_low = a - a_high;
    double b_high = (b * splitter);
    b_high = b_high - (b_high - b);
    double b_low = b - b_high;
    err = ((a_high * b_high - prod) + a_high * b_low + a_low * b_high) + a_low * b_low;
#endif
}

static inline dd dd_add(const dd &x, const dd &y) {
    double s, e1;
    two_sum(x.hi, y.hi, s, e1);
    e1 += x.lo + y.lo;
    double hi, lo;
    two_sum(s, e1, hi, lo);
    return dd(hi, lo);
}

static inline dd dd_add_double(const dd &x, double y) {
    double s, e1;
    two_sum(x.hi, y, s, e1);
    e1 += x.lo;
    double hi, lo;
    two_sum(s, e1, hi, lo);
    return dd(hi, lo);
}

static inline dd dd_mul_double(const dd &x, double y) {
    double p, err;
    two_prod(x.hi, y, p, err);
    err += x.lo * y;
    double hi, lo;
    two_sum(p, err, hi, lo);
    return dd(hi, lo);
}

static inline double dd_to_double(const dd &x) { return x.hi + x.lo; }

// ----------------- Exact-ish Chebyshev accumulation -----------------
// Build Chebyshev polynomials T_k(z) as monomial coefficients (in z variable)
static std::vector<std::vector<double>> chebyshev_monomial_coeffs(int degree) {
    int m = degree + 1;
    std::vector<std::vector<double>> T(m, std::vector<double>(m, 0.0));
    // T0 = 1
    T[0][0] = 1.0;
    if (m >= 2) {
        // T1 = z
        T[1][1] = 1.0;
    }
    for (int k = 1; k + 1 < m; ++k) {
        // T_{k+1} = 2*z*T_k - T_{k-1}
        std::vector<double> two_z_Tk(m, 0.0);
        for (int t = 0; t < m - 1; ++t) {
            two_z_Tk[t+1] = 2.0 * T[k][t];
        }
        for (int t = 0; t < m; ++t) {
            T[k+1][t] = two_z_Tk[t] - T[k-1][t];
        }
    }
    return T;
}

static void pairwise_accumulate_chebyshev_dd(const std::vector<double>& x, const std::vector<float>& y,
                                              int degree,
                                              std::vector<std::vector<dd>>& G_dd,
                                              std::vector<dd>& b_dd,
                                              double mid, double scale) {
    size_t n = x.size();
    int m = degree + 1;
    // initialize
    G_dd.assign(m, std::vector<dd>(m));
    b_dd.assign(m, dd(0.0));

    std::vector<double> Tvals(m);
    for (size_t i = 0; i < n; ++i) {
        double z = (x[i] - mid) / scale;
        // recurrence T0=1, T1=z
        if (m >= 1) Tvals[0] = 1.0;
        if (m >= 2) Tvals[1] = z;
        for (int k = 2; k < m; ++k) Tvals[k] = 2.0 * z * Tvals[k-1] - Tvals[k-2];

        // accumulate b and G (lower triangle) using dd adds
        double yi = double(y[i]);
        for (int j = 0; j < m; ++j) {
            double tj = Tvals[j];
            b_dd[j] = dd_add_double(b_dd[j], tj * yi);
            for (int k = 0; k <= j; ++k) {
                double tk = Tvals[k];
                double prod = tj * tk;
                G_dd[j][k] = dd_add_double(G_dd[j][k], prod);
            }
        }
    }
}

// ----------------- helper: convert z-monomial basis -> x-monomial basis using dd -----------------
static std::vector<double> convert_u_to_x_coeffs_dd(const std::vector<double>& c_u, double mu, double scale) {
    int m = (int)c_u.size();
    std::vector<dd> c_x_dd(m);
    for (int k = 0; k < m; ++k) {
        double cu = c_u[k];
        double invscale_pow = std::pow(scale, -double(k));
        for (int t = 0; t <= k; ++t) {
            // compute comb(k,t)
            double comb = 1.0;
            int kk = k, rr = t;
            if (!(rr == 0 || rr == kk)) {
                if (rr > kk - rr) rr = kk - rr;
                comb = 1.0;
                for (int ii = 1; ii <= rr; ++ii) comb = comb * double(kk - ii + 1) / double(ii);
            }
            double mu_pow = (k - t == 0) ? 1.0 : std::pow(-mu, double(k - t));
            double contrib = cu * comb * mu_pow * invscale_pow;
            c_x_dd[t] = dd_add_double(c_x_dd[t], contrib);
        }
    }
    std::vector<double> out(m);
    for (int i = 0; i < m; ++i) out[i] = dd_to_double(c_x_dd[i]);
    return out;
}

// ----------------- LOW-MEM: Chebyshev orthogonal-basis solver (dd accumulation) -----------------
std::vector<float> AdvancedPolynomialFitter::fitPolynomialD_lowmem(const std::vector<double>& x,
                                                                    const std::vector<float>& y,
                                                                    int degree,
                                                                    double ridge) {
    if (x.size() != y.size() || x.empty() || degree < 1) return {};
    int m = degree + 1;

    // choose mid & scale to map domain to [-1,1]
    double xmin = x[0], xmax = x[0];
    for (size_t i = 1; i < x.size(); ++i) { if (x[i] < xmin) xmin = x[i]; if (x[i] > xmax) xmax = x[i]; }
    double mid = 0.5*(xmin + xmax);
    double scale = 0.5*(xmax - xmin);
    if (scale == 0.0) scale = 1.0;

    // accumulate Gram matrix and rhs in dd
    std::vector<std::vector<dd>> G_dd;
    std::vector<dd> b_dd;
    pairwise_accumulate_chebyshev_dd(x, y, degree, G_dd, b_dd, mid, scale);

    // convert dd G and b into Eigen doubles (symmetrize)
    Eigen::MatrixXd G(m,m);
    Eigen::VectorXd bvec(m);
    for (int j = 0; j < m; ++j) {
        bvec(j) = dd_to_double(b_dd[j]);
        for (int k = 0; k <= j; ++k) {
            double val = dd_to_double(G_dd[j][k]);
            G(j,k) = val;
            G(k,j) = val;
        }
    }

    // solve G * c_cheb = b
    if (ridge > 0.0) for (int i = 0; i < m; ++i) G(i,i) += ridge;
    Eigen::VectorXd c_cheb = robustSymmetricSolve(G, bvec);

    // convert Chebyshev basis -> z-monomial basis using precomputed T_k(z) monomial coeffs
    std::vector<std::vector<double>> Tmono = chebyshev_monomial_coeffs(degree);
    // c_z[t] = sum_k c_cheb[k] * Tmono[k][t]
    std::vector<dd> c_z_dd(m);
    for (int k = 0; k < m; ++k) {
        double ck = c_cheb(k);
        for (int t = 0; t < m; ++t) {
            double coef = Tmono[k][t];
            if (coef == 0.0) continue;
            c_z_dd[t] = dd_add_double(c_z_dd[t], ck * coef);
        }
    }
    std::vector<double> c_z(m);
    for (int t = 0; t < m; ++t) c_z[t] = dd_to_double(c_z_dd[t]);

    // convert z-monomials to x-monomials: z = (x - mid)/scale
    std::vector<double> c_x = convert_u_to_x_coeffs_dd(c_z, mid, scale);

    std::vector<float> out(m);
    for (int i = 0; i < m; ++i) out[i] = static_cast<float>(c_x[i]);
    return out;
}

// ----------------- Levenberg-Marquardt for double-x -----------------
std::vector<float> AdvancedPolynomialFitter::levenbergMarquardtD(std::vector<float>& coeffs,
                                                                 const std::vector<double>& x,
                                                                 const std::vector<float>& y,
                                                                 int degree) {
    const int maxIterations = 100;
    double lambda = 1e-2;
    const double lambdaDown = 0.7;
    const double lambdaUp = 2.0;
    const double tol = 1e-12;

    const Eigen::Index n = (Eigen::Index)x.size();
    const int m = degree + 1;

    Eigen::VectorXd c(m);
    for (int i = 0; i < m; ++i) c[i] = double(coeffs[i]);
    double prevMSE = calculateMSED(coeffs, x, y);

    for (int it = 0; it < maxIterations; ++it) {
        Eigen::MatrixXd J(n, m);
        Eigen::VectorXd r(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            double xi_pow = 1.0;
            for (int j = 0; j < m; ++j) { J(i, j) = xi_pow; xi_pow *= x[(size_t)i]; }
            double pred = 0.0;
            double xp = 1.0;
            for (int j = 0; j < m; ++j) { pred += double(coeffs[j]) * xp; xp *= x[(size_t)i]; }
            r(i) = double(y[(size_t)i]) - pred;
        }
        Eigen::MatrixXd JTJ = J.transpose() * J;
        Eigen::VectorXd JTr = J.transpose() * r;
        for (int j = 0; j < m; ++j) JTJ(j, j) += lambda;
        Eigen::LDLT<Eigen::MatrixXd> ldlt(JTJ);
        if (ldlt.info() != Eigen::Success) break;
        Eigen::VectorXd delta = ldlt.solve(JTr);
        Eigen::VectorXd newc = c + delta;
        std::vector<float> candidate(m);
        for (int j = 0; j < m; ++j) candidate[j] = static_cast<float>(newc[j]);
        double newMSE = calculateMSED(candidate, x, y);
        if (!std::isfinite(newMSE)) break;
        if (newMSE < prevMSE) {
            c = newc;
            for (int j = 0; j < m; ++j) coeffs[j] = static_cast<float>(c[j]);
            prevMSE = newMSE;
            lambda *= lambdaDown;
            if (lambda < 1e-16) lambda = 1e-16;
        } else {
            lambda *= lambdaUp;
            if (lambda > 1e16) break;
        }
        if (std::abs(prevMSE - newMSE) < tol) break;
    }
    std::vector<float> out(m);
    for (int i = 0; i < m; ++i) out[i] = static_cast<float>(c[i]);
    return out;
}

// ----------------- solveLinearSystem using Eigen -----------------
std::vector<double> AdvancedPolynomialFitter::solveLinearSystem(std::vector<std::vector<double>>& A,
                                                                std::vector<double>& b) {
    Eigen::Index n = (Eigen::Index)A.size();
    if (n == 0) return {};
    Eigen::MatrixXd M(n, n);
    Eigen::VectorXd bv(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        bv(i) = b[(size_t)i];
        for (Eigen::Index j = 0; j < n; ++j) M(i, j) = A[(size_t)i][(size_t)j];
    }
    Eigen::VectorXd x = M.colPivHouseholderQr().solve(bv);
    std::vector<double> out((size_t)n);
    for (Eigen::Index i = 0; i < n; ++i) out[(size_t)i] = x(i);
    return out;
}

// ----------------- composePolynomials (unchanged) -----------------
auto qp_solve(const Eigen::MatrixXd &ATA, const Eigen::VectorXd &ATb,
    const std::vector<std::vector<double>>& Aeq_rows, const std::vector<double>& beq,
    const std::vector<std::vector<double>>& Ain_rows, const std::vector<double>& bin, int m) -> std::pair<bool, Eigen::VectorXd> {
    int neq = (int)Aeq_rows.size();
    int nin = (int)Ain_rows.size();

    // Convert Aeq and Ain to Eigen matrices
    Eigen::MatrixXd Aeq = Eigen::MatrixXd::Zero(std::max(0, neq), m);
    Eigen::VectorXd be = Eigen::VectorXd::Zero(std::max(0, neq));
    for (int i = 0; i < neq; ++i) {
        be(i) = beq[i];
        for (int j = 0; j < m; ++j) Aeq(i, j) = Aeq_rows[i][j];
    }
    Eigen::MatrixXd Ain = Eigen::MatrixXd::Zero(std::max(0, nin), m);
    Eigen::VectorXd bi = Eigen::VectorXd::Zero(std::max(0, nin));
    for (int i = 0; i < nin; ++i) {
        bi(i) = bin[i];
        for (int j = 0; j < m; ++j) Ain(i, j) = Ain_rows[i][j];
    }

    // If no inequalities, fallback to equality-constrained solver (KKT)
    if (nin == 0) {
        int K = m + neq;
        Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(K, K);
        for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) KKT(i,j) = ATA(i,j);
        for (int i = 0; i < neq; ++i) for (int j = 0; j < m; ++j) {
            KKT(j, m + i) = Aeq(i, j);
            KKT(m + i, j) = Aeq(i, j);
        }
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(K);
        for (int i = 0; i < m; ++i) rhs(i) = ATb(i);
        for (int i = 0; i < neq; ++i) rhs(m + i) = be(i);
        Eigen::VectorXd sol = KKT.colPivHouseholderQr().solve(rhs);
        if (!std::isfinite(sol.norm())) {
            return {false, Eigen::VectorXd()};
        }
        return {true, sol.head(m)};
    }

    Eigen::VectorXd c;
    {
        // initial guess via KKT for equality part
        int K = m + neq;
        Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(K, K);
        for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) KKT(i,j) = ATA(i,j);
        for (int i = 0; i < neq; ++i) for (int j = 0; j < m; ++j) {
            KKT(j, m + i) = Aeq(i, j);
            KKT(m + i, j) = Aeq(i, j);
        }
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(K);
        for (int i = 0; i < m; ++i) rhs(i) = ATb(i);
        for (int i = 0; i < neq; ++i) rhs(m + i) = be(i);
        Eigen::VectorXd sol = KKT.colPivHouseholderQr().solve(rhs);
        if (std::isfinite(sol.norm())) c = sol.head(m);
        else c = robustSymmetricSolve(ATA, ATb);
    }

    Eigen::VectorXd s = Eigen::VectorXd::Zero(nin);
    Eigen::VectorXd z = Eigen::VectorXd::Ones(nin);
    for (int i = 0; i < nin; ++i) {
        double si = bin[i] - double(Ain.row(i).dot(c));
        if (si <= kEps) si = 1.0; // keep strictly positive initial slack
        s(i) = si;
        z(i) = 1.0;
    }
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(std::max(0, neq));

    const int max_iters = 60;
    const double tol = 1e-10;
    double mu = (nin > 0) ? (s.dot(z)) / double(nin) : 1.0;
    if (!std::isfinite(mu) || mu <= kEps) mu = 1.0;

    for (int iter = 0; iter < max_iters; ++iter) {
        Eigen::VectorXd r1 = ATA * c - ATb;
        if (neq > 0) r1 += Aeq.transpose() * lambda;
        r1 += Ain.transpose() * z;
        Eigen::VectorXd r2 = Eigen::VectorXd::Zero(std::max(0, neq));
        if (neq > 0) r2 = Aeq * c - be;
        Eigen::VectorXd r3 = Ain * c + s - Eigen::Map<const Eigen::VectorXd>(bin.data(), nin);
        Eigen::VectorXd S = s;
        Eigen::VectorXd Z = z;
        double sigma = 0.1;
        Eigen::VectorXd r4 = S.cwiseProduct(Z) - Eigen::VectorXd::Constant(nin, sigma * mu);

        double resnorm = r1.norm() + r2.norm() + r3.norm() + r4.norm();
        if (resnorm < 1e-9 && mu < tol) break;

        // Protect inverses
        Eigen::VectorXd Sinv = s;
        for (int i = 0; i < nin; ++i) {
            if (Sinv(i) < 1e-12) Sinv(i) = 1e-12;
        }
        Sinv = Sinv.cwiseInverse();
        Eigen::VectorXd SinvZ = Sinv.cwiseProduct(Z);

        Eigen::MatrixXd M = ATA;
        for (int i = 0; i < nin; ++i) {
            double w = SinvZ(i);
            if (abs(w) < kEps) continue;
            Eigen::VectorXd ai = Ain.row(i);
            M.noalias() += w * (ai.transpose() * ai);
        }

        Eigen::VectorXd tmp = r4 - Z.cwiseProduct(r3);
        Eigen::VectorXd add = Eigen::VectorXd::Zero(m);
        for (int i = 0; i < nin; ++i) {
            double coef = Sinv(i) * tmp(i);
            if (abs(coef) < kEps) continue;
            add.noalias() += coef * Ain.row(i).transpose();
        }
        Eigen::VectorXd rhs_reduced = -r1 + add;
        Eigen::VectorXd rhs_eq = -r2;

        int K = m + neq;
        Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(K, K);
        for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) KKT(i,j) = M(i,j);
        for (int i = 0; i < neq; ++i) for (int j = 0; j < m; ++j) {
            KKT(j, m + i) = Aeq(i, j);
            KKT(m + i, j) = Aeq(i, j);
        }
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(K);
        for (int i = 0; i < m; ++i) rhs(i) = rhs_reduced(i);
        for (int i = 0; i < neq; ++i) rhs(m + i) = rhs_eq(i);

        Eigen::VectorXd sol = KKT.colPivHouseholderQr().solve(rhs);
        if (!std::isfinite(sol.norm())) return {false, Eigen::VectorXd()};
        Eigen::VectorXd dc = sol.head(m);
        Eigen::VectorXd dl = (neq>0) ? sol.segment(m, neq) : Eigen::VectorXd();

        Eigen::VectorXd ds = -r3 - Ain * dc;
        Eigen::VectorXd dz = Eigen::VectorXd::Zero(nin);
        for (int i = 0; i < nin; ++i) {
            dz(i) = Sinv(i) * ( -r4(i) - Z(i) * ds(i) );
        }

        double alpha_pr = 1.0, alpha_du = 1.0;
        for (int i = 0; i < nin; ++i) {
            if (ds(i) < -kEps) alpha_pr = std::min(alpha_pr, -s(i) / ds(i));
            if (dz(i) < -kEps) alpha_du = std::min(alpha_du, -z(i) / dz(i));
        }
        const double tau = 0.995;
        alpha_pr = std::min(1.0, tau * alpha_pr);
        alpha_du = std::min(1.0, tau * alpha_du);

        c += alpha_pr * dc;
        if (neq > 0) lambda += alpha_du * dl;
        s += alpha_pr * ds;
        z += alpha_du * dz;

        for (int i = 0; i < nin; ++i) {
            if (s(i) <= kEps) s(i) = 1e-12;
            if (z(i) <= kEps) z(i) = 1e-12;
        }

        mu = (s.dot(z)) / double(nin);
        if (mu < tol) break;
    } // interior iterations

    Eigen::VectorXd ineq_res = Ain * c - Eigen::Map<const Eigen::VectorXd>(bin.data(), nin);
    double max_viol = 0.0;
    for (int i = 0; i < nin; ++i) max_viol = std::max(max_viol, ineq_res(i));
    if (max_viol > kFeasTol) {
        return {false, Eigen::VectorXd()};
    }
    return {true, c};
};

std::vector<float> AdvancedPolynomialFitter::composePolynomials(const float* p1_coeffs,
                                                                double p1_delta,
                                                                const float* p2_coeffs,
                                                                double p2_delta,
                                                                int degree) {
    using std::abs;
    int m = degree + 1;
    if (m <= 0) return {};
    double total_delta = p1_delta + p2_delta;
    if (total_delta <= kEps) return std::vector<float>(m, 0.0f);

    // normalized boundary split
    double w1 = p1_delta / total_delta;
    double w2 = 1.0 - w1;

    // convert inputs to double
    std::vector<double> p1d(m, 0.0), p2d(m, 0.0);
    for (int i = 0; i < m; ++i) { p1d[i] = double(p1_coeffs[i]); p2d[i] = double(p2_coeffs[i]); }

    // small combinatorial helper (safe, iterative)
    auto comb = [](int n, int k)->double {
        if (k < 0 || k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;
        if (k > n/2) k = n - k;
        double r = 1.0;
        for (int i = 1; i <= k; ++i) r = r * double(n - i + 1) / double(i);
        return r;
    };

    // ---------- helpers for analytic ATA/ATb building (with weight alpha) ----------
    auto build_weighted_ATA_ATb = [&](double alpha, Eigen::MatrixXd &ATA_out, Eigen::VectorXd &ATb_out, std::vector<dd> &ATb_dd_out) {
        ATA_out = Eigen::MatrixXd(m, m);
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) {
                int n = r + c;
                double I0 = 1.0 / double(n + 1);     // ∫ x^n
                double I1 = 1.0 / double(n + 3);     // ∫ x^{n+2}
                double I2 = 1.0 / double(n + 2);     // ∫ x^{n+1}
                double weighted = I0 + alpha * (I1 - 2.0 * w1 * I2 + w1 * w1 * I0);
                ATA_out(r, c) = weighted;
            }
        }

        ATb_dd_out.assign(m, dd(0.0));
        // part 1: p1 on [0,w1] with internal u
        if (w1 > kEps) {
            // precompute w1^r+1 factors incrementally for r loop
            std::vector<double> w1_pows(m);
            double cur = w1;
            for (int r = 0; r < m; ++r) { w1_pows[r] = cur; cur *= w1; }

            for (int r = 0; r < m; ++r) {
                double factor = w1_pows[r]; // w1^(r+1) because w1_pows[0] = w1
                for (int k = 0; k <= degree; ++k) {
                    double I0 = 1.0 / double(k + r + 1);
                    double I2 = 1.0 / double(k + r + 3);
                    double I1 = 1.0 / double(k + r + 2);
                    double term = p1d[k] * (I0 + alpha * (w1 * w1) * (I2 - 2.0 * I1 + I0));
                    ATb_dd_out[r] = dd_add_double(ATb_dd_out[r], factor * term);
                }
            }
        }

        // part 2: p2 on [w1,1]
        if (w2 > kEps) {
            // Precompute powers of w1 and w2 up to degree m to avoid pow in inner loops
            std::vector<double> w1_pows(m);
            std::vector<double> w2_pows(m);
            w1_pows[0] = 1.0; w2_pows[0] = 1.0;
            for (int i = 1; i < m; ++i) { w1_pows[i] = w1_pows[i-1] * w1; w2_pows[i] = w2_pows[i-1] * w2; }

            for (int r = 0; r < m; ++r) {
                for (int k = 0; k <= degree; ++k) {
                    double inner = 0.0;
                    double inner_weighted = 0.0;
                    // loop over t = 0..r
                    for (int t = 0; t <= r; ++t) {
                        double b = comb(r, t);
                        double w1pow = w1_pows[r - t];
                        double w2pow = w2_pows[t];
                        double Ibase = 1.0 / double(k + t + 1);
                        double Iplus2 = 1.0 / double(k + t + 3);
                        inner += b * w1pow * w2pow * Ibase;
                        inner_weighted += b * w1pow * w2pow * Iplus2;
                    }
                    double term = p2d[k] * (inner + alpha * (w2 * w2) * inner_weighted);
                    ATb_dd_out[r] = dd_add_double(ATb_dd_out[r], term * w2);
                }
            }
        }

        ATb_out = Eigen::VectorXd(m);
        for (int r = 0; r < m; ++r) ATb_out(r) = dd_to_double(ATb_dd_out[r]);
    };

    // exact ∫ f^2 dx constant
    auto compute_F2 = [&]() -> double {
        dd acc(0.0);
        if (w1 > kEps) {
            for (int i = 0; i <= degree; ++i) for (int j = 0; j <= degree; ++j) {
                double denom = 1.0 / double(i + j + 1);
                double contrib = p1d[i] * p1d[j] * denom * w1;
                acc = dd_add_double(acc, contrib);
            }
        }
        if (w2 > kEps) {
            for (int i = 0; i <= degree; ++i) for (int j = 0; j <= degree; ++j) {
                double denom = 1.0 / double(i + j + 1);
                double contrib = p2d[i] * p2d[j] * denom * w2;
                acc = dd_add_double(acc, contrib);
            }
        }
        return dd_to_double(acc);
    };

    double F2_const = compute_F2();

    // helpers evaluate p and derivatives globally
    auto eval_poly = [](const std::vector<double>& c, double u) {
        double r = 0.0;
        for (int k = (int)c.size() - 1; k >= 0; --k) r = r * u + c[k];
        return r;
    };
    auto deriv_coeffs = [](const std::vector<double>& c) {
        int n = (int)c.size();
        std::vector<double> d(std::max(1, n - 1), 0.0);
        if (n <= 1) return d;
        for (int k = 0; k < n - 1; ++k) d[k] = double(k + 1) * c[k + 1];
        return d;
    };
    auto second_deriv_coeffs = [](const std::vector<double>& c) {
        int n = (int)c.size();
        std::vector<double> s(std::max(1, n - 2), 0.0);
        if (n <= 2) return s;
        for (int k = 2; k < n; ++k) s[k - 2] = double(k) * double(k - 1) * c[k];
        return s;
    };

    // Pre-cache derivative coefficients
    auto p1d_deriv = deriv_coeffs(p1d);
    auto p1d_second = second_deriv_coeffs(p1d);
    auto p2d_deriv = deriv_coeffs(p2d);
    auto p2d_second = second_deriv_coeffs(p2d);

    auto eval_f_global = [&](double x)->double {
        if (x <= w1 + kFuzzyBoundary) {
            double u = (abs(w1) < kEps) ? 0.0 : (x / w1);
            return eval_poly(p1d, u);
        } else {
            double u = (abs(w2) < kEps) ? 0.0 : ((x - w1) / w2);
            return eval_poly(p2d, u);
        }
    };
    auto eval_fprime_global = [&](double x)->double {
        if (x <= w1 + kFuzzyBoundary) {
            double scale = (abs(w1) < kEps) ? 1.0 : w1;
            double u = (abs(scale) < kEps) ? 0.0 : (x / scale);
            double val = eval_poly(p1d_deriv, u);
            return val / scale;
        } else {
            double scale = (abs(w2) < kEps) ? 1.0 : w2;
            double u = (abs(scale) < kEps) ? 0.0 : ((x - w1) / scale);
            double val = eval_poly(p2d_deriv, u);
            return val / scale;
        }
    };
    auto eval_fsecond_global = [&](double x)->double {
        if (x <= w1 + kFuzzyBoundary) {
            double scale = (abs(w1) < kEps) ? 1.0 : w1;
            double u = (abs(scale) < kEps) ? 0.0 : (x / scale);
            double val = eval_poly(p1d_second, u);
            return val / (scale * scale);
        } else {
            double scale = (abs(w2) < kEps) ? 1.0 : w2;
            double u = (abs(scale) < kEps) ? 0.0 : ((x - w1) / scale);
            double val = eval_poly(p2d_second, u);
            return val / (scale * scale);
        }
    };

    // find interior extrema
    auto find_real_roots_unit = [&](const std::vector<double>& poly)->std::vector<double> {
        std::vector<double> roots;
        int deg = (int)poly.size() - 1;
        while (deg > 0 && std::abs(poly[deg]) < kEps) --deg;
        if (deg <= 0) return roots;
        if (deg == 1) {
            double a = poly[1], b = poly[0];
            if (abs(a) > kEps) {
                double r = -b/a;
                if (r >= -kRootTol && r <= 1.0 + kRootTol) roots.push_back(std::min(1.0, std::max(0.0, r)));
            }
            return roots;
        }
        double lead = poly[deg];

        if (std::abs(lead) < kLeadCoeffTol) {
            DEBUG_PRINT("--- AdvancedPolynomialFitter: Skipping root finding, lead coeff too small (%g). ---\n", lead);
            return roots;
        }

        std::vector<double> bcoef(deg);
        for (int k = 0; k < deg; ++k) bcoef[k] = poly[k] / lead;
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(deg, deg);
        for (int j = 0; j < deg; ++j) C(0, j) = -bcoef[deg - 1 - j];
        for (int i = 1; i < deg; ++i) C(i, i - 1) = 1.0;
        Eigen::EigenSolver<Eigen::MatrixXd> es(C, false);
        Eigen::VectorXcd eigs = es.eigenvalues();
        for (int i = 0; i < eigs.size(); ++i) {
            std::complex<double> z = eigs[i];
            if (std::abs(z.imag()) < kImagTol) {
                double r = z.real();
                if (r >= -kRootTol && r <= 1.0 + kRootTol) roots.push_back(std::min(1.0, std::max(0.0, r)));
            }
        }
        std::sort(roots.begin(), roots.end());
        roots.erase(std::unique(roots.begin(), roots.end(), [](double a,double b){ return std::abs(a-b) < kRootTol; }), roots.end());
        return roots;
    };

    std::vector<double> extrema_x;
    // p1 interior extrema + endpoints
    {
        auto r1 = find_real_roots_unit(p1d_deriv);
        for (double u : r1) {
            double x = w1 * u;
            if (x >= 0.0 && x <= 1.0) extrema_x.push_back(x);
        }
        extrema_x.push_back(0.0);
        extrema_x.push_back(w1);
    }
    // p2 interior extrema + endpoints
    if (w2 > kEps) {
        auto r2 = find_real_roots_unit(p2d_deriv);
        for (double u : r2) {
            double x = w1 + w2 * u;
            if (x >= 0.0 && x <= 1.0) extrema_x.push_back(x);
        }
        extrema_x.push_back(w1);
        extrema_x.push_back(1.0);
    }
    std::sort(extrema_x.begin(), extrema_x.end());
    extrema_x.erase(std::unique(extrema_x.begin(), extrema_x.end(), [](double a,double b){ return std::abs(a-b) < kRootTol; }), extrema_x.end());

    // scoring & trimming candidates
    struct Ext { double x; double score; };
    std::vector<Ext> scored;
    const double hu = 1e-4;
    for (double x : extrema_x) {
        double fpp = std::abs(eval_fsecond_global(x));
        bool from_p1 = (x <= w1 + kFuzzyBoundary);
        double prom = 0.0;
        if (from_p1) {
            double scale = (abs(w1) < kEps) ? 1.0 : w1;
            double u = (abs(scale) < kEps) ? 0.0 : (x / scale);
            double umin = std::max(0.0, u - hu), umax = std::min(1.0, u + hu);
            prom = std::min(std::abs(eval_poly(p1d, u) - eval_poly(p1d, umin)), std::abs(eval_poly(p1d, u) - eval_poly(p1d, umax)));
        } else {
            double scale = (abs(w2) < kEps) ? 1.0 : w2;
            double u = (abs(scale) < kEps) ? 0.0 : ((x - w1) / scale);
            double umin = std::max(0.0, u - hu), umax = std::min(1.0, u + hu);
            prom = std::min(std::abs(eval_poly(p2d, u) - eval_poly(p2d, umin)), std::abs(eval_poly(p2d, u) - eval_poly(p2d, umax)));
        }
        double score = fpp + prom;
        scored.push_back({x, score});
    }
    std::sort(scored.begin(), scored.end(), [](const Ext &a, const Ext &b){ return a.score > b.score; });

    // candidate generation
    std::vector<double> alphas;
    alphas.push_back(0.0);
    double base = 1.0;
    for (int i = -2; i <= 2; ++i) alphas.push_back(base * std::pow(10.0, i));
    std::sort(alphas.begin(), alphas.end());
    alphas.erase(std::unique(alphas.begin(), alphas.end()), alphas.end());

    int max_keep = std::max(0, m/2);
    std::vector<int> keep_counts;
    for (int k = 0; k <= max_keep; ++k) keep_counts.push_back(k);

    // endpoints preservation
    double f0 = eval_f_global(0.0);
    double f1 = eval_f_global(1.0);
    double fprime0 = eval_fprime_global(0.0);
    double fprime1 = eval_fprime_global(1.0);
    bool preserve_d0 = (std::abs(fprime0) < kRootTol);
    bool preserve_d1 = (std::abs(fprime1) < kRootTol);

    // boundary blending targets
    double left_val = (w1 > kEps) ? eval_poly(p1d, 1.0) : eval_f_global(w1);
    double right_val = (w2 > kEps) ? eval_poly(p2d, 0.0) : eval_f_global(w1);
    double boundary_val_target = w1 * left_val + w2 * right_val;
    double left_d = 0.0, right_d = 0.0;
    if (w1 > kEps) { left_d = eval_poly(p1d_deriv, 1.0) / w1; } else left_d = eval_fprime_global(w1);
    if (w2 > kEps) { right_d = eval_poly(p2d_deriv, 0.0) / w2; } else right_d = eval_fprime_global(w1);
    double boundary_d_target = w1 * left_d + w2 * right_d;

    auto build_value_row = [&](double x) {
        std::vector<double> row(m);
        double xp = 1.0;
        for (int k = 0; k < m; ++k) { row[k] = xp; xp *= x; }
        return row;
    };
    auto build_deriv_row = [&](double x) {
        std::vector<double> row(m, 0.0);
        double xpow = 1.0;
        for (int k = 1; k < m; ++k) { row[k] = double(k) * xpow; xpow *= x; }
        return row;
    };
    auto build_second_deriv_row = [&](double x) {
        std::vector<double> row(m, 0.0);
        // efficient iterative powers: x^(k-2) accumulation
        if (m >= 3) {
            double xp = 1.0; // x^0 for k=2
            for (int k = 2; k < m; ++k) {
                row[k] = double(k) * double(k - 1) * xp;
                xp *= x;
            }
        }
        return row;
    };

    // ---------- candidate search and usage of qp_solve ----------
    double best_err = std::numeric_limits<double>::infinity();
    std::vector<float> best_coeffs(m, 0.0f);

    // prepare interior_extrema
    std::vector<double> interior_extrema;
    for (auto &e : scored) {
        if (e.x <= kFuzzyBoundary || e.x >= 1.0 - kFuzzyBoundary) continue;
        if (std::abs(e.x - w1) < kFuzzyBoundary) continue;
        interior_extrema.push_back(e.x);
    }

    // boundary & endpoints
    std::vector<double> endpoints = {0.0, 1.0};
    std::vector<double> endpoint_values = {f0, f1};

    for (double alpha : alphas) {
        Eigen::MatrixXd ATA_alpha;
        Eigen::VectorXd ATb_alpha;
        std::vector<dd> ATb_dd_tmp;
        build_weighted_ATA_ATb(alpha, ATA_alpha, ATb_alpha, ATb_dd_tmp);

        // small ridge
        double trace = ATA_alpha.trace();
        double ridge = 1e-14 * std::max(1.0, std::abs(trace));
        for (int i = 0; i < m; ++i) ATA_alpha(i,i) += ridge;

        for (int keepCount : keep_counts) {
            // build equality constraints
            std::vector<std::vector<double>> C_eq;
            std::vector<double> d_eq;
            // endpoints value
            for (int ei = 0; ei < 2; ++ei) { C_eq.push_back(build_value_row(endpoints[ei])); d_eq.push_back(endpoint_values[ei]); }
            // endpoint derivative if stationary
            if (preserve_d0) { C_eq.push_back(build_deriv_row(0.0)); d_eq.push_back(0.0); }
            if (preserve_d1) { C_eq.push_back(build_deriv_row(1.0)); d_eq.push_back(0.0); }
            // boundary C0/C1 targets
            C_eq.push_back(build_value_row(w1)); d_eq.push_back(boundary_val_target);
            C_eq.push_back(build_deriv_row(w1)); d_eq.push_back(boundary_d_target);

            // add kept interior extrema (value + derivative)
            for (int k = 0; k < keepCount && k < (int)interior_extrema.size(); ++k) {
                double x = interior_extrema[k];
                C_eq.push_back(build_value_row(x)); d_eq.push_back(eval_f_global(x));
                if (m >= 2) { C_eq.push_back(build_deriv_row(x)); d_eq.push_back(0.0); }
            }

            if ((int)C_eq.size() >= m) {
                continue;
            }

            // build inequalities
            std::vector<std::vector<double>> Ain_rows;
            std::vector<double> bin_rows;
            // boundary second derivative sign
            double b_fpp = eval_fsecond_global(w1);
            {
                std::vector<double> row = build_second_deriv_row(w1);
                if (b_fpp >= 0.0) {
                    std::vector<double> ar = row; for (double &v : ar) v = -v;
                    Ain_rows.push_back(ar); bin_rows.push_back(0.0);
                } else {
                    Ain_rows.push_back(row); bin_rows.push_back(0.0);
                }
            }
            // interior extrema second derivative signs
            for (int k = 0; k < keepCount && k < (int)interior_extrema.size(); ++k) {
                double x = interior_extrema[k];
                double orig_fpp = eval_fsecond_global(x);
                std::vector<double> row = build_second_deriv_row(x);
                if (orig_fpp >= 0.0) {
                    std::vector<double> ar = row; for (double &v : ar) v = -v;
                    Ain_rows.push_back(ar); bin_rows.push_back(0.0);
                } else {
                    Ain_rows.push_back(row); bin_rows.push_back(0.0);
                }
            }

            // call QP solver
            auto sol = qp_solve(ATA_alpha, ATb_alpha, C_eq, d_eq, Ain_rows, bin_rows, m);
            if (!sol.first) continue;
            Eigen::VectorXd c = sol.second;

            // compute exact error
            double err = double(c.transpose() * ATA_alpha * c) - 2.0 * double(ATb_alpha.dot(c)) + F2_const;
            if (err < best_err) {
                best_err = err;
                for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(c(i));
                // small early-exit: if we have an almost-zero residual relative to F2_const, stop
                if (best_err <= 1e-14 * std::max(1.0, std::abs(F2_const))) {
                    break;
                }
            }
        } // keepCount
        // break outer alpha loop early if we already found a very small error
        if (best_err <= 1e-14 * std::max(1.0, std::abs(F2_const))) break;
    } // alpha

    // --- ITERATIVE FALLBACK LOGIC ---
    if (std::isfinite(best_err) && best_err < std::numeric_limits<double>::infinity()) {
        return best_coeffs;
    }

    DEBUG_PRINT("--- AdvancedPolynomialFitter: Main QP search failed. Entering fallback mode. ---\n");

    // Get the base L2 objective (alpha=0)
    Eigen::MatrixXd ATA0; Eigen::VectorXd ATb0; std::vector<dd> tmp_dd;
    build_weighted_ATA_ATb(0.0, ATA0, ATb0, tmp_dd);

    // Add ridge
    double trace0 = ATA0.trace();
    double ridge0 = 1e-14 * std::max(1.0, std::abs(trace0));
    for (int i = 0; i < m; ++i) ATA0(i,i) += ridge0;

    // --- Fallback Level 0: C0 Endpoints + C0 Boundary ---
    DEBUG_PRINT("--- Fallback Level 0: Trying C0 Endpoints + C0 Boundary ---\n");
    std::vector<std::vector<double>> C_eq_fb0; std::vector<double> d_eq_fb0;
    C_eq_fb0.push_back(build_value_row(0.0)); d_eq_fb0.push_back(f0);
    C_eq_fb0.push_back(build_value_row(1.0)); d_eq_fb0.push_back(f1);
    C_eq_fb0.push_back(build_value_row(w1)); d_eq_fb0.push_back(boundary_val_target);

    if ((int)C_eq_fb0.size() < m) {
        auto sol0 = qp_solve(ATA0, ATb0, C_eq_fb0, d_eq_fb0, {}, {}, m); // No inequalities
        if (sol0.first) {
            DEBUG_PRINT("--- Fallback Level 0 Succeeded. ---\n");
            for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(sol0.second(i));
            return best_coeffs; // Solution found!
        }
    }
    DEBUG_PRINT("--- Fallback Level 0 Failed. ---\n");

    // --- Fallback Level 1: C0 Endpoints only ---
    DEBUG_PRINT("--- Fallback Level 1: Trying C0 Endpoints only ---\n");
    std::vector<std::vector<double>> C_eq_fb1; std::vector<double> d_eq_fb1;
    C_eq_fb1.push_back(build_value_row(0.0)); d_eq_fb1.push_back(f0);
    C_eq_fb1.push_back(build_value_row(1.0)); d_eq_fb1.push_back(f1);

    if ((int)C_eq_fb1.size() < m) {
         auto sol1 = qp_solve(ATA0, ATb0, C_eq_fb1, d_eq_fb1, {}, {}, m);
         if (sol1.first) {
            DEBUG_PRINT("--- Fallback Level 1 Succeeded. ---\n");
            for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(sol1.second(i));
            return best_coeffs; // Solution found!
         }
    }
    DEBUG_PRINT("--- Fallback Level 1 Failed. ---\n");

    // --- Final Fallback: Unconstrained Least Squares ---
    DEBUG_PRINT("--- All constrained fallbacks failed. Computing unconstrained L2 fit. ---\n");
    Eigen::VectorXd c = robustSymmetricSolve(ATA0, ATb0);
    if (std::isfinite(c.norm())) {
         for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(c(i));
         return best_coeffs; // Return unconstrained solution
    }

    // --- Absolute Failure ---
    DEBUG_PRINT("--- CRITICAL: All fallbacks failed, even unconstrained solve. Returning empty. ---\n");
    return {};
}

// ----------------- NormalizeAndFitPolynomial — calls fitPolynomialD (caller handles normalization) -----------------
std::vector<float> AdvancedPolynomialFitter::NormalizeAndFitPolynomial(const std::vector<float>& x,
                                                                       const std::vector<float>& y,
                                                                       int degree,
                                                                       OptimizationMethod method) {
    std::vector<double> xd(x.size());
    for (size_t i = 0; i < x.size(); ++i) xd[i] = double(x[i]);
    return fitPolynomialD(xd, y, degree, method);
}

// ----------------- Segmented polynomials (double-x wrapper) -----------------
std::vector<float> AdvancedPolynomialFitter::fitSegmentedPolynomials(const std::vector<float>& x,
                                                                     const std::vector<float>& y,
                                                                     int degree,
                                                                     int segments) {
    std::vector<float> result;
    if (segments <= 0) return result;
    size_t N = x.size();
    size_t segmentSize = std::max((size_t)1, N / (size_t)segments);
    for (int s = 0; s < segments; ++s) {
        size_t startIdx = s * segmentSize;
        size_t endIdx = (s == segments - 1) ? N : std::min(N, (size_t)((s + 1) * segmentSize));
        if (startIdx >= endIdx) break;
        std::vector<double> xs_double(endIdx - startIdx);
        std::vector<float> ys(y.begin() + startIdx, y.begin() + endIdx);
        for (size_t i = 0; i < xs_double.size(); ++i) xs_double[i] = double(x[startIdx + i]);
        std::vector<float> coeffs = fitPolynomialD(xs_double, ys, degree);
        result.insert(result.end(), coeffs.begin(), coeffs.end());
    }
    return result;
}

// End of AdvancedPolynomialFitter.cpp
