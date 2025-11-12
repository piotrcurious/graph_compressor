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
std::vector<float> AdvancedPolynomialFitter::composePolynomials(const float* p1_coeffs,
                                                                double p1_delta,
                                                                const float* p2_coeffs,
                                                                double p2_delta,
                                                                int degree) {
    double total_delta = p1_delta + p2_delta;
    int m = degree + 1;
    if (total_delta <= 0.0) return std::vector<float>(m, 0.0f);
    double w1 = p1_delta / total_delta;
    double w2 = 1.0 - w1;
    std::vector<double> p1d(m), p2d(m);
    for (int i = 0; i < m; ++i) { p1d[i] = double(p1_coeffs[i]); p2d[i] = double(p2_coeffs[i]); }
    VecMatrix legendre_mono = getShiftedLegendreMonomialCoeffs(degree);
    std::vector<double> legendre_coeffs(m, 0.0);
    for (int j = 0; j <= degree; ++j) {
        const std::vector<double>& L = legendre_mono[j];
        double integral_1 = 0.0;
        for (int k = 0; k <= degree; ++k) for (int l = 0; l <= j; ++l)
            integral_1 += p1d[k] * L[l] * (std::pow(w1, l+1) / (k + l + 1.0));
        std::vector<double> q(j+1, 0.0);
        for (int l = 0; l <= j; ++l) for (int mm = 0; mm <= l; ++mm)
            q[mm] += L[l] * combinations(l, mm) * std::pow(w1, l - mm) * std::pow(w2, mm);
        double integral_2 = 0.0;
        for (int k = 0; k <= degree; ++k) for (int mm = 0; mm <= j; ++mm)
            integral_2 += p2d[k] * q[mm] / (k + mm + 1.0);
        integral_2 *= w2;
        double integral_f_Lj = integral_1 + integral_2;
        double normalization = 1.0 / (2.0*j + 1.0);
        legendre_coeffs[j] = integral_f_Lj / normalization;
    }
    std::vector<double> newc(m, 0.0);
    for (int k = 0; k <= degree; ++k) for (int j = k; j <= degree; ++j) newc[k] += legendre_coeffs[j] * legendre_mono[j][k];
    std::vector<float> out(m);
    for (int i = 0; i < m; ++i) out[i] = static_cast<float>(newc[i]);
    return out;
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
