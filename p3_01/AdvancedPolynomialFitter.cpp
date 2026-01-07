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

// Requires: #include <Eigen/Dense>, <vector>, <cmath>, <algorithm>, <numeric>, <limits>
std::vector<float> AdvancedPolynomialFitter::fitPolynomialD_superpos5c(
    const std::vector<double>& x,
    const std::vector<float>& y,
    int degree,
    OptimizationMethod /*method*/)
{
    const size_t n = x.size();
    const int m = degree + 1;
    if (n == 0) return {};

    // --- 1) center & scale x to z = (x - mu) / scale (use original points for mu/scale) ---
    double mu = 0.0;
    for (size_t i = 0; i < n; ++i) mu += x[i];
    mu /= static_cast<double>(n);

    double maxabs = 0.0;
    for (size_t i = 0; i < n; ++i) maxabs = std::max(maxabs, std::abs(x[i] - mu));
    double scale = (maxabs == 0.0) ? 1.0 : maxabs;

    // --- 2) detect large gaps by sorting x,y pairs ---
    std::vector<std::pair<double, double>> pts;
    pts.reserve(n);
    for (size_t i = 0; i < n; ++i) pts.emplace_back(x[i], static_cast<double>(y[i]));
    std::sort(pts.begin(), pts.end(), [](auto &a, auto &b){ return a.first < b.first; });

    // compute gaps
    std::vector<double> gaps;
    gaps.reserve((n > 1) ? n - 1 : 0);
    for (size_t i = 1; i < pts.size(); ++i) gaps.push_back(pts[i].first - pts[i-1].first);

    // robust median gap (fallback to small epsilon if degenerate)
    double median_gap = 0.0;
    if (!gaps.empty()) {
        std::vector<double> gaps_copy = gaps;
        std::nth_element(gaps_copy.begin(), gaps_copy.begin() + gaps_copy.size()/2, gaps_copy.end());
        median_gap = gaps_copy[gaps_copy.size()/2];
        if (gaps_copy.size() % 2 == 0) {
            // average middle two for even count
            auto max_left = *std::max_element(gaps_copy.begin(), gaps_copy.begin() + gaps_copy.size()/2);
            median_gap = 0.5 * (median_gap + max_left);
        }
    }
    if (median_gap <= 0.0) {
        // fallback: small fraction of total range or 1e-6
        double xrng = (pts.empty() ? 1.0 : (pts.back().first - pts.front().first));
        median_gap = (xrng > 0.0) ? (xrng / static_cast<double>(std::max<size_t>(1, n-1))) : 1e-6;
    }

    // --- 3) build augmented dataset with synthetic points inside large gaps ---
    std::vector<double> x_aug;
    std::vector<double> y_aug;
    std::vector<double> w_aug; // weights: 1.0 for real, <1 for synthetic

    const double synth_weight = 0.20; // tune: influence of synthetic points (0..1)
    const int max_inserts_per_gap = 200; // safety cap

    for (size_t i = 0; i < pts.size(); ++i) {
        // push the real point
        x_aug.push_back(pts[i].first);
        y_aug.push_back(pts[i].second);
        w_aug.push_back(1.0);

        if (i + 1 < pts.size()) {
            double x0 = pts[i].first;
            double x1 = pts[i+1].first;
            double gap = x1 - x0;

            // identify if the gap is "large" relative to median_gap
            // threshold: gap >= factor * median_gap
            const double gap_factor = 2.5; // tuneable: how much bigger than median to consider a "large" gap
            if (gap > gap_factor * median_gap) {
                // decide how many inserted points -> roughly proportional to gap/median_gap
                int inserts = static_cast<int>(std::floor(gap / median_gap)) - 1;
                if (inserts < 1) inserts = 1;
                inserts = std::min(inserts, max_inserts_per_gap);

                // linear interpolate y across gap (simple and stable)
                double y0 = pts[i].second;
                double y1 = pts[i+1].second;
                for (int k = 1; k <= inserts; ++k) {
                    double t = static_cast<double>(k) / static_cast<double>(inserts + 1);
                    double xi = x0 + t * (x1 - x0);
                    double yi = y0 + t * (y1 - y0);
                    x_aug.push_back(xi);
                    y_aug.push_back(yi);
                    w_aug.push_back(synth_weight);
                }
            }
        }
    }

    const size_t n_aug = x_aug.size();
    if (n_aug == 0) return {};

    // --- 4) build Vandermonde A (n_aug x m) in z basis and weighted Y ---
    Eigen::MatrixXd A(static_cast<int>(n_aug), m);
    Eigen::VectorXd Y(static_cast<int>(n_aug));
    Eigen::VectorXd sqrtW(static_cast<int>(n_aug));
    for (size_t i = 0; i < n_aug; ++i) {
        double z = (x_aug[i] - mu) / scale;
        double zp = 1.0;
        for (int j = 0; j < m; ++j) {
            A(static_cast<int>(i), j) = zp;
            zp *= z;
        }
        Y(static_cast<int>(i)) = static_cast<double>(y_aug[i]);
        double wi = w_aug[i];
        if (!(wi > 0.0)) wi = 1.0;
        sqrtW(static_cast<int>(i)) = std::sqrt(wi);
    }

    // Apply sqrtW row-scaling to implement weighted least squares: Aw = diag(sqrtW) * A, Yw = diag(sqrtW)*Y
    Eigen::MatrixXd Aw(static_cast<int>(n_aug), m);
    Eigen::VectorXd Yw(static_cast<int>(n_aug));
    for (int i = 0; i < static_cast<int>(n_aug); ++i) {
        Aw.row(i) = A.row(i) * sqrtW(i);
        Yw(i) = Y(i) * sqrtW(i);
    }

    // If underdetermined use SVD minimum-norm on weighted system
    Eigen::VectorXd a_z(m);
    if (n_aug < static_cast<size_t>(m)) {
        a_z = Aw.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Yw);
    } else {
        // Prefer QR for speed & stability; fallback to SVD if rank-deficient
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(Aw);
        qr.setThreshold(1e-12);
        if (qr.rank() == m) {
            a_z = qr.solve(Yw);
        } else {
            a_z = Aw.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Yw);
        }
    }

    // --- 5) convert 'a_z' (coeffs in z^j) back to polynomial coeffs in x^k ---
    auto binomial = [](int n_, int k_) -> double {
        if (k_ < 0 || k_ > n_) return 0.0;
        int kk = std::min(k_, n_ - k_);
        double r = 1.0;
        for (int t=1; t<=kk; ++t) {
            r *= static_cast<double>(n_ - (kk - t));
            r /= static_cast<double>(t);
        }
        return r;
    };

    std::vector<double> minusMuPow(m, 1.0), invScalePow(m, 1.0);
    for (int j = 1; j < m; ++j) {
        minusMuPow[j] = minusMuPow[j-1] * (-mu);
        invScalePow[j] = invScalePow[j-1] / scale;
    }

    std::vector<float> out(m);
    for (int k = 0; k < m; ++k) {
        double accum = 0.0;
        for (int j = k; j < m; ++j) {
            double C = binomial(j, k);
            double term = static_cast<double>(a_z(j)) * C * minusMuPow[j - k] * invScalePow[j];
            accum += term;
        }
        out[k] = static_cast<float>(accum);
    }

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
#include <numeric> // for iota

std::vector<float> AdvancedPolynomialFitter::fitPolynomialD_lowmem(const std::vector<double>& x,
                                                                    const std::vector<float>& y,
                                                                    int degree,
                                                                    double ridge) {
    if (x.size() != y.size() || x.empty() || degree < 1) return {};
    int m = degree + 1;
    const size_t n = x.size();

    // choose mid & scale to map domain to [-1,1]
    double xmin = x[0], xmax = x[0];
    for (size_t i = 1; i < n; ++i) { if (x[i] < xmin) xmin = x[i]; if (x[i] > xmax) xmax = x[i]; }
    double mid = 0.5*(xmin + xmax);
    double scale = 0.5*(xmax - xmin);
    if (scale == 0.0) scale = 1.0;

    // accumulate Gram matrix and rhs in dd for the real points
    std::vector<std::vector<dd>> G_dd;
    std::vector<dd> b_dd;
    pairwise_accumulate_chebyshev_dd(x, y, degree, G_dd, b_dd, mid, scale);

    // ---------- detect large gaps (use only index array to keep memory low) ----------
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return x[a] < x[b]; });

    // compute gaps
    std::vector<double> gaps;
    gaps.reserve((n > 1) ? n - 1 : 0);
    for (size_t i = 1; i < n; ++i) gaps.push_back(x[idx[i]] - x[idx[i-1]]);

    // robust median gap
    double median_gap = 0.0;
    if (!gaps.empty()) {
        std::vector<double> gaps_copy = gaps; // temporary O(n) small vector
        size_t midpos = gaps_copy.size()/2;
        std::nth_element(gaps_copy.begin(), gaps_copy.begin() + midpos, gaps_copy.end());
        median_gap = gaps_copy[midpos];
        if (gaps_copy.size() % 2 == 0) {
            double left_max = *std::max_element(gaps_copy.begin(), gaps_copy.begin() + midpos);
            median_gap = 0.5 * (median_gap + left_max);
        }
    }
    if (median_gap <= 0.0) {
        double xrng = (xmax - xmin);
        median_gap = (xrng > 0.0) ? (xrng / static_cast<double>(std::max<size_t>(1, n-1))) : 1e-12;
    }

    // ---------- synthetic-points policy (tunable) ----------
    const double gap_factor = 2.5;         // gap > gap_factor * median_gap considered "large"
    const double synth_weight = 0.20;      // weight assigned to synthetic points (0..1)
    const int max_inserts_per_gap = 200;   // cap to avoid explosion (keeps memory/time bounded)

    // ---------- accumulate synthetic contributions directly into G_dd and b_dd ----------
    if (n > 1 && median_gap > 0.0) {
        for (size_t s = 0; s + 1 < n; ++s) {
            size_t i_idx = idx[s];
            size_t j_idx = idx[s+1];
            double x0 = x[i_idx];
            double x1 = x[j_idx];
            double gap = x1 - x0;
            if (!(gap > gap_factor * median_gap)) continue;

            // how many synthetic points to insert (proportional to gap)
            int inserts = static_cast<int>(std::floor(gap / median_gap)) - 1;
            if (inserts < 1) inserts = 1;
            inserts = std::min(inserts, max_inserts_per_gap);

            double y0 = static_cast<double>(y[i_idx]);
            double y1 = static_cast<double>(y[j_idx]);

            for (int kIns = 1; kIns <= inserts; ++kIns) {
                double t = static_cast<double>(kIns) / static_cast<double>(inserts + 1);
                double xi = x0 + t * (x1 - x0);
                double yi = y0 + t * (y1 - y0); // linear interpolation - stable & monotone

                double z = (xi - mid) / scale;  // Chebyshev argument (should be in [-1,1] normally)

                // compute Chebyshev basis T_0..T_degree at z (recurrence)
                // allocate small vector on stack
                std::vector<double> T(m);
                T[0] = 1.0;
                if (m > 1) T[1] = z;
                for (int tt = 2; tt < m; ++tt) T[tt] = 2.0 * z * T[tt-1] - T[tt-2];

                // accumulate weighted outer product into lower-triangular G_dd (j >= k)
                for (int jj = 0; jj < m; ++jj) {
                    double Tj = T[jj];
                    for (int kk = 0; kk <= jj; ++kk) {
                        double Tk = T[kk];
                        double add = synth_weight * (Tj * Tk);
                        // G_dd[jj][kk] is lower-triangular storage (same convention used later)
                        G_dd[jj][kk] = dd_add_double(G_dd[jj][kk], add);
                    }
                }
                // accumulate weighted rhs into b_dd
                for (int jj = 0; jj < m; ++jj) {
                    double addb = synth_weight * (yi * T[jj]);
                    b_dd[jj] = dd_add_double(b_dd[jj], addb);
                }
            } // inserts
        } // gaps
    } // if n>1

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

// qp_solve: quadratic program solver (primal-dual interior point, equality + inequality).
auto qp_solve(const Eigen::MatrixXd &ATA, const Eigen::VectorXd &ATb,
    const std::vector<std::vector<double>>& Aeq_rows, const std::vector<double>& beq,
    const std::vector<std::vector<double>>& Ain_rows, const std::vector<double>& bin, int m) -> std::pair<bool, Eigen::VectorXd> {

    using std::abs;

    // Basic shape checks
    int neq = static_cast<int>(Aeq_rows.size());
    int nin = static_cast<int>(Ain_rows.size());
    assert(m >= 0);

    // Defensive checks: all rows must have length m
    for (const auto &row : Aeq_rows) assert((int)row.size() == m);
    for (const auto &row : Ain_rows) assert((int)row.size() == m);
    assert((int)beq.size() == neq);
    assert((int)bin.size() == nin);

    // Convert Aeq and Ain to Eigen matrices (zero-sized handled)
    Eigen::MatrixXd Aeq = Eigen::MatrixXd::Zero(std::max(0, neq), m);
    Eigen::VectorXd be = Eigen::VectorXd::Zero(std::max(0, neq));
    if (neq > 0) {
        for (int i = 0; i < neq; ++i) {
            be(i) = beq[i];
            for (int j = 0; j < m; ++j) Aeq(i, j) = Aeq_rows[i][j];
        }
    }

    Eigen::MatrixXd Ain = Eigen::MatrixXd::Zero(std::max(0, nin), m);
    Eigen::VectorXd bi = Eigen::VectorXd::Zero(std::max(0, nin));
    if (nin > 0) {
        for (int i = 0; i < nin; ++i) {
            bi(i) = bin[i];
            for (int j = 0; j < m; ++j) Ain(i, j) = Ain_rows[i][j];
        }
    }

    // If no inequalities -> equality constrained KKT solve and return
    if (nin == 0) {
        int K = m + neq;
        Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(K, K);
        if (m > 0) KKT.topLeftCorner(m, m) = ATA;               // top-left: ATA
        if (neq > 0) {
            // top-right = Aeq^T, bottom-left = Aeq
            KKT.topRightCorner(m, neq) = Aeq.transpose();
            KKT.bottomLeftCorner(neq, m) = Aeq;
        }
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(K);
        if (m > 0) rhs.head(m) = ATb;
        if (neq > 0) rhs.segment(m, neq) = be;

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(KKT);
        Eigen::VectorXd sol = solver.solve(rhs);
        if (solver.info() != Eigen::Success) return {false, Eigen::VectorXd()};
        if (!std::isfinite(sol.norm())) return {false, Eigen::VectorXd()};
        return {true, sol.head(m)};
    }

    // General case with inequalities: build initial guess via KKT on equalities (if any)
    Eigen::VectorXd c;
    {
        int K = m + neq;
        Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(K, K);
        if (m > 0) KKT.topLeftCorner(m, m) = ATA;
        if (neq > 0) {
            KKT.topRightCorner(m, neq) = Aeq.transpose();
            KKT.bottomLeftCorner(neq, m) = Aeq;
        }
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(K);
        if (m > 0) rhs.head(m) = ATb;
        if (neq > 0) rhs.segment(m, neq) = be;

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(KKT);
        Eigen::VectorXd sol = solver.solve(rhs);
        if (solver.info() == Eigen::Success && std::isfinite(sol.norm())) {
            c = sol.head(m);
        } else {
            c = robustSymmetricSolve(ATA, ATb); // fallback
        }
    }

    // Initialize slack s and dual z for inequalities
    Eigen::VectorXd s = Eigen::VectorXd::Zero(nin);
    Eigen::VectorXd z = Eigen::VectorXd::Ones(nin);
    for (int i = 0; i < nin; ++i) {
        double si = bin[i] - double(Ain.row(i).dot(c));
        if (si <= kEps) si = 1.0; // ensure positive initial slack
        s(i) = si;
        z(i) = 1.0;
    }
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(std::max(0, neq));

    const int max_iters = 60;
    const double tol = 1e-10;
    double mu = (nin > 0) ? (s.dot(z)) / double(nin) : 1.0;
    if (!std::isfinite(mu) || mu <= kEps) mu = 1.0;

    // Pre-create bin_vec safely (avoid ternary mixing Eigen expression types)
    Eigen::VectorXd bin_vec;
    if (nin > 0) {
        bin_vec = Eigen::Map<const Eigen::VectorXd>(bin.data(), nin);
    } else {
        bin_vec = Eigen::VectorXd::Zero(0);
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        // residuals
        Eigen::VectorXd r1 = ATA * c - ATb;
        if (neq > 0) r1 += Aeq.transpose() * lambda;
        r1 += Ain.transpose() * z;

        Eigen::VectorXd r2 = Eigen::VectorXd::Zero(std::max(0, neq));
        if (neq > 0) r2 = Aeq * c - be;

        Eigen::VectorXd r3 = Ain * c + s - bin_vec;

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

        // Build M = ATA + sum_i w_i * ai^T ai  (use temporaries to avoid aliasing)
        Eigen::MatrixXd M = ATA; // copy
        for (int i = 0; i < nin; ++i) {
            double w = SinvZ(i);
            if (abs(w) < kEps) continue;
            // materialize row as column vector to form outer product safely
            Eigen::VectorXd ai_col = Ain.row(i).transpose();           // m x 1
            Eigen::MatrixXd outer = ai_col * ai_col.transpose();      // m x m
            M.noalias() += w * outer;
        }

        Eigen::VectorXd tmp = r4 - Z.cwiseProduct(r3);

        // build 'add' vector safely using temporaries
        Eigen::VectorXd add = Eigen::VectorXd::Zero(m);
        for (int i = 0; i < nin; ++i) {
            double coef = Sinv(i) * tmp(i);
            if (abs(coef) < kEps) continue;
            Eigen::VectorXd ai_col = Ain.row(i).transpose();
            add += coef * ai_col;
        }
        Eigen::VectorXd rhs_reduced = -r1 + add;
        Eigen::VectorXd rhs_eq = -r2;

        // Assemble KKT for reduced system:
        int K = m + neq;
        Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(K, K);
        if (m > 0) KKT.topLeftCorner(m, m) = M;
        if (neq > 0) {
            KKT.topRightCorner(m, neq) = Aeq.transpose();
            KKT.bottomLeftCorner(neq, m) = Aeq;
        }

        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(K);
        if (m > 0) rhs.head(m) = rhs_reduced;
        if (neq > 0) rhs.segment(m, neq) = rhs_eq;

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solver(KKT);
        Eigen::VectorXd sol = solver.solve(rhs);
        if (solver.info() != Eigen::Success) return {false, Eigen::VectorXd()};
        if (!std::isfinite(sol.norm())) return {false, Eigen::VectorXd()};

        Eigen::VectorXd dc = sol.head(m);
        Eigen::VectorXd dl = (neq > 0) ? sol.segment(m, neq) : Eigen::VectorXd();

        Eigen::VectorXd ds = -r3 - Ain * dc;
        Eigen::VectorXd dz = Eigen::VectorXd::Zero(nin);
        for (int i = 0; i < nin; ++i) {
            dz(i) = Sinv(i) * ( -r4(i) - Z(i) * ds(i) );
        }

        // step length
        double alpha_pr = 1.0, alpha_du = 1.0;
        for (int i = 0; i < nin; ++i) {
            if (ds(i) < -kEps) alpha_pr = std::min(alpha_pr, -s(i) / ds(i));
            if (dz(i) < -kEps) alpha_du = std::min(alpha_du, -z(i) / dz(i));
        }
        const double tau = 0.995;
        alpha_pr = std::min(1.0, tau * alpha_pr);
        alpha_du = std::min(1.0, tau * alpha_du);

        // primal/dual update
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

    // final feasibility check for inequalities
    Eigen::VectorXd ineq_res = Ain * c - bin_vec;
    double max_viol = 0.0;
    if (nin > 0) {
        max_viol = ineq_res.maxCoeff();
    } else {
        max_viol = -std::numeric_limits<double>::infinity(); // not used
    }
    if (max_viol > kFeasTol) {
        return {false, Eigen::VectorXd()};
    }
    return {true, c};
}

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
    for (int i = 0; i < m; ++i) {
        p1d[i] = double(p1_coeffs[i]);
        p2d[i] = double(p2_coeffs[i]);
    }

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
        // zero-init ATA to avoid uninitialized memory
        ATA_out = Eigen::MatrixXd::Zero(m, m);

        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) {
                int n = r + c;
                double I0 = 1.0 / double(n + 1);     // ∫ x^n over [0,1]
                double I1 = 1.0 / double(n + 3);     // ∫ x^{n+2}
                double I2 = 1.0 / double(n + 2);     // ∫ x^{n+1}
                // weighted objective uses continuity/deriv weighting expressed w.r.t. normalized segments
                double weighted = I0 + alpha * (I1 - 2.0 * w1 * I2 + w1 * w1 * I0);
                ATA_out(r, c) = weighted;
            }
        }

        // ATb as double-double accumulator per coefficient
        ATb_dd_out.assign(m, dd(0.0));

        // part 1: contribution from p1 on [0,w1] via substitution x = w1 * u, dx = w1 du
        if (w1 > kEps) {
            // w1_pows[r] := w1^{r+1}  (since factor from x^r and dx)
            std::vector<double> w1_pows(m, 1.0);
            if (m > 0) {
                double cur = w1;
                for (int r = 0; r < m; ++r) { w1_pows[r] = cur; cur *= w1; }
            }

            for (int r = 0; r < m; ++r) {
                double factor = w1_pows[r];
                for (int k = 0; k <= degree; ++k) {
                    double I0 = 1.0 / double(k + r + 1);   // ∫ u^{k+r} du over [0,1]
                    double I2 = 1.0 / double(k + r + 3);   // ∫ u^{k+r+2}
                    double I1 = 1.0 / double(k + r + 2);   // ∫ u^{k+r+1}
                    // note: when mapping derivatives, scale factors appear (w1^2)
                    double term = p1d[k] * (I0 + alpha * (w1 * w1) * (I2 - 2.0 * I1 + I0));
                    ATb_dd_out[r] = dd_add_double(ATb_dd_out[r], factor * term);
                }
            }
        }

        // part 2: contribution from p2 on [w1,1] via substitution x = w1 + w2 * u, dx = w2 du
        if (w2 > kEps) {
            // Precompute powers of w1 and w2
            std::vector<double> w1_pows(m, 1.0);
            std::vector<double> w2_pows(m, 1.0);
            for (int i = 1; i < m; ++i) {
                w1_pows[i] = w1_pows[i-1] * w1;
                w2_pows[i] = w2_pows[i-1] * w2;
            }

            for (int r = 0; r < m; ++r) {
                for (int k = 0; k <= degree; ++k) {
                    double inner = 0.0;
                    double inner_weighted = 0.0;
                    // loop over binomial expansion terms t = 0..r
                    for (int t = 0; t <= r; ++t) {
                        double b = comb(r, t);
                        double w1pow = w1_pows[r - t]; // w1^{r-t}
                        double w2pow = w2_pows[t];     // w2^{t}
                        double Ibase = 1.0 / double(k + t + 1);  // ∫ u^{k+t}
                        double Iplus2 = 1.0 / double(k + t + 3); // ∫ u^{k+t+2}
                        inner += b * w1pow * w2pow * Ibase;
                        inner_weighted += b * w1pow * w2pow * Iplus2;
                    }
                    // multiply by p2 coefficient and the derivative-weight term
                    double term = p2d[k] * (inner + alpha * (w2 * w2) * inner_weighted);
                    // overall dx gives an extra factor w2
                    ATb_dd_out[r] = dd_add_double(ATb_dd_out[r], term * w2);
                }
            }
        }

        // convert dd accumulators to double vector
        ATb_out = Eigen::VectorXd::Zero(m);
        for (int r = 0; r < m; ++r) ATb_out(r) = dd_to_double(ATb_dd_out[r]);
    };

    // exact ∫ f^2 dx constant used in objective
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

    // polynomial evaluation helpers (Horner)
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

    // global evaluators on x in [0,1]
    auto eval_f_global = [&](double x)->double {
        if (x <= w1 + kFuzzyBoundary) {
            if (w1 <= kEps) return eval_poly(p1d, 0.0);
            double u = x / w1;
            return eval_poly(p1d, u);
        } else {
            if (w2 <= kEps) return eval_poly(p2d, 0.0);
            double u = (x - w1) / w2;
            return eval_poly(p2d, u);
        }
    };
    auto eval_fprime_global = [&](double x)->double {
        if (x <= w1 + kFuzzyBoundary) {
            double scale = (abs(w1) < kEps) ? 1.0 : w1;
            if (scale <= kEps) return 0.0;
            double u = x / scale;
            double val = eval_poly(p1d_deriv, u);
            return val / scale;
        } else {
            double scale = (abs(w2) < kEps) ? 1.0 : w2;
            if (scale <= kEps) return 0.0;
            double u = (x - w1) / scale;
            double val = eval_poly(p2d_deriv, u);
            return val / scale;
        }
    };
    auto eval_fsecond_global = [&](double x)->double {
        if (x <= w1 + kFuzzyBoundary) {
            double scale = (abs(w1) < kEps) ? 1.0 : w1;
            if (scale <= kEps) return 0.0;
            double u = x / scale;
            double val = eval_poly(p1d_second, u);
            return val / (scale * scale);
        } else {
            double scale = (abs(w2) < kEps) ? 1.0 : w2;
            if (scale <= kEps) return 0.0;
            double u = (x - w1) / scale;
            double val = eval_poly(p2d_second, u);
            return val / (scale * scale);
        }
    };

    // find interior extrema: root finding on derivative polynomials in unit interval
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

    // prepare interior_extrema (filtered)
    std::vector<double> interior_extrema;
    for (auto &e : scored) {
        if (e.x <= kFuzzyBoundary || e.x >= 1.0 - kFuzzyBoundary) continue;
        if (std::abs(e.x - w1) < kFuzzyBoundary) continue;
        interior_extrema.push_back(e.x);
    }

    // boundary & endpoints
    std::vector<double> endpoints = {0.0, 1.0};
    std::vector<double> endpoint_values = {f0, f1};

    // main candidate search
    for (double alpha : alphas) {
        Eigen::MatrixXd ATA_alpha;
        Eigen::VectorXd ATb_alpha;
        std::vector<dd> ATb_dd_tmp;
        build_weighted_ATA_ATb(alpha, ATA_alpha, ATb_alpha, ATb_dd_tmp);

        // small ridge to stabilize ill-conditioning
        double trace = ATA_alpha.trace();
        double ridge = 1e-14 * std::max(1.0, std::abs(trace));
        for (int i = 0; i < m; ++i) ATA_alpha(i,i) += ridge;

        for (int keepCount : keep_counts) {
            // build equality constraints
            std::vector<std::vector<double>> C_eq;
            std::vector<double> d_eq;
            // endpoint values
            for (int ei = 0; ei < 2; ++ei) { C_eq.push_back(build_value_row(endpoints[ei])); d_eq.push_back(endpoint_values[ei]); }
            // endpoint derivative if stationary
            if (preserve_d0) { C_eq.push_back(build_deriv_row(0.0)); d_eq.push_back(0.0); }
            if (preserve_d1) { C_eq.push_back(build_deriv_row(1.0)); d_eq.push_back(0.0); }
            // boundary C0/C1 targets
            C_eq.push_back(build_value_row(w1)); d_eq.push_back(boundary_val_target);
            C_eq.push_back(build_deriv_row(w1)); d_eq.push_back(boundary_d_target);

            // add interior extrema constraints (value + derivative)
            for (int k = 0; k < keepCount && k < (int)interior_extrema.size(); ++k) {
                double x = interior_extrema[k];
                C_eq.push_back(build_value_row(x)); d_eq.push_back(eval_f_global(x));
                if (m >= 2) { C_eq.push_back(build_deriv_row(x)); d_eq.push_back(0.0); }
            }

            // if equality constraints already saturate DOF, skip (we cannot fit)
            if ((int)C_eq.size() >= m) {
                continue;
            }

            // build inequalities (sign of second derivative at extrema)
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

            // compute exact error safely (avoid chained Product expressions)
            Eigen::VectorXd v = ATA_alpha * c;
            double err = c.dot(v) - 2.0 * ATb_alpha.dot(c) + F2_const;

            if (err < best_err) {
                best_err = err;
                for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(c(i));
                // small early-exit: if we have an almost-zero residual relative to F2_const, stop
                if (best_err <= 1e-14 * std::max(1.0, std::abs(F2_const))) {
                    break;
                }
            }
        } // keepCount

        if (best_err <= 1e-14 * std::max(1.0, std::abs(F2_const))) break;
    } // alpha

    // --- Robust selection: compare with unconstrained LS (alpha=0) and choose the better-behaved solution ---
    // build base L2 ATA0/ATb0
    Eigen::MatrixXd ATA0; Eigen::VectorXd ATb0; std::vector<dd> tmp_dd;
    build_weighted_ATA_ATb(0.0, ATA0, ATb0, tmp_dd);
    double trace0 = ATA0.trace();
    double ridge0 = 1e-14 * std::max(1.0, std::abs(trace0));
    for (int i = 0; i < m; ++i) ATA0(i,i) += ridge0;

    Eigen::VectorXd c_ls = robustSymmetricSolve(ATA0, ATb0);
    bool ls_ok = std::isfinite(c_ls.norm());
    double err_ls = std::numeric_limits<double>::infinity();
    if (ls_ok) {
        Eigen::VectorXd vl = ATA0 * c_ls;
        err_ls = c_ls.dot(vl) - 2.0 * ATb0.dot(c_ls) + F2_const;
        DEBUG_PRINT("composePolynomials: unconstrained LS: ||c_ls||=%g err_ls=%g\n", c_ls.norm(), err_ls);
    }

    Eigen::VectorXd c_qp = Eigen::VectorXd::Zero(m);
    for (int i = 0; i < m; ++i) c_qp(i) = best_coeffs[i];
    double err_qp = std::numeric_limits<double>::infinity();
    if (std::isfinite(best_err)) {
        Eigen::VectorXd vq = ATA0 * c_qp; // compare in consistent quadratic form
        err_qp = c_qp.dot(vq) - 2.0 * ATb0.dot(c_qp) + F2_const;
        DEBUG_PRINT("composePolynomials: QP best: ||c_qp||=%g err_qp=%g\n", c_qp.norm(), err_qp);
    }

    // decision heuristic: prefer LS if it materially reduces residual or QP exploded
    bool use_ls = false;
    if (ls_ok) {
        if (!std::isfinite(best_err)) use_ls = true;
        else if (err_ls < 0.9 * err_qp) use_ls = true;  // LS noticeably better
        else if (c_qp.norm() > 1e6 && c_ls.norm() < 1e3) use_ls = true; // QP exploded
    }
    if (use_ls) {
        DEBUG_PRINT("composePolynomials: selecting unconstrained LS solution (safer).\n");
        std::vector<float> out(m);
        for (int i = 0; i < m; ++i) out[i] = static_cast<float>(c_ls(i));
        return out;
    }

    // If QP gave a finite best_err, use it
    if (std::isfinite(best_err) && best_err < std::numeric_limits<double>::infinity()) {
        return best_coeffs;
    }

    // Fallback levels (attempt constrained solves with progressively fewer constraints)
    DEBUG_PRINT("--- AdvancedPolynomialFitter: Main QP search failed. Entering fallback mode. ---\n");
    // Fallback 0: C0 endpoints + C0 boundary
    {
        std::vector<std::vector<double>> C_eq_fb0; std::vector<double> d_eq_fb0;
        C_eq_fb0.push_back(build_value_row(0.0)); d_eq_fb0.push_back(f0);
        C_eq_fb0.push_back(build_value_row(1.0)); d_eq_fb0.push_back(f1);
        C_eq_fb0.push_back(build_value_row(w1)); d_eq_fb0.push_back(boundary_val_target);

        if ((int)C_eq_fb0.size() < m) {
            auto sol0 = qp_solve(ATA0, ATb0, C_eq_fb0, d_eq_fb0, {}, {}, m); // No inequalities
            if (sol0.first) {
                for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(sol0.second(i));
                return best_coeffs;
            }
        }
    }

    // Fallback 1: endpoints only
    {
        std::vector<std::vector<double>> C_eq_fb1; std::vector<double> d_eq_fb1;
        C_eq_fb1.push_back(build_value_row(0.0)); d_eq_fb1.push_back(f0);
        C_eq_fb1.push_back(build_value_row(1.0)); d_eq_fb1.push_back(f1);

        if ((int)C_eq_fb1.size() < m) {
            auto sol1 = qp_solve(ATA0, ATb0, C_eq_fb1, d_eq_fb1, {}, {}, m);
            if (sol1.first) {
                for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(sol1.second(i));
                return best_coeffs;
            }
        }
    }

    // Final fallback: unconstrained least-squares (ATA0 already computed)
    {
        Eigen::VectorXd c_uncon = robustSymmetricSolve(ATA0, ATb0);
        if (std::isfinite(c_uncon.norm())) {
            std::vector<float> out(m);
            for (int i = 0; i < m; ++i) out[i] = static_cast<float>(c_uncon(i));
            return out;
        }
    }

    // absolute failure
    DEBUG_PRINT("--- CRITICAL: All fallbacks failed. Returning zeros. ---\n");
    return std::vector<float>(m, 0.0f);
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


// experimental : Lebesgue integral instead of Riemann integral fitter 


// small combinatorics helper
static double binomial_double(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k == 0 || k == n) return 1.0;
    if (k > n/2) k = n - k;
    double res = 1.0;
    for (int i = 1; i <= k; ++i) res = res * double(n - i + 1) / double(i);
    return res;
}

// compute integrals I_p = ∫ x^p * y_lin(x) dx over domain union of intervals defined by (x_in,y_in) piecewise-linear
// returns vector I of length (maxPow+1), uses dd accumulation
static std::vector<double> integrate_xp_times_piecewise_linear_y(const std::vector<double>& x_in,
                                                                 const std::vector<float>& y_in,
                                                                 int maxPow)
{
    size_t n = x_in.size();
    std::vector<std::vector<dd>> accum(maxPow + 1, std::vector<dd>(1, dd(0.0))); // use dd per power
    for (size_t p = 0; p <= (size_t)maxPow; ++p) accum[p][0] = dd(0.0);

    if (n <= 1) {
        // degenerate: treat single point as atomic measure: ∫ x^p y dx -> y0 * x0^p * small epsilon (practical fallback)
        if (n == 1) {
            double x0 = x_in[0];
            double y0 = double(y_in[0]);
            for (int p = 0; p <= maxPow; ++p) accum[p][0] = dd_add_double(accum[p][0], y0 * std::pow(x0, double(p)) * 0.0); // zero-length domain
        }
    } else {
        // sort indices by x (no copying of y)
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0u);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return x_in[a] < x_in[b]; });

        for (size_t s = 0; s + 1 < n; ++s) {
            size_t i0 = idx[s], i1 = idx[s+1];
            double x0 = x_in[i0];
            double x1 = x_in[i1];
            if (!(x1 > x0)) continue;
            double h = x1 - x0;
            double y0 = double(y_in[i0]);
            double y1 = double(y_in[i1]);
            // affine y(x) = alpha + beta * x
            double beta = (y1 - y0) / h;
            double alpha = y0 - beta * x0;

            // integrals J_p = ∫_{x0}^{x1} x^p dx needed up to p = maxPow+1 (for alpha*J_p + beta*J_{p+1})
            for (int p = 0; p <= maxPow; ++p) {
                // J_p = (x1^{p+1} - x0^{p+1}) / (p+1)
                double Jp = (std::pow(x1, double(p + 1)) - std::pow(x0, double(p + 1))) / double(p + 1);
                double Jp1 = (std::pow(x1, double(p + 2)) - std::pow(x0, double(p + 2))) / double(p + 2);
                double add = alpha * Jp + beta * Jp1;
                accum[p][0] = dd_add_double(accum[p][0], add);
            }
        }
    }

    std::vector<double> out(maxPow + 1);
    for (int p = 0; p <= maxPow; ++p) out[p] = dd_to_double(accum[p][0]);
    return out;
}

// Project piecewise-linear y onto shifted Legendre basis on [xmin,xmax] analytically.
// returns coefficients in monomial basis for x^0..x^degree
std::vector<float> AdvancedPolynomialFitter::fitPolynomialLebesgueL2_AnalyticProjection(
    const std::vector<double>& x_in,
    const std::vector<float>& y_in,
    int degree,
    OptimizationMethod method)
{
    if (x_in.size() != y_in.size() || x_in.empty() || degree < 0) return {};
    int m = degree + 1;

    // domain
    double xmin = x_in[0], xmax = x_in[0];
    for (size_t i = 1; i < x_in.size(); ++i) {
        if (x_in[i] < xmin) xmin = x_in[i];
        if (x_in[i] > xmax) xmax = x_in[i];
    }
    double scale = xmax - xmin;
    if (scale == 0.0) scale = 1.0; // degenerate domain -> fallback scale

    // get shifted Legendre monomial coefficients L[k][t] where P_k(u)=sum_t L[k][t] u^t, u in [0,1]
    VecMatrix L = getShiftedLegendreMonomialCoeffs(degree); // L.size() == m, L[k].size()==k+1

    // Need integrals Iy_j = ∫ x^j y_lin(x) dx for j up to degree (but we will need up to t where t runs up to degree)
    // however u^t expands to x^j up to j=t, but when we compute P_k(u) y dx we need Iy_j for j up to degree (safe to compute up to degree)
    // To be safe compute up to degree (because highest t <= degree, and binomial expands to x^j where j<=t <= degree)
    int maxIyPow = degree;
    std::vector<double> Iy = integrate_xp_times_piecewise_linear_y(x_in, y_in, maxIyPow);

    // compute inner products <P_k, y> = ∫ P_k(u(x)) * y(x) dx
    // P_k(u) = sum_t L[k][t] u^t  with u = (x - xmin)/scale.
    // u^t = scale^{-t} * sum_{j=0..t} binom(t,j) * (-xmin)^{t-j} * x^j
    std::vector<double> inner(m, 0.0);
    for (int k = 0; k < m; ++k) {
        double accum_k = 0.0;
        for (int t = 0; t <= k; ++t) {
            double Lkt = L[k][t]; // coefficient of u^t
            if (Lkt == 0.0) continue;
            double invScalePow = std::pow(scale, -double(t));
            // expand u^t into x^j
            double inner_sum = 0.0;
            for (int j = 0; j <= t; ++j) {
                double comb = binomial_double(t, j);
                double mul = comb * std::pow(-xmin, double(t - j)) * invScalePow;
                // Iy[j] = ∫ x^j y dx
                inner_sum += mul * Iy[j];
            }
            accum_k += Lkt * inner_sum;
        }
        inner[k] = accum_k;
    }

    // norms: ∫ P_k(u)^2 dx = scale * ∫_0^1 P_k(u)^2 du = scale / (2*k + 1)
    std::vector<double> norms(m, 0.0);
    for (int k = 0; k < m; ++k) norms[k] = scale / double(2 * k + 1);

    // projection coefficients in Legendre basis: a_k = inner_k / norm_k
    std::vector<double> a_leg(m, 0.0);
    for (int k = 0; k < m; ++k) {
        double nrm = norms[k];
        if (!(nrm > 0.0)) nrm = 1e-30;
        a_leg[k] = inner[k] / nrm;
    }

    // convert back to x-monomial basis:
    // p(x) = sum_k a_leg[k] * P_k(u(x)) = sum_k a_leg[k] * sum_{t=0..k} L[k][t] u^t
    // expand u^t into x^j as before; accumulate into monomial coefficients c_x[j]
    std::vector<dd> c_x_dd(m);
    for (int j = 0; j < m; ++j) c_x_dd[j] = dd(0.0);

    for (int k = 0; k < m; ++k) {
        double ak = a_leg[k];
        if (ak == 0.0) continue;
        for (int t = 0; t <= k; ++t) {
            double coef_ut = L[k][t] * ak;
            if (coef_ut == 0.0) continue;
            double invScalePow = std::pow(scale, -double(t));
            for (int j = 0; j <= t; ++j) {
                double comb = binomial_double(t, j);
                double contrib = coef_ut * comb * std::pow(-xmin, double(t - j)) * invScalePow;
                c_x_dd[j] = dd_add_double(c_x_dd[j], contrib);
            }
        }
    }

    std::vector<float> out(m);
    for (int j = 0; j < m; ++j) out[j] = static_cast<float>(dd_to_double(c_x_dd[j]));
    return out;
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
