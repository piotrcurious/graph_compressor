// ----------------- Helpers for L1 Lebesgue fitter -----------------
#include <complex>

static std::vector<double> trim_leading_zeros_poly(const std::vector<double>& p) {
    int deg = (int)p.size() - 1;
    while (deg > 0 && std::abs(p[deg]) < 1e-18) --deg;
    return std::vector<double>(p.begin(), p.begin() + (deg + 1));
}

// Find real roots of polynomial p(x) = sum_{k=0..d} p[k] x^k inside (a,b) using companion matrix.
// Returns sorted unique real roots inside [a - tol, b + tol].
static std::vector<double> find_real_roots_interval(const std::vector<double>& poly, double a, double b) {
    std::vector<double> roots;
    auto p = trim_leading_zeros_poly(poly);
    int deg = (int)p.size() - 1;
    const double tol = 1e-10;
    const double imagTol = 1e-8;
    if (deg <= 0) return roots;
    if (deg == 1) {
        double A = p[1], B = p[0];
        if (std::abs(A) > 0.0) {
            double r = -B / A;
            if (r >= a - tol && r <= b + tol) roots.push_back(std::min(b, std::max(a, r)));
        }
        return roots;
    }
    // make monic: a_n x^n + ... + a0
    double an = p[deg];
    if (std::abs(an) < 1e-300) return roots; // degenerate
    // Build companion matrix (deg x deg)
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(deg, deg);
    // first row: -a_{n-1}/a_n ... -a_0/a_n
    for (int j = 0; j < deg; ++j) C(0, j) = -p[deg - 1 - j] / an;
    // subdiagonal ones
    for (int i = 1; i < deg; ++i) C(i, i - 1) = 1.0;
    Eigen::EigenSolver<Eigen::MatrixXd> es(C, false);
    Eigen::VectorXcd eigs = es.eigenvalues();
    for (int i = 0; i < eigs.size(); ++i) {
        std::complex<double> z = eigs[i];
        if (std::abs(z.imag()) < imagTol) {
            double r = z.real();
            if (r >= a - tol && r <= b + tol) roots.push_back(std::min(b, std::max(a, r)));
        }
    }
    std::sort(roots.begin(), roots.end());
    // unique
    roots.erase(std::unique(roots.begin(), roots.end(), [&](double u, double v){ return std::abs(u - v) < 1e-9; }), roots.end());
    return roots;
}

// Evaluate polynomial (coeffs in increasing order) at x
static double eval_poly(const std::vector<double>& c, double x) {
    double r = 0.0;
    for (int k = (int)c.size() - 1; k >= 0; --k) r = r * x + c[k];
    return r;
}

// Antiderivative of polynomial q(x) = sum_{k=0..d} q[k] x^k: F(x) = sum q[k] x^{k+1}/(k+1)
// and affine part - integrate alpha + beta x separately.
static double antideriv_poly_minus_affine_at(const std::vector<double>& q, double alpha, double beta, double x) {
    double s = 0.0;
    double xp = x; // x^{1}
    for (int k = 0; k < (int)q.size(); ++k) {
        // term q[k] * x^{k+1} / (k+1)
        double denom = double(k + 1);
        // compute x^{k+1} incrementally: xp = x^{k+1}
        double term = q[k] * xp / denom;
        s += term;
        xp *= x;
    }
    // subtract affine antiderivative: alpha * x + 0.5 * beta * x^2
    s -= (alpha * x + 0.5 * beta * x * x);
    return s;
}

// Compute exact integral of |p(x) - (alpha + beta x)| over [a,b].
// p is coeffs [c0..c_deg] (increasing order).
static double integrate_abs_residual_over_interval(const std::vector<double>& p,
                                                   double alpha, double beta,
                                                   double a, double b) {
    // residual r(x) = p(x) - (alpha + beta x)
    // form coefficients rcoef = p with adjusted constant and linear terms
    std::vector<double> rcoef = p;
    if (rcoef.size() < 2) rcoef.resize(2, 0.0);
    rcoef[0] -= alpha;
    rcoef[1] -= beta;
    rcoef = trim_leading_zeros_poly(rcoef);
    int deg = (int)rcoef.size() - 1;
    if (deg < 0) return 0.0;
    if (deg == 0) {
        // constant residual
        return std::abs(rcoef[0]) * (b - a);
    }
    // find interior roots in (a,b)
    std::vector<double> roots = find_real_roots_interval(rcoef, a, b);
    // create segments: [a, r0, r1, ..., b]
    std::vector<double> pts;
    pts.push_back(a);
    for (double r : roots) {
        // clip roots to [a,b]
        double rr = std::min(b, std::max(a, r));
        if (rr > a + 1e-15 && rr < b - 1e-15) pts.push_back(rr);
    }
    pts.push_back(b);
    std::sort(pts.begin(), pts.end());
    // integrate on each subinterval with sign determined by midpoint
    double total = 0.0;
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        double L = pts[i], R = pts[i+1];
        double mid = 0.5 * (L + R);
        double sign = std::signbit(eval_poly(rcoef, mid)) ? -1.0 : 1.0;
        double Fa = antideriv_poly_minus_affine_at(rcoef, 0.0, 0.0, L); // antiderivative of r (without subtracting affine here because r includes it)
        double Fb = antideriv_poly_minus_affine_at(rcoef, 0.0, 0.0, R);
        // But antideriv_poly_minus_affine_at expects p and alpha,beta to subtract; since rcoef already is p-(alpha+beta x) we call with alpha=0,beta=0 and then add back the affine antiderivative sign accordingly.
        double integral_segment = Fb - Fa; // ∫ r(x) dx on [L,R]
        total += std::abs(integral_segment); // sign determined by checking midpoint
        // Note: using abs(integral_segment) is equivalent since we checked sign at midpoint; numerical sign flip very near roots could happen, but roots inclusion was clipped tightly.
    }
    return total;
}

// Build ATA and ATb for weighted integral objective: sum_i w_i ∫_{x_i}^{x_{i+1}} (p(x)-y_lin(x))^2 dx
// here weights_per_interval must be of size n-1 corresponding to intervals between sorted x.
static void buildWeightedATA_ATb_piecewise_linear(const std::vector<double>& x_in,
                                                  const std::vector<float>& y_in,
                                                  const std::vector<double>& weights_per_interval,
                                                  int degree,
                                                  Eigen::MatrixXd &ATA_out,
                                                  Eigen::VectorXd &ATb_out)
{
    size_t n = x_in.size();
    int m = degree + 1;
    ATA_out = Eigen::MatrixXd::Zero(m, m);
    ATb_out = Eigen::VectorXd::Zero(m);
    if (n <= 1) return;

    // sort indices
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return x_in[a] < x_in[b]; });

    // dd accumulators lower-triangular
    std::vector<std::vector<dd>> ATA_dd(m, std::vector<dd>(m, dd(0.0)));
    std::vector<dd> ATb_dd(m, dd(0.0));

    for (size_t s = 0; s + 1 < n; ++s) {
        size_t i0 = idx[s], i1 = idx[s+1];
        double x0 = x_in[i0];
        double x1 = x_in[i1];
        if (!(x1 > x0)) continue;
        double h = x1 - x0;
        double y0 = double(y_in[i0]);
        double y1 = double(y_in[i1]);
        double beta = (y1 - y0) / h;
        double alpha = y0 - beta * x0;
        double w = (s < weights_per_interval.size()) ? weights_per_interval[s] : 1.0;
        if (!(w > 0.0)) w = 1.0;

        int maxPow = 2 * degree;
        std::vector<double> I(maxPow + 1);
        for (int p = 0; p <= maxPow; ++p) I[p] = integral_pow(x0, x1, p);

        // ATA contribution: ∫_x0^x1 x^{r+c} dx * w
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c <= r; ++c) {
                double add = w * I[r + c];
                ATA_dd[r][c] = dd_add_double(ATA_dd[r][c], add);
            }
        }

        // ATb contribution: ∫ x^r * (alpha + beta x) dx * w
        for (int r = 0; r < m; ++r) {
            double addb = w * (alpha * I[r] + beta * ( (r + 1 <= maxPow) ? I[r + 1] : integral_pow(x0, x1, r + 1)));
            ATb_dd[r] = dd_add_double(ATb_dd[r], addb);
        }
    }

    for (int j = 0; j < m; ++j) {
        ATb_out(j) = dd_to_double(ATb_dd[j]);
        for (int k = 0; k <= j; ++k) {
            double v = dd_to_double(ATA_dd[j][k]);
            ATA_out(j, k) = v;
            ATA_out(k, j) = v;
        }
    }
}

// ----------------- Public API: fitPolynomialLebesgueL1_IRLS -----------------
// Iteratively reweighted least squares approximating L1 with exact interval integrals
std::vector<float> AdvancedPolynomialFitter::fitPolynomialLebesgueL1_IRLS(const std::vector<double>& x,
                                                                          const std::vector<float>& y,
                                                                          int degree,
                                                                          int maxIters /*= 60*/,
                                                                          double tol /*= 1e-9*/)
{
    if (x.size() != y.size() || x.empty() || degree < 1) return {};
    if (x.size() < 2) {
        // degenerate: fallback to L2 Lebesgue or discrete LS
        return fitPolynomialLebesgueL2(x, y, degree);
    }

    size_t n = x.size();
    // initial solution from L2 Lebesgue (good warm start)
    std::vector<float> coeffs = fitPolynomialLebesgueL2(x, y, degree);
    int m = degree + 1;
    std::vector<double> c_old(m);
    for (int i = 0; i < m; ++i) c_old[i] = double(coeffs[i]);

    // sort indices once
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return x[a] < x[b]; });

    const double eps = 1e-12;
    const double weight_floor = 1e-12;
    const double weight_cap = 1e12;

    for (int iter = 0; iter < maxIters; ++iter) {
        // compute per-interval average absolute residual (exact)
        std::vector<double> weights;
        weights.reserve(n > 1 ? n - 1 : 0);
        double max_avg = 0.0;
        for (size_t s = 0; s + 1 < n; ++s) {
            size_t i0 = idx[s], i1 = idx[s+1];
            double x0 = x[i0], x1 = x[i1];
            if (!(x1 > x0)) { weights.push_back(1.0); continue; }
            double h = x1 - x0;
            double y0 = double(y[i0]), y1 = double(y[i1]);
            double beta = (y1 - y0) / h;
            double alpha = y0 - beta * x0;
            // residual polynomial r(x) = p(x) - (alpha + beta x)
            std::vector<double> pcoef(m);
            for (int k = 0; k < m; ++k) pcoef[k] = c_old[k];
            double absint = integrate_abs_residual_over_interval(pcoef, alpha, beta, x0, x1);
            double avg = (h > 0.0) ? (absint / h) : 0.0;
            if (!std::isfinite(avg)) avg = eps;
            if (avg < 0.0) avg = 0.0;
            weights.push_back(avg);
            if (avg > max_avg) max_avg = avg;
        }
        // convert avg -> IRLS weights: w = 1 / max(eps, avg)
        std::vector<double> w_intervals;
        w_intervals.reserve(weights.size());
        double global_eps = std::max(1e-12, 1e-12 * std::max(1.0, max_avg));
        for (double a : weights) {
            double w = 1.0 / std::max(global_eps, a);
            if (!(w > 0.0)) w = weight_floor;
            if (w > weight_cap) w = weight_cap;
            w_intervals.push_back(w);
        }

        // build weighted ATA and ATb analytically
        Eigen::MatrixXd ATA(m, m);
        Eigen::VectorXd ATb(m);
        buildWeightedATA_ATb_piecewise_linear(x, y, w_intervals, degree, ATA, ATb);

        // small ridge
        double tr = ATA.trace();
        double ridge = 1e-14 * std::max(1.0, std::abs(tr));
        for (int i = 0; i < m; ++i) ATA(i, i) += ridge;

        // solve weighted normal equations
        Eigen::VectorXd c_vec = robustSymmetricSolve(ATA, ATb);
        if (!std::isfinite(c_vec.norm())) break;

        // check convergence
        double change_norm = 0.0;
        for (int k = 0; k < m; ++k) {
            double diff = c_vec(k) - c_old[k];
            change_norm += diff * diff;
        }
        change_norm = std::sqrt(change_norm);
        for (int k = 0; k < m; ++k) c_old[k] = c_vec(k);

        if (change_norm < tol) break;
    }

    std::vector<float> out(m);
    for (int i = 0; i < m; ++i) out[i] = static_cast<float>(c_old[i]);
    return out;
}
