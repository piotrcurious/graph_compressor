// ----------------- Analytical projection onto shifted-Legendre basis (Lebesgue L2, no linear solver) -----------------

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
    int degree)
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
