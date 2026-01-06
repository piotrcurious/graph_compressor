// ----------------- Lebesgue-style L2 over piecewise-linear interpolant -----------------
// Build ATA and ATb by integrating over piecewise-linear y(x) on sorted x-grid.
// Uses dd accumulation to avoid catastrophic cancellation.
static inline double integral_pow(double a, double b, int n) {
    // ∫_a^b x^n dx = (b^{n+1} - a^{n+1}) / (n+1)
    // handle n == -1 not needed (we only use n>=0)
    double np1 = double(n + 1);
    // Use std::pow; we'll accumulate in dd for robustness.
    return (std::pow(b, np1) - std::pow(a, np1)) / np1;
}

static void buildATA_ATb_piecewise_linear(const std::vector<double>& x_in,
                                          const std::vector<float>& y_in,
                                          int degree,
                                          Eigen::MatrixXd &ATA_out,
                                          Eigen::VectorXd &ATb_out)
{
    size_t n = x_in.size();
    int m = degree + 1;
    ATA_out = Eigen::MatrixXd::Zero(m, m);
    ATb_out = Eigen::VectorXd::Zero(m);
    if (n == 0) return;

    // Need at least two points to define intervals. If only 1 point, no intervals -> fallback behavior should be handled by caller.
    // Sort indices by x to get domain partition
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return x_in[a] < x_in[b]; });

    // double-double accumulators: lower-triangular ATA_dd[j][k] for j>=k, and ATb_dd[j]
    std::vector<std::vector<dd>> ATA_dd(m, std::vector<dd>(m, dd(0.0)));
    std::vector<dd> ATb_dd(m, dd(0.0));

    // Walk intervals [x_i, x_{i+1}] and integrate
    for (size_t s = 0; s + 1 < n; ++s) {
        size_t i0 = idx[s];
        size_t i1 = idx[s+1];
        double x0 = x_in[i0];
        double x1 = x_in[i1];
        if (!(x1 > x0)) {
            // degenerate interval (equal x). Treat as tiny interval skip to avoid division by 0.
            continue;
        }
        double h = x1 - x0;
        double y0 = double(y_in[i0]);
        double y1 = double(y_in[i1]);
        // slope and intercept for y(x) = alpha + beta*x
        double beta = (y1 - y0) / h;            // slope
        double alpha = y0 - beta * x0;          // intercept

        // Precompute integrals I_n = ∫_{x0}^{x1} x^n dx for n up to 2*degree (used for ATA entries)
        int maxPow = 2 * degree;
        std::vector<double> I(maxPow + 1);
        for (int p = 0; p <= maxPow; ++p) I[p] = integral_pow(x0, x1, p);

        // Update ATA entries: ATA[r][c] += ∫ x^{r+c} dx
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c <= r; ++c) { // lower triangle
                int powIdx = r + c;
                double add = I[powIdx];
                ATA_dd[r][c] = dd_add_double(ATA_dd[r][c], add);
            }
        }
        // Update ATb entries: ATb[r] += ∫ x^r * (alpha + beta*x) dx = alpha * I[r] + beta * I[r+1]
        for (int r = 0; r < m; ++r) {
            double addb = alpha * I[r];
            if (r + 1 <= maxPow) addb += beta * I[r + 1];
            ATb_dd[r] = dd_add_double(ATb_dd[r], addb);
        }
    } // intervals

    // If there are interior gaps (x domain not fully covered?) we intentionally only integrate over convex hull where interpolant defined.
    // Convert dd accumulators into ATA_out (symmetric) and ATb_out
    for (int j = 0; j < m; ++j) {
        ATb_out(j) = dd_to_double(ATb_dd[j]);
        for (int k = 0; k <= j; ++k) {
            double v = dd_to_double(ATA_dd[j][k]);
            ATA_out(j, k) = v;
            ATA_out(k, j) = v;
        }
    }
}

// New public-facing Lebesgue L2 fitter (piecewise-linear interpolant)
std::vector<float> AdvancedPolynomialFitter::fitPolynomialLebesgueL2(const std::vector<double>& x,
                                                                     const std::vector<float>& y,
                                                                     int degree,
                                                                     double ridge /*= 1e-14*/)
{
    // If insufficient points, fallback to original discrete LS
    if (x.size() != y.size() || x.empty() || degree < 1) return {};

    if (x.size() < 2) {
        // fall back to per-point LS (discrete) to avoid division by zero in piecewise linear setup
        return fitPolynomialD(x, y, degree, NONE); // assumes NONE is a valid enum; else call existing fitPolynomialD
    }

    int m = degree + 1;
    Eigen::MatrixXd ATA(m, m);
    Eigen::VectorXd ATb(m);

    buildATA_ATb_piecewise_linear(x, y, degree, ATA, ATb);

    // small ridge to stabilize ill-conditioning (scale by trace)
    double tr = ATA.trace();
    double lambda = 1e-14 * std::max(1.0, std::abs(tr));
    for (int i = 0; i < m; ++i) ATA(i, i) += std::max(lambda, ridge);

    Eigen::VectorXd c = robustSymmetricSolve(ATA, ATb);
    if (!std::isfinite(c.norm())) {
        // fallback to discrete least-squares
        return fitPolynomialD(x, y, degree, NONE);
    }

    std::vector<float> out(m);
    for (int i = 0; i < m; ++i) out[i] = static_cast<float>(c(i));
    return out;
}
