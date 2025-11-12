// Exact-integral constrained composition preserving extrema (value + optional derivative=0)
std::vector<float> AdvancedPolynomialFitter::composePolynomials(const float* p1_coeffs,
                                                                double p1_delta,
                                                                const float* p2_coeffs,
                                                                double p2_delta,
                                                                int degree) {
    int m = degree + 1;
    if (m <= 0) return {};
    double total_delta = p1_delta + p2_delta;
    if (total_delta <= 0.0) return std::vector<float>(m, 0.0f);

    // weights
    double w1 = p1_delta / total_delta;
    double w2 = 1.0 - w1;

    // convert to doubles
    std::vector<double> p1d(m, 0.0), p2d(m, 0.0);
    for (int i = 0; i < m; ++i) { p1d[i] = double(p1_coeffs[i]); p2d[i] = double(p2_coeffs[i]); }

    // small helper: multiplicative binomial comb to avoid big precomputed table
    auto binom = [](int n, int k)->double {
        if (k < 0 || k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;
        if (k > n/2) k = n - k;
        double res = 1.0;
        for (int i = 1; i <= k; ++i) res = res * double(n - i + 1) / double(i);
        return res;
    };

    // ----------------------------
    // Build ATA analytically: ATA[r,c] = 1/(r+c+1)
    Eigen::MatrixXd ATA(m, m);
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < m; ++c) {
            ATA(r, c) = 1.0 / double(r + c + 1);
        }
    }

    // ----------------------------
    // Build ATb by exact integrals:
    // ATb_r = integral_0^{w1} p1(x/w1) x^r dx  + integral_{w1}^{1} p2((x-w1)/w2) x^r dx
    // The first term simplifies to: w1^{r+1} * sum_k p1_k / (k + r + 1)
    // The second term uses binomial expansion:
    // w2 * sum_k p2_k * sum_{t=0..r} binom(r,t) w1^{r-t} w2^{t} / (k + t + 1)

    std::vector<dd> ATb_dd(m); // dd accumulators
    for (int r = 0; r < m; ++r) {
        // part 1: from p1 on [0,w1]
        if (w1 > 0.0) {
            // factor w1^{r+1}
            double factor = std::pow(w1, double(r + 1));
            for (int k = 0; k <= degree; ++k) {
                double denom = double(k + r + 1);
                double contrib = p1d[k] / denom;
                // add factor * contrib
                ATb_dd[r] = dd_add_double(ATb_dd[r], factor * contrib);
            }
        }
        // part 2: from p2 on [w1,1]
        if (w2 > 0.0) {
            // precompute powers w1^{r-t} on the fly, and w2^{t}
            // We'll compute for t=0..r
            for (int k = 0; k <= degree; ++k) {
                double p2k = p2d[k];
                // sum over t
                double inner_sum = 0.0;
                for (int t = 0; t <= r; ++t) {
                    double b = binom(r, t);
                    // w1^{r-t} * w2^{t}
                    double w1pow = (r - t == 0) ? 1.0 : std::pow(w1, double(r - t));
                    double w2pow = (t == 0) ? 1.0 : std::pow(w2, double(t));
                    double denom = double(k + t + 1);
                    inner_sum += b * w1pow * w2pow / denom;
                }
                // multiply by p2k and by outer w2
                double contrib = p2k * inner_sum * w2;
                ATb_dd[r] = dd_add_double(ATb_dd[r], contrib);
            }
        }
    }

    // convert ATb_dd to Eigen vector of doubles
    Eigen::VectorXd ATb(m);
    for (int r = 0; r < m; ++r) ATb(r) = dd_to_double(ATb_dd[r]);

    // small ridge regularization proportional to trace for stability
    double trace = ATA.trace();
    double ridge = 1e-14 * std::max(1.0, std::abs(trace));
    for (int i = 0; i < m; ++i) ATA(i,i) += ridge;

    // ----------------------------
    // Find extrema (roots of derivatives) for p1 and p2 in their local u∈[0,1] domains
    auto derivative_coeffs = [](const std::vector<double>& c) {
        int n = (int)c.size();
        if (n <= 1) return std::vector<double>{0.0};
        std::vector<double> d(n - 1);
        for (int k = 0; k < n - 1; ++k) d[k] = double(k + 1) * c[k + 1];
        return d;
    };

    auto find_real_roots_unit = [&](const std::vector<double>& poly){
        std::vector<double> roots_in_unit;
        int deg = (int)poly.size() - 1;
        while (deg > 0 && std::abs(poly[deg]) < 1e-18) --deg;
        if (deg <= 0) return roots_in_unit;
        if (deg == 1) {
            double a = poly[1], b = poly[0];
            if (a != 0.0) {
                double r = -b / a;
                if (r >= -1e-12 && r <= 1.0 + 1e-12) roots_in_unit.push_back(std::min(1.0, std::max(0.0, r)));
            }
            return roots_in_unit;
        }
        double lead = poly[deg];
        if (std::abs(lead) < 1e-30) return roots_in_unit;
        std::vector<double> bcoef(deg);
        for (int k = 0; k < deg; ++k) bcoef[k] = poly[k] / lead;
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(deg, deg);
        for (int j = 0; j < deg; ++j) C(0, j) = -bcoef[deg - 1 - j];
        for (int i = 1; i < deg; ++i) C(i, i - 1) = 1.0;
        Eigen::EigenSolver<Eigen::MatrixXd> es(C, false);
        Eigen::VectorXcd eigs = es.eigenvalues();
        for (int i = 0; i < eigs.size(); ++i) {
            std::complex<double> z = eigs[i];
            if (std::abs(z.imag()) < 1e-9) {
                double r = z.real();
                if (r >= -1e-12 && r <= 1.0 + 1e-12) roots_in_unit.push_back(std::min(1.0, std::max(0.0, r)));
            }
        }
        std::sort(roots_in_unit.begin(), roots_in_unit.end());
        roots_in_unit.erase(std::unique(roots_in_unit.begin(), roots_in_unit.end(), [](double a,double b){ return std::abs(a-b) < 1e-12; }), roots_in_unit.end());
        return roots_in_unit;
    };

    std::vector<double> extrema_x;
    // p1
    {
        auto d1 = derivative_coeffs(p1d);
        auto r1 = find_real_roots_unit(d1);
        for (double u : r1) {
            double x = w1 * u; // map to global x in [0,1]
            if (x >= 0.0 && x <= 1.0) extrema_x.push_back(x);
        }
        // endpoints
        extrema_x.push_back(0.0);
        extrema_x.push_back(w1);
    }
    // p2
    if (w2 > 0.0) {
        auto d2 = derivative_coeffs(p2d);
        auto r2 = find_real_roots_unit(d2);
        for (double u : r2) {
            double x = w1 + w2 * u;
            if (x >= 0.0 && x <= 1.0) extrema_x.push_back(x);
        }
        extrema_x.push_back(w1);
        extrema_x.push_back(1.0);
    }

    // unique & clean
    std::sort(extrema_x.begin(), extrema_x.end());
    extrema_x.erase(std::unique(extrema_x.begin(), extrema_x.end(), [](double a,double b){ return std::abs(a-b) < 1e-12; }), extrema_x.end());
    std::vector<double> extrema_x_clean;
    for (double xv : extrema_x) {
        double xv2 = std::min(1.0, std::max(0.0, xv));
        if (extrema_x_clean.empty() || std::abs(xv2 - extrema_x_clean.back()) > 1e-12) extrema_x_clean.push_back(xv2);
    }
    extrema_x.swap(extrema_x_clean);

    // ----------------------------
    // Build constraints C * c = d (value constraints and derivative=0) until we have < m constraints
    std::vector<std::vector<double>> Crows;
    std::vector<double> dvals;

    auto eval_poly = [](const std::vector<double>& c, double u)->double{
        double r = 0.0;
        for (int k = (int)c.size()-1; k >= 0; --k) r = r * u + c[k];
        return r;
    };

    auto f_target = [&](double x)->double {
        if (x <= w1) {
            double u = (w1 == 0.0) ? 0.0 : (x / w1);
            return eval_poly(p1d, u);
        } else {
            double u = (w2 == 0.0) ? 0.0 : ((x - w1) / w2);
            return eval_poly(p2d, u);
        }
    };

    for (double x : extrema_x) {
        if ((int)Crows.size() >= m) break;
        // value constraint
        std::vector<double> crow(m);
        double xp = 1.0;
        for (int k = 0; k < m; ++k) { crow[k] = xp; xp *= x; }
        Crows.push_back(crow);
        dvals.push_back(f_target(x));
        if ((int)Crows.size() >= m) break;
        // derivative constraint (if degree allows)
        if (m >= 2 && (int)Crows.size() < m) {
            std::vector<double> crowd(m, 0.0);
            double xpow = 1.0;
            for (int k = 1; k < m; ++k) {
                crowd[k] = double(k) * xpow;
                xpow *= x;
            }
            Crows.push_back(crowd);
            dvals.push_back(0.0);
        }
    }

    // If no constraints, simple unconstrained solve
    if (Crows.empty()) {
        Eigen::VectorXd coeffs = robustSymmetricSolve(ATA, ATb);
        std::vector<float> out(m);
        for (int i = 0; i < m; ++i) out[i] = static_cast<float>(coeffs(i));
        return out;
    }

    int rcount = (int)Crows.size();

    // Build augmented system:
    // [ ATA   C^T ] [ c ] = [ ATb ]
    // [ C     0  ] [ λ ]   [  d  ]
    Eigen::MatrixXd Aug(m + rcount, m + rcount);
    Aug.setZero();

    for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) Aug(i, j) = ATA(i, j);
    for (int i = 0; i < rcount; ++i) for (int j = 0; j < m; ++j) {
        Aug(j, m + i) = Crows[i][j];
        Aug(m + i, j) = Crows[i][j];
    }

    Eigen::VectorXd RHS(m + rcount);
    for (int i = 0; i < m; ++i) RHS(i) = ATb(i);
    for (int i = 0; i < rcount; ++i) RHS(m + i) = dvals[i];

    // Solve dense augmented system with QR. If it fails, fall back to unconstrained robust solve.
    Eigen::VectorXd sol = Aug.colPivHouseholderQr().solve(RHS);
    if (!std::isfinite(sol.norm())) {
        Eigen::VectorXd coeffs = robustSymmetricSolve(ATA, ATb);
        std::vector<float> out(m);
        for (int i = 0; i < m; ++i) out[i] = static_cast<float>(coeffs(i));
        return out;
    }

    Eigen::VectorXd coeffs = sol.head(m);
    std::vector<float> out(m);
    for (int i = 0; i < m; ++i) out[i] = static_cast<float>(coeffs(i));
    return out;
}
