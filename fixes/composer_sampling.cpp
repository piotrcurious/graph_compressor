// Constrained sample-based composition which preserves extrema (values and optional derivative=0)
std::vector<float> AdvancedPolynomialFitter::composePolynomials(const float* p1_coeffs,
                                                                double p1_delta,
                                                                const float* p2_coeffs,
                                                                double p2_delta,
                                                                int degree) {
    using ld = long double;
    int m = degree + 1;
    if (m <= 0) return {};
    double total_delta = p1_delta + p2_delta;
    if (total_delta <= 0.0) return std::vector<float>(m, 0.0f);

    // convert inputs to double
    std::vector<double> p1d(m, 0.0), p2d(m, 0.0);
    for (int i = 0; i < m; ++i) { p1d[i] = double(p1_coeffs[i]); p2d[i] = double(p2_coeffs[i]); }

    // weights and mapping
    double w1 = p1_delta / total_delta;
    double w2 = 1.0 - w1;
    if (w1 <= 0.0) return std::vector<float>(m, 0.0f);

    // ---------------------------
    // helper: evaluate polynomial in monomial basis at u (horner)
    auto eval_poly = [](const std::vector<double>& c, double u) {
        double r = 0.0;
        for (int k = (int)c.size() - 1; k >= 0; --k) r = r * u + c[k];
        return r;
    };

    // ---------------------------
    // helper: compute derivative polynomial coefficients (monomial basis)
    auto derivative_coeffs = [](const std::vector<double>& c) {
        int n = (int)c.size();
        if (n <= 1) return std::vector<double>{0.0};
        std::vector<double> d(n - 1);
        for (int k = 0; k < n - 1; ++k) d[k] = double(k + 1) * c[k + 1];
        return d;
    };

    // ---------------------------
    // helper: find real roots in [0,1] of polynomial (coeffs in increasing order)
    auto find_real_roots_unit = [&](const std::vector<double>& poly) {
        std::vector<double> roots_in_unit;
        // trim leading zeros
        int deg = (int)poly.size() - 1;
        while (deg > 0 && std::abs(poly[deg]) < 1e-18) --deg;
        if (deg <= 0) return roots_in_unit;
        // if degree == 1 simple root
        if (deg == 1) {
            double a = poly[1], b = poly[0];
            if (a != 0.0) {
                double r = -b / a;
                if (r >= 0.0 - 1e-12 && r <= 1.0 + 1e-12) roots_in_unit.push_back(std::min(1.0, std::max(0.0, r)));
            }
            return roots_in_unit;
        }
        // build monic polynomial coefficients for companion
        double lead = poly[deg];
        if (std::abs(lead) < 1e-30) return roots_in_unit;
        std::vector<double> b(deg);
        for (int k = 0; k < deg; ++k) b[k] = poly[k] / lead; // note: b[deg-1] is coeff of x^{deg-1} divided by lead
        // companion matrix size deg x deg
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(deg, deg);
        // first row: -b_{deg-1}, -b_{deg-2}, ..., -b_0
        for (int j = 0; j < deg; ++j) {
            C(0, j) = -b[deg - 1 - j];
        }
        for (int i = 1; i < deg; ++i) C(i, i - 1) = 1.0;
        Eigen::EigenSolver<Eigen::MatrixXd> es(C, /* computeEigenvectors = */ false);
        Eigen::VectorXcd eigs = es.eigenvalues();
        for (int i = 0; i < eigs.size(); ++i) {
            std::complex<double> z = eigs[i];
            if (std::abs(z.imag()) < 1e-9) {
                double r = z.real();
                if (r >= -1e-12 && r <= 1.0 + 1e-12) roots_in_unit.push_back(std::min(1.0, std::max(0.0, r)));
            }
        }
        // sort + unique (tolerance)
        std::sort(roots_in_unit.begin(), roots_in_unit.end());
        roots_in_unit.erase(std::unique(roots_in_unit.begin(), roots_in_unit.end(), [](double a, double b){ return std::abs(a-b) < 1e-12; }), roots_in_unit.end());
        return roots_in_unit;
    };

    // ---------------------------
    // find extrema (roots of derivative) for p1 and p2 on u in [0,1]
    std::vector<double> extrema_x; // mapped into x in [0,1]
    // p1
    {
        auto d1 = derivative_coeffs(p1d);
        auto r1 = find_real_roots_unit(d1);
        for (double u : r1) {
            // map u ∈ [0,1] -> x ∈ [0,w1] : x = w1 * u
            double x = w1 * u;
            if (x >= 0.0 && x <= 1.0) extrema_x.push_back(x);
        }
        // optionally include endpoints 0 and w1 (preserve boundary extrema)
        extrema_x.push_back(0.0);
        extrema_x.push_back(w1);
    }
    // p2
    if (w2 > 0.0) {
        auto d2 = derivative_coeffs(p2d);
        auto r2 = find_real_roots_unit(d2);
        for (double u : r2) {
            // map u ∈ [0,1] -> x ∈ [w1,1] : x = w1 + w2 * u
            double x = w1 + w2 * u;
            if (x >= 0.0 && x <= 1.0) extrema_x.push_back(x);
        }
        extrema_x.push_back(1.0);
        extrema_x.push_back(w1);
    }

    // unique sort extrema, clamp into (0,1)
    std::sort(extrema_x.begin(), extrema_x.end());
    extrema_x.erase(std::unique(extrema_x.begin(), extrema_x.end(), [](double a, double b){ return std::abs(a-b) < 1e-12; }), extrema_x.end());
    // clamp and drop values extremely close to each other
    std::vector<double> extrema_x_clean;
    for (double xv : extrema_x) {
        double xv2 = std::min(1.0, std::max(0.0, xv));
        if (extrema_x_clean.empty() || std::abs(xv2 - extrema_x_clean.back()) > 1e-12) extrema_x_clean.push_back(xv2);
    }
    extrema_x.swap(extrema_x_clean);

    // ---------------------------
    // sampling to form normal equations (dense sampling on [0,1])
    int N = std::max(1000, 50 * m); // large sample for fidelity (speed not important)
    // ATA and ATb as long double accumulators
    std::vector<std::vector<ld>> ATA_ld(m, std::vector<ld>(m, 0.0L));
    std::vector<ld> ATb_ld(m, 0.0L);

    for (int i = 0; i < N; ++i) {
        // center sampling within bin
        double x = double(i + 0.5) / double(N);
        double yv;
        if (x <= w1) {
            double u = (w1 == 0.0) ? 0.0 : (x / w1);
            yv = eval_poly(p1d, u);
        } else {
            double u = (w2 == 0.0) ? 0.0 : ((x - w1) / w2);
            yv = eval_poly(p2d, u);
        }
        // compute monomial powers at x
        std::vector<ld> xp(m);
        xp[0] = 1.0L;
        for (int k = 1; k < m; ++k) xp[k] = xp[k - 1] * (ld)x;

        // accumulate ATA and ATb
        for (int r = 0; r < m; ++r) {
            ATb_ld[r] += (ld)yv * xp[r];
            for (int c = 0; c <= r; ++c) {
                ATA_ld[r][c] += xp[r] * xp[c];
            }
        }
    }
    // symmetrize ATA
    for (int r = 0; r < m; ++r) for (int c = 0; c < r; ++c) ATA_ld[c][r] = ATA_ld[r][c];

    // convert ATA and ATb to Eigen double
    Eigen::MatrixXd ATA(m, m);
    Eigen::VectorXd ATb(m);
    for (int r = 0; r < m; ++r) {
        ATb(r) = double(ATb_ld[r]);
        for (int c = 0; c < m; ++c) ATA(r, c) = double(ATA_ld[r][c]);
    }

    // ---------------------------
    // build constraints C * c = d
    // Each value constraint: sum_k c_k * x^k == f(x)
    // Optionally, derivative constraint: sum_k k*c_k * x^{k-1} == 0

    std::vector<std::vector<double>> Crows;
    std::vector<double> dvals;

    // function to evaluate target piecewise f(x)
    auto f_target = [&](double x)->double {
        if (x <= w1) {
            double u = (w1 == 0.0) ? 0.0 : (x / w1);
            return eval_poly(p1d, u);
        } else {
            double u = (w2 == 0.0) ? 0.0 : ((x - w1) / w2);
            return eval_poly(p2d, u);
        }
    };

    // We'll add constraints but must ensure total constraints < m (otherwise system may be overconstrained).
    // Strategy: try adding value+derivative per extremum, but stop when constraints >= m.
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
            // derivative: sum_{k=1..m-1} k*c_k * x^{k-1} == 0
            double xpow = 1.0;
            for (int k = 1; k < m; ++k) {
                crowd[k] = double(k) * xpow;
                xpow *= x;
            }
            Crows.push_back(crowd);
            dvals.push_back(0.0);
        }
    }

    // If no constraints were added, do the unconstrained solve (regular least-squares normal equations)
    if (Crows.empty()) {
        // small ridge for stability
        double ridge = 1e-14 * ATA.trace();
        if (std::isfinite(ridge) && ridge > 0.0) for (int i = 0; i < m; ++i) ATA(i,i) += ridge;
        Eigen::VectorXd coeffs = robustSymmetricSolve(ATA, ATb);
        std::vector<float> out(m);
        for (int i = 0; i < m; ++i) out[i] = static_cast<float>(coeffs(i));
        return out;
    }

    int rcount = (int)Crows.size();
    // build augmented matrix:
    // [ ATA  C^T ]
    // [ C    0  ]
    Eigen::MatrixXd Aug(m + rcount, m + rcount);
    Aug.setZero();
    // top-left
    for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) Aug(i,j) = ATA(i,j);
    // top-right = C^T
    for (int i = 0; i < rcount; ++i) for (int j = 0; j < m; ++j) Aug(j, m + i) = Crows[i][j];
    // bottom-left = C
    for (int i = 0; i < rcount; ++i) for (int j = 0; j < m; ++j) Aug(m + i, j) = Crows[i][j];
    // bottom-right zeros

    // RHS
    Eigen::VectorXd RHS(m + rcount);
    for (int i = 0; i < m; ++i) RHS(i) = ATb(i);
    for (int i = 0; i < rcount; ++i) RHS(m + i) = dvals[i];

    // Solve augmented system (dense). Use ColPivHouseholderQr (robust for rectangular/indefinite)
    Eigen::VectorXd sol = Aug.colPivHouseholderQr().solve(RHS);
    if (!std::isfinite(sol.norm())) {
        // fallback: unconstrained robust solve
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
