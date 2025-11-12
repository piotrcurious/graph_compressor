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

    // Debug print macro for Arduino portability
    #if defined(ARDUINO)
    #include <Arduino.h>
    #define DEBUG_PRINT(...) do { if (Serial) Serial.printf(__VA_ARGS__); } while(0)
    #else
    #define DEBUG_PRINT(...) do { printf(__VA_ARGS__); } while(0)
    #endif

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

    // ---------- QP solver (primal-dual interior point) ----------
    auto qp_solve = [&](const Eigen::MatrixXd &ATA, const Eigen::VectorXd &ATb,
                        const std::vector<std::vector<double>>& Aeq_rows, const std::vector<double>& beq,
                        const std::vector<std::vector<double>>& Ain_rows, const std::vector<double>& bin) -> std::pair<bool, Eigen::VectorXd> {
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
            auto sol = qp_solve(ATA_alpha, ATb_alpha, C_eq, d_eq, Ain_rows, bin_rows);
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
        auto sol0 = qp_solve(ATA0, ATb0, C_eq_fb0, d_eq_fb0, {}, {}); // No inequalities
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
         auto sol1 = qp_solve(ATA0, ATb0, C_eq_fb1, d_eq_fb1, {}, {});
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
