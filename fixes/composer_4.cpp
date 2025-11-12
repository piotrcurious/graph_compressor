// Advanced composePolynomials:
// - preserves endpoint extrema (values, derivative if stationary)
// - enforces C0/C1 blending at shared boundary (target = weighted avg of two sides)
// - trims weak extrema automatically and searches candidate fits (alpha weighting vs boundary + different kept extrema counts)
// - computes ATA/ATb by exact integrals (no sampling), uses dd accumulation for robustness
// - uses active-set style inequalities to guarantee min/max (second-derivative sign)
// - returns best candidate by exact L2 error
std::vector<float> AdvancedPolynomialFitter::composePolynomials(const float* p1_coeffs,
                                                                double p1_delta,
                                                                const float* p2_coeffs,
                                                                double p2_delta,
                                                                int degree) {
    using std::abs;
    int m = degree + 1;
    if (m <= 0) return {};
    double total_delta = p1_delta + p2_delta;
    if (total_delta <= 0.0) return std::vector<float>(m, 0.0f);

    // normalized boundary split
    double w1 = p1_delta / total_delta;
    double w2 = 1.0 - w1;

    // convert inputs to double
    std::vector<double> p1d(m, 0.0), p2d(m, 0.0);
    for (int i = 0; i < m; ++i) { p1d[i] = double(p1_coeffs[i]); p2d[i] = double(p2_coeffs[i]); }

    // small combinatorial helper
    auto comb = [](int n, int k)->double {
        if (k < 0 || k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;
        if (k > n/2) k = n - k;
        double r = 1.0;
        for (int i = 1; i <= k; ++i) r = r * double(n - i + 1) / double(i);
        return r;
    };

    // ------------------------------
    // analytic ATA with optional polynomial weighting around boundary:
    // weight(x) = 1 + alpha*(x - w1)^2  (polynomial weight -> closed-form integrals)
    auto build_weighted_ATA_ATb = [&](double alpha, Eigen::MatrixXd &ATA_out, Eigen::VectorXd &ATb_out, std::vector<dd> &ATb_dd_out) {
        // ATA[r,c] = ∫_0^1 x^{r+c} * weight(x) dx
        // weight(x) = 1 + alpha*(x^2 - 2*w1*x + w1^2)
        ATA_out = Eigen::MatrixXd(m, m);
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) {
                int n = r + c;
                double I0 = 1.0 / double(n + 1);               // ∫ x^n
                double I1 = 1.0 / double(n + 3);               // ∫ x^{n+2}
                double I2 = 1.0 / double(n + 2);               // ∫ x^{n+1}
                // ∫ x^n * (x^2 - 2*w1*x + w1^2) dx = I1 - 2*w1*I2 + w1*w1*I0
                double weighted = I0 + alpha * (I1 - 2.0 * w1 * I2 + w1 * w1 * I0);
                ATA_out(r, c) = weighted;
            }
        }

        // ATb[r] = ∫_0^1 f(x) * x^r * weight(x) dx
        // compute exactly as sum of p1 on [0,w1] and p2 on [w1,1]
        ATb_dd_out.assign(m, dd(0.0));
        // part 1: x = w1 * u, dx = w1 du, weight(w1*u) = 1 + alpha*w1^2*(u^2 - 2u + 1)
        if (w1 > 0.0) {
            for (int r = 0; r < m; ++r) {
                double factor = std::pow(w1, double(r + 1)); // outside factor
                for (int k = 0; k <= degree; ++k) {
                    // integrals: ∫_0^1 u^{k+r} du = 1/(k+r+1)
                    double I0 = 1.0 / double(k + r + 1);
                    // ∫ u^{k+r+2} = 1/(k+r+3)
                    double I2 = 1.0 / double(k + r + 3);
                    // ∫ u^{k+r+1} = 1/(k+r+2)
                    double I1 = 1.0 / double(k + r + 2);
                    // weight factor inside: 1 + alpha*w1^2*(u^2 - 2u + 1)
                    double term = p1d[k] * (I0 + alpha * (w1 * w1) * (I2 - 2.0 * I1 + I0));
                    ATb_dd_out[r] = dd_add_double(ATb_dd_out[r], factor * term);
                }
            }
        }
        // part 2: x = w1 + w2*u, dx = w2 du.
        // weight(w1 + w2*u) = 1 + alpha*(w2^2 * u^2)
        if (w2 > 0.0) {
            for (int r = 0; r < m; ++r) {
                for (int k = 0; k <= degree; ++k) {
                    // inner sum t=0..r: binom(r,t) * w1^{r-t} * w2^{t} * ∫ u^{k+t} du (and for weighted extra u^{k+t+2})
                    double inner = 0.0;
                    double inner_weighted = 0.0;
                    for (int t = 0; t <= r; ++t) {
                        double b = comb(r, t);
                        double w1pow = (r - t == 0) ? 1.0 : std::pow(w1, double(r - t));
                        double w2pow = (t == 0) ? 1.0 : std::pow(w2, double(t));
                        double Ibase = 1.0 / double(k + t + 1);           // ∫ u^{k+t}
                        double Iplus2 = 1.0 / double(k + t + 3);          // ∫ u^{k+t+2}
                        inner += b * w1pow * w2pow * Ibase;
                        inner_weighted += b * w1pow * w2pow * Iplus2; // weighted uses u^{k+t+2} for u^2 term
                    }
                    // multiply by p2k and outer factor w2; weight contributes alpha*w2^2*inner_weighted
                    double term = p2d[k] * (inner + alpha * (w2 * w2) * inner_weighted);
                    ATb_dd_out[r] = dd_add_double(ATb_dd_out[r], term * w2);
                }
            }
        }

        // convert dd to ATb_out
        ATb_out = Eigen::VectorXd(m);
        for (int r = 0; r < m; ++r) ATb_out(r) = dd_to_double(ATb_dd_out[r]);
    };

    // ------------------------------
    // exact ∫ f^2 dx constant for L2 error comparison:
    auto compute_F2 = [&]() -> double {
        // F2 = ∫_0^{w1} p1(u)^2 * w1 du  + ∫_0^{w2} p2(u)^2 * w2 du (u in [0,1] for both)
        dd acc(0.0);
        if (w1 > 0.0) {
            for (int i = 0; i <= degree; ++i) for (int j = 0; j <= degree; ++j) {
                double denom = 1.0 / double(i + j + 1);
                double contrib = p1d[i] * p1d[j] * denom * w1;
                acc = dd_add_double(acc, contrib);
            }
        }
        if (w2 > 0.0) {
            for (int i = 0; i <= degree; ++i) for (int j = 0; j <= degree; ++j) {
                double denom = 1.0 / double(i + j + 1);
                double contrib = p2d[i] * p2d[j] * denom * w2;
                acc = dd_add_double(acc, contrib);
            }
        }
        return dd_to_double(acc);
    };

    double F2_const = compute_F2();

    // ------------------------------
    // helpers: evaluate p and derivatives in global x
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

    auto eval_f_global = [&](double x)->double {
        if (x <= w1) {
            double u = (w1 == 0.0) ? 0.0 : (x / w1);
            return eval_poly(p1d, u);
        } else {
            double u = (w2 == 0.0) ? 0.0 : ((x - w1) / w2);
            return eval_poly(p2d, u);
        }
    };
    auto eval_fprime_global = [&](double x)->double {
        if (x <= w1) {
            double scale = (w1 == 0.0) ? 1.0 : w1;
            double u = (scale == 0.0) ? 0.0 : (x / scale);
            auto d1 = deriv_coeffs(p1d);
            double val = eval_poly(d1, u);
            return val / scale;
        } else {
            double scale = (w2 == 0.0) ? 1.0 : w2;
            double u = (scale == 0.0) ? 0.0 : ((x - w1) / scale);
            auto d2 = deriv_coeffs(p2d);
            double val = eval_poly(d2, u);
            return val / scale;
        }
    };
    auto eval_fsecond_global = [&](double x)->double {
        if (x <= w1) {
            double scale = (w1 == 0.0) ? 1.0 : w1;
            double u = (scale == 0.0) ? 0.0 : (x / scale);
            auto s2 = second_deriv_coeffs(p1d);
            double val = eval_poly(s2, u);
            return val / (scale * scale);
        } else {
            double scale = (w2 == 0.0) ? 1.0 : w2;
            double u = (scale == 0.0) ? 0.0 : ((x - w1) / scale);
            auto s2 = second_deriv_coeffs(p2d);
            double val = eval_poly(s2, u);
            return val / (scale * scale);
        }
    };

    // ------------------------------
    // detect candidate extrema (roots of derivative), include endpoints, then score & trim weak ones
    auto find_real_roots_unit = [&](const std::vector<double>& poly)->std::vector<double> {
        std::vector<double> roots;
        int deg = (int)poly.size() - 1;
        while (deg > 0 && std::abs(poly[deg]) < 1e-18) --deg;
        if (deg <= 0) return roots;
        if (deg == 1) {
            double a = poly[1], b = poly[0];
            if (a != 0.0) {
                double r = -b / a;
                if (r >= -1e-12 && r <= 1.0 + 1e-12) roots.push_back(std::min(1.0, std::max(0.0, r)));
            }
            return roots;
        }
        double lead = poly[deg];
        if (std::abs(lead) < 1e-30) return roots;
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
                if (r >= -1e-12 && r <= 1.0 + 1e-12) roots.push_back(std::min(1.0, std::max(0.0, r)));
            }
        }
        std::sort(roots.begin(), roots.end());
        roots.erase(std::unique(roots.begin(), roots.end(), [](double a,double b){ return std::abs(a-b) < 1e-12; }), roots.end());
        return roots;
    };

    std::vector<double> extrema_x;
    // p1 interior extrema
    {
        auto d1 = deriv_coeffs(p1d);
        auto r1 = find_real_roots_unit(d1);
        for (double u : r1) {
            double x = w1 * u;
            if (x >= 0.0 && x <= 1.0) extrema_x.push_back(x);
        }
        // endpoints: include left endpoints
        extrema_x.push_back(0.0);
        extrema_x.push_back(w1);
    }
    // p2 interior extrema
    if (w2 > 0.0) {
        auto d2 = deriv_coeffs(p2d);
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

    // scoring: combine |f''(x)| and "local prominence" measured in u-space
    struct Ext { double x; double score; };
    std::vector<Ext> scored;
    const double hu = 1e-4;
    for (double x : extrema_x) {
        double fpp = std::abs(eval_fsecond_global(x));
        // prominence estimate
        bool from_p1 = (x <= w1 + 1e-15);
        double prom = 0.0;
        if (from_p1) {
            double scale = (w1==0.0)?1.0:w1;
            double u = (scale == 0.0) ? 0.0 : (x / scale);
            double umin = std::max(0.0, u - hu), umax = std::min(1.0, u + hu);
            prom = std::min(std::abs(eval_poly(p1d, u) - eval_poly(p1d, umin)), std::abs(eval_poly(p1d, u) - eval_poly(p1d, umax)));
        } else {
            double scale = (w2==0.0)?1.0:w2;
            double u = (scale == 0.0) ? 0.0 : ((x - w1) / scale);
            double umin = std::max(0.0, u - hu), umax = std::min(1.0, u + hu);
            prom = std::min(std::abs(eval_poly(p2d, u) - eval_poly(p2d, umin)), std::abs(eval_poly(p2d, u) - eval_poly(p2d, umax)));
        }
        double score = fpp + prom;
        scored.push_back({x, score});
    }
    // sort descending
    std::sort(scored.begin(), scored.end(), [](const Ext &a, const Ext &b){ return a.score > b.score; });

    // candidate search parameters (adaptive search instead of single hand-tuned)
    // generate a sequence of alpha (weighting around boundary) values algorithmically:
    std::vector<double> alphas;
    alphas.push_back(0.0); // baseline
    // create multiplicative series around 1.0 to explore locality
    double base = 1.0;
    for (int i = -2; i <= 2; ++i) alphas.push_back(base * std::pow(10.0, i)); // 0.01,0.1,1,10,100
    // make unique and sort
    std::sort(alphas.begin(), alphas.end());
    alphas.erase(std::unique(alphas.begin(), alphas.end()), alphas.end());

    // candidate extrema keep counts: from 0 up to floor(m/2)
    int max_keep = std::max(0, m/2);
    std::vector<int> keep_counts;
    for (int k = 0; k <= max_keep; ++k) keep_counts.push_back(k);

    // Always preserve endpoints: x=0 and x=1 values
    double f0 = eval_f_global(0.0);
    double f1 = eval_f_global(1.0);
    // derivative at endpoints: if original derivative ~0 we preserve derivative equality
    double fprime0 = eval_fprime_global(0.0);
    double fprime1 = eval_fprime_global(1.0);
    bool preserve_d0 = (std::abs(fprime0) < 1e-12);
    bool preserve_d1 = (std::abs(fprime1) < 1e-12);

    // boundary C0/C1 target: use weighted average of left and right side to create smooth join.
    // left value: p1(u=1) if w1>0, right value: p2(u=0) if w2>0
    double left_val = (w1 > 0.0) ? eval_poly(p1d, 1.0) : eval_f_global(w1);
    double right_val = (w2 > 0.0) ? eval_poly(p2d, 0.0) : eval_f_global(w1);
    double boundary_val_target = w1 * left_val + w2 * right_val; // weighted average
    // derivative targets (global x): convert p1'_u(1) / w1 and p2'_u(0) / w2
    double left_d = 0.0, right_d = 0.0;
    if (w1 > 0.0) {
        auto d1 = deriv_coeffs(p1d);
        left_d = eval_poly(d1, 1.0) / w1;
    } else left_d = eval_fprime_global(w1);
    if (w2 > 0.0) {
        auto d2 = deriv_coeffs(p2d);
        right_d = eval_poly(d2, 0.0) / w2;
    } else right_d = eval_fprime_global(w1);
    double boundary_d_target = w1 * left_d + w2 * right_d;

    // convenience: build equality constraint rows for value at x
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

    // active-set QP utilities (equality constraints then inequalities as in previous code)
    auto solve_equality_constrained = [&](const Eigen::MatrixXd &ATA, const Eigen::VectorXd &ATb,
                                         const std::vector<std::vector<double>>& Aeq,
                                         const std::vector<double>& beq) -> std::pair<bool, Eigen::VectorXd> {
        int neq = (int)Aeq.size();
        int K = m + neq;
        Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(K, K);
        // top-left ATA
        for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) KKT(i,j) = ATA(i,j);
        // top-right and bottom-left Aeq^T / Aeq
        for (int ei = 0; ei < neq; ++ei) for (int j = 0; j < m; ++j) {
            KKT(j, m + ei) = Aeq[ei][j];
            KKT(m + ei, j) = Aeq[ei][j];
        }
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(K);
        for (int i = 0; i < m; ++i) rhs(i) = ATb(i);
        for (int ei = 0; ei < neq; ++ei) rhs(m + ei) = beq[ei];
        Eigen::VectorXd sol = KKT.colPivHouseholderQr().solve(rhs);
        if (!std::isfinite(sol.norm())) return {false, Eigen::VectorXd()};
        return {true, sol.head(m)};
    };

    // evaluate exact L2 error: Err = c^T ATA c - 2 ATb^T c + F2_const
    auto compute_error_from_coeffs = [&](const Eigen::MatrixXd &ATA, const Eigen::VectorXd &ATb, const Eigen::VectorXd &c)->double {
        double t1 = double(c.transpose() * ATA * c);
        double t2 = double(2.0 * ATb.dot(c));
        double err = t1 - t2 + F2_const;
        if (err < 0.0 && err > -1e-14) err = 0.0;
        return err;
    };

    // inequality active-set: second derivative sign constraints (min: f''(x) >= 0 , max: f''(x) <= 0)
    struct Inequality { std::vector<double> row; double bound; bool want_positive; double x; };
    auto build_second_deriv_row = [&](double x) {
        std::vector<double> row(m, 0.0);
        for (int k = 2; k < m; ++k) row[k] = double(k) * double(k - 1) * std::pow(x, double(k - 2));
        return row;
    };

    // ------------------------------
    // Candidate search loop: try alphas and different number of kept extrema
    double best_err = std::numeric_limits<double>::infinity();
    std::vector<float> best_coeffs(m, 0.0f);

    // Precompute candidate extrema list excluding endpoints (we always preserve endpoints)
    std::vector<double> interior_extrema;
    for (auto &e: scored) {
        if (e.x <= 1e-15 || e.x >= 1.0 - 1e-15) continue; // skip endpoints
        // avoid duplicating the boundary location w1
        if (std::abs(e.x - w1) < 1e-14) continue;
        interior_extrema.push_back(e.x);
    }

    // Ensure boundary constraints included in every candidate (value and derivative)
    // value at w1 target = boundary_val_target, derivative target = boundary_d_target
    std::vector<double> endpoints = {0.0, 1.0};
    std::vector<double> endpoint_values = {f0, f1};

    // We'll try different keep counts (from 0 to max_keep) and alphas
    for (double alpha : alphas) {
        // build ATA/ATb for this alpha
        Eigen::MatrixXd ATA_alpha;
        Eigen::VectorXd ATb_alpha;
        std::vector<dd> ATb_dd_tmp;
        build_weighted_ATA_ATb(alpha, ATA_alpha, ATb_alpha, ATb_dd_tmp);

        // small ridge
        double trace = ATA_alpha.trace();
        double ridge = 1e-14 * std::max(1.0, std::abs(trace));
        for (int i = 0; i < m; ++i) ATA_alpha(i,i) += ridge;

        for (int keepCount : keep_counts) {
            // build equality constraints: endpoints value enforced, endpoints derivative enforced if stationary
            std::vector<std::vector<double>> C_eq;
            std::vector<double> d_eq;
            // endpoints values
            for (int ei = 0; ei < 2; ++ei) {
                C_eq.push_back(build_value_row(endpoints[ei]));
                d_eq.push_back(endpoint_values[ei]);
            }
            // endpoints derivative if stationary
            if (preserve_d0) { C_eq.push_back(build_deriv_row(0.0)); d_eq.push_back(0.0); }
            if (preserve_d1) { C_eq.push_back(build_deriv_row(1.0)); d_eq.push_back(0.0); }

            // boundary C0 and C1 (always include, for better blending)
            C_eq.push_back(build_value_row(w1));
            d_eq.push_back(boundary_val_target);
            // derivative at boundary always enforced (smooth join)
            C_eq.push_back(build_deriv_row(w1));
            d_eq.push_back(boundary_d_target);

            // add top keepCount interior extrema (value + derivative) trimmed by score
            for (int k = 0; k < keepCount && k < (int)interior_extrema.size(); ++k) {
                double x = interior_extrema[k];
                // value
                C_eq.push_back(build_value_row(x));
                d_eq.push_back(eval_f_global(x));
                // derivative (stationary)
                if (m >= 2) {
                    C_eq.push_back(build_deriv_row(x));
                    d_eq.push_back(0.0);
                }
            }

            // ensure we haven't overconstrained: if #equalities >= m, drop lowest-priority extras
            if ((int)C_eq.size() >= m) {
                // keep only first m-1 equalities to allow some slack; prefer preserving endpoints and boundary which are first
                C_eq.resize(std::max(0, m - 1));
                d_eq.resize((int)C_eq.size());
            }

            // solve equality-constrained QP (exact ATA_alpha & ATb_alpha)
            auto eq_res = solve_equality_constrained(ATA_alpha, ATb_alpha, C_eq, d_eq);
            Eigen::VectorXd c;
            if (!eq_res.first) {
                // fallback unconstrained
                c = robustSymmetricSolve(ATA_alpha, ATb_alpha);
            } else {
                c = eq_res.second;
            }

            // Build inequality constraints for kept extrema: second derivative sign must match original
            std::vector<Inequality> ineqs;
            // boundary second derivative sign from original
            double b_fpp = eval_fsecond_global(w1);
            {
                Inequality in;
                in.row = build_second_deriv_row(w1);
                in.bound = 0.0;
                in.want_positive = (b_fpp >= 0.0);
                in.x = w1;
                ineqs.push_back(in);
            }
            // for each kept interior extremum determine sign by original second derivative
            for (int k = 0; k < keepCount && k < (int)interior_extrema.size(); ++k) {
                double x = interior_extrema[k];
                double orig_fpp = eval_fsecond_global(x);
                Inequality in;
                in.row = build_second_deriv_row(x);
                in.bound = 0.0;
                in.want_positive = (orig_fpp >= 0.0);
                in.x = x;
                ineqs.push_back(in);
            }

            // Active-set: iteratively add most violated inequality as equality (can't add more equalities than m-1)
            for (int iter = 0; iter < m && !ineqs.empty(); ++iter) {
                // check violations
                double worst_val = 0.0;
                int worst_idx = -1;
                for (int ii = 0; ii < (int)ineqs.size(); ++ii) {
                    const auto &inq = ineqs[ii];
                    double lhs = 0.0;
                    for (int k = 0; k < m; ++k) lhs += inq.row[k] * double(c(k));
                    double viol = 0.0;
                    if (inq.want_positive) {
                        if (lhs < inq.bound - 1e-12) viol = inq.bound - lhs; // positive amount
                    } else {
                        if (lhs > inq.bound + 1e-12) viol = lhs - inq.bound;
                    }
                    if (viol > worst_val) { worst_val = viol; worst_idx = ii; }
                }
                if (worst_idx == -1) break; // all satisfied

                // promote worst inequality to equality
                C_eq.push_back(ineqs[worst_idx].row);
                d_eq.push_back(ineqs[worst_idx].bound);
                // remove promoted inequality
                ineqs.erase(ineqs.begin() + worst_idx);

                // if equalities too many, stop promoting
                if ((int)C_eq.size() >= m) break;

                // resolve equality-constrained problem
                auto eq_res2 = solve_equality_constrained(ATA_alpha, ATb_alpha, C_eq, d_eq);
                if (!eq_res2.first) break; // give up promotions
                c = eq_res2.second;
            }

            // compute exact error for this candidate
            double err = compute_error_from_coeffs(ATA_alpha, ATb_alpha, c);

            // keep best
            if (err < best_err) {
                best_err = err;
                for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(c(i));
            }
        } // keepCount
    } // alpha

    // Fallback: if no candidate improved, solve unconstrained baseline (alpha=0)
    if (!std::isfinite(best_err)) {
        Eigen::MatrixXd ATA0; Eigen::VectorXd ATb0; std::vector<dd> tmp;
        build_weighted_ATA_ATb(0.0, ATA0, ATb0, tmp);
        Eigen::VectorXd c = robustSymmetricSolve(ATA0, ATb0);
        for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(c(i));
    }

    return best_coeffs;
}
