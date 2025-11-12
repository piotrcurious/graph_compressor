// Compose polynomials with weak-extrema trimming + active-set QP to guarantee minima/maxima.
// Relies on dd helpers and Eigen present in the TU.
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

    // map weights
    double w1 = p1_delta / total_delta;
    double w2 = 1.0 - w1;

    // copy coefficients as double
    std::vector<double> p1d(m, 0.0), p2d(m, 0.0);
    for (int i = 0; i < m; ++i) { p1d[i] = double(p1_coeffs[i]); p2d[i] = double(p2_coeffs[i]); }

    // small binomial multiplicative comb (cheap)
    auto comb = [](int n, int k)->double{
        if (k < 0 || k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;
        if (k > n/2) k = n - k;
        double r = 1.0;
        for (int i = 1; i <= k; ++i) r = r * double(n - i + 1) / double(i);
        return r;
    };

    // ----------------------------
    // Build ATA analytically (monomial basis) ATA[r,c] = ∫_0^1 x^{r+c} dx = 1/(r+c+1)
    Eigen::MatrixXd ATA(m,m);
    for (int r = 0; r < m; ++r) for (int c = 0; c < m; ++c) ATA(r,c) = 1.0 / double(r + c + 1);

    // ----------------------------
    // Build ATb exactly using dd accumulation:
    // ATb_r = ∫_0^{w1} p1(x/w1) x^r dx + ∫_{w1}^{1} p2((x-w1)/w2) x^r dx
    std::vector<dd> ATb_dd(m);
    for (int r = 0; r < m; ++r) {
        // part from p1 on [0,w1]: w1^{r+1} * sum_k p1_k/(k+r+1)
        if (w1 > 0.0) {
            double factor = std::pow(w1, double(r + 1));
            for (int k = 0; k <= degree; ++k) {
                double contrib = p1d[k] / double(k + r + 1);
                ATb_dd[r] = dd_add_double(ATb_dd[r], factor * contrib);
            }
        }
        // part from p2 on [w1,1]: w2 * sum_k p2_k * sum_{t=0..r} C(r,t) w1^{r-t} w2^{t} / (k+t+1)
        if (w2 > 0.0) {
            for (int k = 0; k <= degree; ++k) {
                double p2k = p2d[k];
                double inner = 0.0;
                for (int t = 0; t <= r; ++t) {
                    double b = comb(r, t);
                    double w1pow = (r - t == 0) ? 1.0 : std::pow(w1, double(r - t));
                    double w2pow = (t == 0) ? 1.0 : std::pow(w2, double(t));
                    inner += b * w1pow * w2pow / double(k + t + 1);
                }
                ATb_dd[r] = dd_add_double(ATb_dd[r], p2k * inner * w2);
            }
        }
    }
    Eigen::VectorXd ATb(m);
    for (int r = 0; r < m; ++r) ATb(r) = dd_to_double(ATb_dd[r]);

    // small ridge for numeric stability
    double trace = ATA.trace();
    double ridge = 1e-14 * std::max(1.0, std::abs(trace));
    for (int i = 0; i < m; ++i) ATA(i,i) += ridge;

    // ----------------------------
    // Helper: evaluate polynomial in monomial basis (horner)
    auto eval_poly = [](const std::vector<double>& c, double u) {
        double r = 0.0;
        for (int k = (int)c.size()-1; k >= 0; --k) r = r * u + c[k];
        return r;
    };
    // derivative coefficients (monomial)
    auto deriv = [](const std::vector<double>& c) {
        int n = (int)c.size();
        if (n <= 1) return std::vector<double>{0.0};
        std::vector<double> d(n-1);
        for (int k = 0; k < n-1; ++k) d[k] = double(k+1) * c[k+1];
        return d;
    };
    // second derivative coefficients
    auto second_deriv = [](const std::vector<double>& c) {
        int n = (int)c.size();
        if (n <= 2) return std::vector<double>{0.0};
        std::vector<double> s(n-2, 0.0);
        for (int k = 2; k < n; ++k) s[k-2] = double(k) * double(k-1) * c[k];
        return s;
    };

    // ----------------------------
    // Find extrema (roots of derivatives) for p1 and p2 in u∈[0,1], mapped to global x∈[0,1].
    auto find_real_roots_unit = [&](const std::vector<double>& poly)->std::vector<double>{
        std::vector<double> roots;
        int deg = (int)poly.size() - 1;
        while (deg > 0 && std::abs(poly[deg]) < 1e-18) --deg;
        if (deg <= 0) return roots;
        if (deg == 1) {
            double a = poly[1], b = poly[0];
            if (a != 0.0) {
                double r = -b/a;
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
        for (int i = 1; i < deg; ++i) C(i, i-1) = 1.0;
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
    // p1 extrema
    {
        auto d1 = deriv(p1d);
        auto r1 = find_real_roots_unit(d1);
        for (double u : r1) {
            double x = w1 * u;
            if (x >= 0.0 && x <= 1.0) extrema_x.push_back(x);
        }
        // include endpoints to preserve boundary extrema
        extrema_x.push_back(0.0);
        extrema_x.push_back(w1);
    }
    // p2 extrema
    if (w2 > 0.0) {
        auto d2 = deriv(p2d);
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
    std::vector<double> extrema_clean;
    for (double xv : extrema_x) {
        double xv2 = std::min(1.0, std::max(0.0, xv));
        if (extrema_clean.empty() || std::abs(xv2 - extrema_clean.back()) > 1e-12) extrema_clean.push_back(xv2);
    }
    extrema_x.swap(extrema_clean);

    // ----------------------------
    // Scoring & trimming of extrema:
    // Score = alpha * |f''(x)| (scaled into x-space) + beta * prominence
    struct Ext {
        double x;
        double score;
        bool is_from_p1; // helpful if needed
    };
    std::vector<Ext> extents;

    // small functions to evaluate f, f' and f'' in global x
    auto eval_f_global = [&](double x)->double {
        if (x <= w1) {
            double u = (w1 == 0.0) ? 0.0 : (x / w1);
            return eval_poly(p1d, u);
        } else {
            double u = (w2 == 0.0) ? 0.0 : ((x - w1) / w2);
            return eval_poly(p2d, u);
        }
    };
    auto eval_fpp_global = [&](double x)->double {
        // compute second derivative w.r.t x: f''(x) = p''(u) / scale^2 where u = (x - shift)/scale
        if (x <= w1) {
            double scale = w1; if (scale == 0.0) scale = 1.0;
            double u = (x / w1);
            std::vector<double> s2 = second_deriv(p1d);
            double v = 0.0;
            double upow = 1.0;
            for (int k = 0; k < (int)s2.size(); ++k) { v += s2[k] * upow; upow *= u; }
            return v / (scale * scale);
        } else {
            double scale = w2; if (scale == 0.0) scale = 1.0;
            double u = (x - w1) / w2;
            std::vector<double> s2 = second_deriv(p2d);
            double v = 0.0;
            double upow = 1.0;
            for (int k = 0; k < (int)s2.size(); ++k) { v += s2[k] * upow; upow *= u; }
            return v / (scale * scale);
        }
    };

    // prominence: measure small-offset neighbor difference in u-space (h = 1e-3)
    const double h_u = 1e-3;
    for (double x : extrema_x) {
        // determine which polynomial produced this extremum (by u location)
        bool from_p1 = (x <= w1 + 1e-15);
        double fval = eval_f_global(x);
        // second derivative magnitude
        double fpp = eval_fpp_global(x);
        double fpp_abs = std::abs(fpp);
        // prominence: difference to nearby points in same segment
        double prom = 0.0;
        if (from_p1) {
            double scale = (w1 == 0.0) ? 1.0 : w1;
            double u = (scale == 0.0) ? 0.0 : (x / scale);
            double umin = std::max(0.0, u - h_u), umax = std::min(1.0, u + h_u);
            double f1 = eval_poly(p1d, umin);
            double f2 = eval_poly(p1d, umax);
            prom = std::min(std::abs(fval - f1), std::abs(fval - f2));
        } else {
            double scale = (w2 == 0.0) ? 1.0 : w2;
            double u = (scale == 0.0) ? 0.0 : ((x - w1) / scale);
            double umin = std::max(0.0, u - h_u), umax = std::min(1.0, u + h_u);
            double f1 = eval_poly(p2d, umin);
            double f2 = eval_poly(p2d, umax);
            prom = std::min(std::abs(fval - f1), std::abs(fval - f2));
        }
        // score weights
        double alpha = 1.0;
        double beta = 1.0;
        double score = alpha * fpp_abs + beta * prom;
        extents.push_back({x, score, from_p1});
    }

    // if none found, proceed unconstrained solution
    if (extents.empty()) {
        Eigen::VectorXd coeffs = robustSymmetricSolve(ATA, ATb);
        std::vector<float> out(m);
        for (int i = 0; i < m; ++i) out[i] = static_cast<float>(coeffs(i));
        return out;
    }

    // sort descending by score
    std::sort(extents.begin(), extents.end(), [](const Ext& a, const Ext& b){ return a.score > b.score; });

    // capacity: each extremum uses 2 equality constraints (value + derivative). Keep at most floor(m/2).
    int max_extrema_keep = std::max(1, m / 2);
    int keep_count = std::min((int)extents.size(), max_extrema_keep);

    // ensure we always include endpoints if present and strong (they're in extents sorted)
    std::vector<double> kept_x;
    for (int i = 0; i < keep_count; ++i) kept_x.push_back(extents[i].x);

    // ----------------------------
    // Build equality constraints C_eq * c = d_eq from kept extrema:
    // For each x: value constraint sum_k c_k x^k = f(x)
    //            derivative constraint sum_k k*c_k x^{k-1} = 0
    std::vector<std::vector<double>> C_eq;
    std::vector<double> d_eq;
    for (double x : kept_x) {
        // value
        std::vector<double> rowV(m);
        double xp = 1.0;
        for (int k = 0; k < m; ++k) { rowV[k] = xp; xp *= x; }
        C_eq.push_back(rowV);
        d_eq.push_back(eval_f_global(x));
        // derivative
        if (m >= 2) {
            std::vector<double> rowD(m, 0.0);
            double xpow = 1.0;
            for (int k = 1; k < m; ++k) {
                rowD[k] = double(k) * xpow;
                xpow *= x;
            }
            C_eq.push_back(rowD);
            d_eq.push_back(0.0);
        }
    }

    // If we've inadvertently created too many equalities (>= m), cut the last ones to keep solvable
    if ((int)C_eq.size() >= m) {
        // keep only first (m-1) equalities and leave room for KKT; but safer to reduce to m-1 equalities
        C_eq.resize(std::max(0, m-1));
        d_eq.resize(C_eq.size());
    }

    // ----------------------------
    // Helper: build KKT and solve equality-constrained QP:
    // minimize 1/2 c^T ATA c - ATb^T c  subject to A c = b
    auto solve_eq_constrained = [&](const std::vector<std::vector<double>>& Aeq, const std::vector<double>& beq) -> std::pair<bool, Eigen::VectorXd> {
        int neq = (int)Aeq.size();
        // Build KKT matrix:
        // [ ATA  Aeq^T ]
        // [ Aeq   0   ]
        int K = m + neq;
        Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(K, K);
        for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) KKT(i,j) = ATA(i,j);
        for (int i = 0; i < neq; ++i) for (int j = 0; j < m; ++j) {
            KKT(j, m + i) = Aeq[i][j];
            KKT(m + i, j) = Aeq[i][j];
        }
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(K);
        for (int i = 0; i < m; ++i) rhs(i) = ATb(i);
        for (int i = 0; i < neq; ++i) rhs(m + i) = beq[i];
        // Solve
        Eigen::VectorXd sol;
        // prefer a robust solver; use colPivHouseholderQr here
        sol = KKT.colPivHouseholderQr().solve(rhs);
        if (!std::isfinite(sol.norm())) return {false, Eigen::VectorXd()};
        Eigen::VectorXd c = sol.head(m);
        return {true, c};
    };

    // initial equality-constrained solve
    std::pair<bool, Eigen::VectorXd> res = solve_eq_constrained(C_eq, d_eq);
    Eigen::VectorXd c;
    if (!res.first) {
        // fallback to unconstrained robust solve
        Eigen::VectorXd unconstr = robustSymmetricSolve(ATA, ATb);
        c = unconstr;
    } else {
        c = res.second;
    }

    // ----------------------------
    // Build inequality constraints (second derivative sign) for kept extrema.
    // For each kept x, second derivative vector row: a[k] = k*(k-1)*x^{k-2} for k>=2 else 0
    struct Inequality {
        std::vector<double> row;
        double bound; // we use 0.0 bound for second derivative sign
        bool require_positive; // true -> second derivative >= 0 (minimum); false -> <= 0 (maximum)
        double x;
    };
    std::vector<Inequality> inequalities;
    // Determine which source extremum is min or max by inspecting original second derivative sign
    for (double
