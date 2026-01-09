// Improved composePolynomials using analytical Legendre projection
// This approach analytically integrates the piecewise polynomial representation
// and projects onto shifted-Legendre basis for better numerical conditioning

std::vector<float> AdvancedPolynomialFitter::composePolynomials(
    const float* p1_coeffs,
    double p1_delta,
    const float* p2_coeffs,
    double p2_delta,
    int degree) 
{
    using std::abs;
    int m = degree + 1;
    if (m <= 0) return {};
    double total_delta = p1_delta + p2_delta;
    if (total_delta <= kEps) return std::vector<float>(m, 0.0f);

    // Normalized boundary split
    double w1 = p1_delta / total_delta;
    double w2 = 1.0 - w1;

    // Convert inputs to double
    std::vector<double> p1d(m, 0.0), p2d(m, 0.0);
    for (int i = 0; i < m; ++i) {
        p1d[i] = double(p1_coeffs[i]);
        p2d[i] = double(p2_coeffs[i]);
    }

    // ========== ANALYTICAL INTEGRATION APPROACH ==========
    // Treat the composition as a piecewise polynomial function f(x) on [0,1]:
    //   f(x) = p1(x/w1)         for x in [0, w1]
    //   f(x) = p2((x-w1)/w2)    for x in [w1, 1]
    //
    // We'll compute integrals I_p = ∫_0^1 x^p * f(x) dx analytically
    // then project onto shifted-Legendre basis

    auto binomial_double = [](int n, int k) -> double {
        if (k < 0 || k > n) return 0.0;
        if (k == 0 || k == n) return 1.0;
        if (k > n/2) k = n - k;
        double res = 1.0;
        for (int i = 1; i <= k; ++i) res = res * double(n - i + 1) / double(i);
        return res;
    };

    // Compute I_p = ∫_0^1 x^p * f(x) dx for p = 0..degree
    // Split into two regions
    std::vector<dd> I_p_dd(m);
    for (int p = 0; p < m; ++p) I_p_dd[p] = dd(0.0);

    // Region 1: [0, w1], f(x) = p1(x/w1) = sum_k p1d[k] * (x/w1)^k
    if (w1 > kEps) {
        for (int k = 0; k < m; ++k) {
            if (abs(p1d[k]) < 1e-30) continue;
            double coef = p1d[k] / std::pow(w1, double(k));
            
            // ∫_0^{w1} x^p * (x^k) dx = ∫_0^{w1} x^{p+k} dx = w1^{p+k+1} / (p+k+1)
            for (int p = 0; p < m; ++p) {
                double integral = std::pow(w1, double(p + k + 1)) / double(p + k + 1);
                I_p_dd[p] = dd_add_double(I_p_dd[p], coef * integral);
            }
        }
    }

    // Region 2: [w1, 1], f(x) = p2((x-w1)/w2) = sum_k p2d[k] * ((x-w1)/w2)^k
    // Expand ((x-w1)/w2)^k = sum_j binom(k,j) * (-w1)^{k-j} * x^j / w2^k
    if (w2 > kEps) {
        for (int k = 0; k < m; ++k) {
            if (abs(p2d[k]) < 1e-30) continue;
            double w2_pow_k = std::pow(w2, double(k));
            
            for (int j = 0; j <= k; ++j) {
                double binom_coef = binomial_double(k, j);
                double term_coef = p2d[k] * binom_coef * 
                                  std::pow(-w1, double(k - j)) / w2_pow_k;
                
                // ∫_{w1}^1 x^p * x^j dx = (1^{p+j+1} - w1^{p+j+1}) / (p+j+1)
                for (int p = 0; p < m; ++p) {
                    double integral = (1.0 - std::pow(w1, double(p + j + 1))) / 
                                     double(p + j + 1);
                    I_p_dd[p] = dd_add_double(I_p_dd[p], term_coef * integral);
                }
            }
        }
    }

    // Convert to double
    std::vector<double> I_p(m);
    for (int p = 0; p < m; ++p) I_p[p] = dd_to_double(I_p_dd[p]);

    // ========== PROJECT ONTO SHIFTED-LEGENDRE BASIS ==========
    // Get shifted Legendre monomial coefficients
    VecMatrix L = getShiftedLegendreMonomialCoeffs(degree);

    // Compute inner products <P_k, f> = ∫_0^1 P_k(x) * f(x) dx
    // P_k(x) = sum_t L[k][t] * x^t
    std::vector<double> inner(m, 0.0);
    for (int k = 0; k < m; ++k) {
        dd accum(0.0);
        for (int t = 0; t <= k; ++t) {
            if (abs(L[k][t]) < 1e-30) continue;
            // ∫ x^t * f(x) dx = I_p[t]
            accum = dd_add_double(accum, L[k][t] * I_p[t]);
        }
        inner[k] = dd_to_double(accum);
    }

    // Norms: ∫_0^1 P_k(x)^2 dx = 1 / (2*k + 1) for shifted Legendre
    std::vector<double> norms(m);
    for (int k = 0; k < m; ++k) {
        norms[k] = 1.0 / double(2 * k + 1);
    }

    // Projection coefficients: a_k = <P_k, f> / ||P_k||^2
    std::vector<double> a_leg(m);
    for (int k = 0; k < m; ++k) {
        a_leg[k] = inner[k] / norms[k];
    }

    // ========== CONVERT BACK TO MONOMIAL BASIS ==========
    // p(x) = sum_k a_leg[k] * P_k(x) = sum_k a_leg[k] * sum_t L[k][t] * x^t
    std::vector<dd> c_x_dd(m);
    for (int j = 0; j < m; ++j) c_x_dd[j] = dd(0.0);

    for (int k = 0; k < m; ++k) {
        if (abs(a_leg[k]) < 1e-30) continue;
        for (int t = 0; t <= k; ++t) {
            double contrib = a_leg[k] * L[k][t];
            c_x_dd[t] = dd_add_double(c_x_dd[t], contrib);
        }
    }

    // ========== OPTIONAL: ENFORCE BOUNDARY CONDITIONS ==========
    // If you want to preserve endpoint values and derivatives
    // (similar to original algorithm), apply constrained adjustment
    
    // Evaluate original function at boundaries
    auto eval_poly = [](const std::vector<double>& c, double x) {
        double r = 0.0;
        for (int k = (int)c.size() - 1; k >= 0; --k) r = r * x + c[k];
        return r;
    };
    
    double f0_orig = eval_poly(p1d, 0.0);
    double f1_orig = eval_poly(p2d, 1.0);
    
    // Boundary value at junction
    double left_val = (w1 > kEps) ? eval_poly(p1d, 1.0) : eval_poly(p1d, 0.0);
    double right_val = (w2 > kEps) ? eval_poly(p2d, 0.0) : eval_poly(p2d, 0.0);
    double fw1_target = w1 * left_val + w2 * right_val;  // Blend at junction
    
    // Convert dd coefficients to Eigen vector for constraint solving
    Eigen::VectorXd c_proj(m);
    for (int i = 0; i < m; ++i) c_proj(i) = dd_to_double(c_x_dd[i]);
    
    // Check if boundary constraints are already well-satisfied
    double f0_proj = c_proj(0);  // x^0 term = f(0)
    double f1_proj = 0.0;
    double xpow = 1.0;
    for (int k = 0; k < m; ++k) {
        f1_proj += c_proj(k) * xpow;
        xpow *= 1.0;
    }
    double fw1_proj = 0.0;
    xpow = 1.0;
    for (int k = 0; k < m; ++k) {
        fw1_proj += c_proj(k) * xpow;
        xpow *= w1;
    }
    
    double boundary_err = abs(f0_proj - f0_orig) + 
                          abs(f1_proj - f1_orig) + 
                          abs(fw1_proj - fw1_target);
    
    // If boundary error is significant, apply minimal correction
    if (boundary_err > 1e-6 && m >= 3) {
        // Build constraint system: C * c = d
        Eigen::MatrixXd C(3, m);
        Eigen::VectorXd d(3);
        
        // f(0) = c_0
        for (int j = 0; j < m; ++j) C(0, j) = (j == 0) ? 1.0 : 0.0;
        d(0) = f0_orig;
        
        // f(1) = sum_j c_j
        for (int j = 0; j < m; ++j) C(1, j) = 1.0;
        d(1) = f1_orig;
        
        // f(w1) = sum_j c_j * w1^j
        xpow = 1.0;
        for (int j = 0; j < m; ++j) {
            C(2, j) = xpow;
            xpow *= w1;
        }
        d(2) = fw1_target;
        
        // Solve constrained least-squares: minimize ||c - c_proj||^2 s.t. C*c = d
        // Solution: c = c_proj + (C^T * (C*C^T)^{-1}) * (d - C*c_proj)
        Eigen::MatrixXd CCT = C * C.transpose();
        Eigen::VectorXd residual = d - C * c_proj;
        Eigen::VectorXd lambda = CCT.ldlt().solve(residual);
        Eigen::VectorXd correction = C.transpose() * lambda;
        c_proj += correction;
    }
    
    // Convert to float output
    std::vector<float> out(m);
    for (int i = 0; i < m; ++i) {
        out[i] = static_cast<float>(c_proj(i));
    }
    
    return out;
}
