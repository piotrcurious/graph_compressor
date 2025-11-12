// Improved composePolynomials: numerically robust, precomputed binom/powers, dd accumulation
std::vector<float> AdvancedPolynomialFitter::composePolynomials(const float* p1_coeffs,
                                                                double p1_delta,
                                                                const float* p2_coeffs,
                                                                double p2_delta,
                                                                int degree) {
    int m = degree + 1;
    // basic validation
    double total_delta = p1_delta + p2_delta;
    if (m <= 0) return {};
    if (total_delta <= 0.0) return std::vector<float>(m, 0.0f);

    // Convert inputs to double for internal work
    std::vector<double> p1d(m, 0.0), p2d(m, 0.0);
    for (int i = 0; i < m; ++i) {
        p1d[i] = double(p1_coeffs[i]);
        p2d[i] = double(p2_coeffs[i]);
    }

    // weights and small-shortcut
    double w1 = p1_delta / total_delta;
    double w2 = 1.0 - w1;

    // precompute binomial coefficients (Pascal triangle) up to degree
    std::vector<std::vector<double>> binom(m, std::vector<double>(m, 0.0));
    for (int n = 0; n < m; ++n) {
        binom[n][0] = 1.0;
        binom[n][n] = 1.0;
        for (int k = 1; k < n; ++k) binom[n][k] = binom[n-1][k-1] + binom[n-1][k];
    }

    // precompute powers of w1 and w2 up to degree (iteratively)
    std::vector<double> pow_w1(m+1, 1.0), pow_w2(m+1, 1.0); // note pow_w1[l] = w1^l
    for (int i = 1; i <= m; ++i) {
        pow_w1[i] = pow_w1[i-1] * w1;
        pow_w2[i] = pow_w2[i-1] * w2;
    }

    // precompute inverses of (idx + 1) for idx in [0 .. 2*degree]
    std::vector<double> invDen(2 * degree + 2, 0.0);
    for (int idx = 0; idx < (int)invDen.size(); ++idx) invDen[idx] = 1.0 / double(idx + 1);

    // shifted Legendre polynomials as monomial coefficients (same as before)
    VecMatrix legendre_mono = getShiftedLegendreMonomialCoeffs(degree);

    // accumulators for Legendre projection coefficients (use dd for better precision)
    std::vector<dd> legendre_coeffs_dd(m);
    for (int j = 0; j <= degree; ++j) {
        // integral_1 = \sum_{k=0..deg}\sum_{l=0..j} p1[k] * L[l] * (w1^{l+1} / (k + l + 1))
        dd integral1_dd(0.0);
        for (int l = 0; l <= j; ++l) {
            double L_l = (l < (int)legendre_mono[j].size()) ? legendre_mono[j][l] : 0.0;
            if (L_l == 0.0) continue;
            double w1pow = pow_w1[l + 1]; // w1^(l+1)
            for (int k = 0; k <= degree; ++k) {
                double contrib = p1d[k] * L_l * w1pow * invDen[k + l];
                integral1_dd = dd_add_double(integral1_dd, contrib);
            }
        }

        // Build q[0..j] = sum_{l=0..j} L[l] * C(l,mm) * w1^{l-mm} * w2^{mm} (evaluate once per j)
        std::vector<double> q(j + 1, 0.0);
        for (int l = 0; l <= j; ++l) {
            double L_l = (l < (int)legendre_mono[j].size()) ? legendre_mono[j][l] : 0.0;
            if (L_l == 0.0) continue;
            // we want terms mm = 0..l: binom[l][mm] * w1^{l-mm} * w2^{mm}
            // use pow_w1 and pow_w2
            for (int mm = 0; mm <= l; ++mm) {
                double term = L_l * binom[l][mm] * (pow_w1[l - mm]) * (pow_w2[mm]);
                q[mm] += term;
            }
        }

        // integral_2 = w2 * sum_{k=0..deg} sum_{mm=0..j} p2[k] * q[mm] / (k + mm + 1)
        dd integral2_dd(0.0);
        for (int mm = 0; mm <= j; ++mm) {
            double qmm = q[mm];
            if (qmm == 0.0) continue;
            for (int k = 0; k <= degree; ++k) {
                double contrib = p2d[k] * qmm * invDen[k + mm];
                integral2_dd = dd_add_double(integral2_dd, contrib);
            }
        }
        // multiply integral2 by w2 (use dd_mul_double)
        integral2_dd = dd_mul_double(integral2_dd, w2);

        // sum integrals, divide by normalization 1/(2*j+1)
        dd total_dd = dd_add(integral1_dd, integral2_dd);
        double total = dd_to_double(total_dd);
        double normalization = 1.0 / (2.0 * j + 1.0);
        legendre_coeffs_dd[j] = dd(total * normalization);
    }

    // convert dd legendre coeffs to doubles for final conversion
    std::vector<double> legendre_coeffs(m, 0.0);
    for (int j = 0; j <= degree; ++j) legendre_coeffs[j] = dd_to_double(legendre_coeffs_dd[j]);

    // convert back to monomial coefficients in x using legendre_mono (accumulate with dd)
    std::vector<dd> newc_dd(m);
    for (int j = 0; j <= degree; ++j) {
        double lc = legendre_coeffs[j];
        if (lc == 0.0) continue;
        const std::vector<double>& Lmono = legendre_mono[j];
        // Lmono[t] gives coefficient for x^t in L_j (shifted monomial form)
        for (int t = 0; t <= degree && t < (int)Lmono.size(); ++t) {
            double coef = Lmono[t];
            if (coef == 0.0) continue;
            double contrib = lc * coef;
            newc_dd[t] = dd_add_double(newc_dd[t], contrib);
        }
    }

    // convert to float output
    std::vector<float> out(m, 0.0f);
    for (int t = 0; t < m; ++t) out[t] = static_cast<float>(dd_to_double(newc_dd[t]));
    return out;
}
