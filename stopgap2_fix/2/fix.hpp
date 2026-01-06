// Analytical L2 projection (Lebesgue L2) using shifted-Legendre basis on domain hull
std::vector<float> fitPolynomialLebesgueL2_AnalyticProjection(const std::vector<double>& x,
                                                              const std::vector<float>& y,
                                                              int degree);

// Backwards-compatible wrapper (keeps old API; ridge param ignored for analytic projection)
std::vector<float> fitPolynomialLebesgueL2(const std::vector<double>& x,
                                           const std::vector<float>& y,
                                           int degree,
                                           double ridge = 1e-14);
