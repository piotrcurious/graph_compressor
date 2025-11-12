// --- IMPROVED Fallback Level 0: C0 Endpoints + iterative relaxation of C0 Boundary ---
DEBUG_PRINT("--- Fallback Level 0 (improved): Trying C0 Endpoints + C0 Boundary (with iterative relaxation) ---\n");

// Base equality set: endpoints + boundary value (exact)
std::vector<std::vector<double>> C_eq_fb0; std::vector<double> d_eq_fb0;
C_eq_fb0.push_back(build_value_row(0.0)); d_eq_fb0.push_back(f0);
C_eq_fb0.push_back(build_value_row(1.0)); d_eq_fb0.push_back(f1);
C_eq_fb0.push_back(build_value_row(w1));  d_eq_fb0.push_back(boundary_val_target);

// If not overconstrained, attempt exact solve first
if ((int)C_eq_fb0.size() < m) {
    auto sol0 = qp_solve(ATA0, ATb0, C_eq_fb0, d_eq_fb0, {}, {});
    if (sol0.first) {
        DEBUG_PRINT("--- Fallback Level 0 (exact) Succeeded. ---\n");
        for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(sol0.second(i));
        return best_coeffs; // Solution found!
    }
}
DEBUG_PRINT("--- Fallback Level 0 (exact) Failed. Entering relaxation attempts. ---\n");

// Iterative relaxation parameters (increase delta multiplicatively)
double base_eps = 1e-10 * std::max(1.0, std::abs(boundary_val_target));
std::vector<double> relax_factors = {1.0, 1e1, 1e2, 1e4, 1e8};

// Helper to build "band" inequalities for a value row: row*c in [target-delta, target+delta]
// returns pair {Ain_rows, bin_rows}
auto make_band_inequalities = [&](const std::vector<double>& row, double target, double delta) {
    std::vector<std::vector<double>> Ain_rows_local;
    std::vector<double> bin_rows_local;
    // row * c <= target + delta
    Ain_rows_local.push_back(row);
    bin_rows_local.push_back(target + delta);
    // -row * c <= -target + delta  <=>  row * c >= target - delta
    std::vector<double> negrow = row;
    for (double &v : negrow) v = -v;
    Ain_rows_local.push_back(negrow);
    bin_rows_local.push_back(-target + delta);
    return std::make_pair(Ain_rows_local, bin_rows_local);
};

// Stage A: relax boundary only (endpoints remain exact)
if ((int)C_eq_fb0.size() < m) {
    std::vector<double> boundary_row = build_value_row(w1);
    for (double f : relax_factors) {
        double delta = base_eps * f;
        auto band = make_band_inequalities(boundary_row, boundary_val_target, delta);
        // call qp_solve with endpoints as equalities and boundary as inequalities (band)
        auto sol_relaxed = qp_solve(ATA0, ATb0, C_eq_fb0, d_eq_fb0, band.first, band.second);
        if (sol_relaxed.first) {
            DEBUG_PRINT("--- Fallback Level 0: Succeeded with boundary relaxed (delta=%g). ---\n", delta);
            for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(sol_relaxed.second(i));
            return best_coeffs;
        }
    }
}

// Stage B: relax boundary + endpoints (convert all three equalities to bands)
{
    // Build rows for endpoints and boundary
    std::vector<double> row0 = build_value_row(0.0);
    std::vector<double> row1 = build_value_row(1.0);
    std::vector<double> rowb = build_value_row(w1);

    for (double f : relax_factors) {
        double delta = base_eps * f;
        std::vector<std::vector<double>> Ain_all;
        std::vector<double> bin_all;

        // Add band for boundary
        auto band_b = make_band_inequalities(rowb, boundary_val_target, delta);
        Ain_all.insert(Ain_all.end(), band_b.first.begin(), band_b.first.end());
        bin_all.insert(bin_all.end(), band_b.second.begin(), band_b.second.end());

        // Add band for endpoint 0
        auto band_0 = make_band_inequalities(row0, f0, delta);
        Ain_all.insert(Ain_all.end(), band_0.first.begin(), band_0.first.end());
        bin_all.insert(bin_all.end(), band_0.second.begin(), band_0.second.end());

        // Add band for endpoint 1
        auto band_1 = make_band_inequalities(row1, f1, delta);
        Ain_all.insert(Ain_all.end(), band_1.first.begin(), band_1.first.end());
        bin_all.insert(bin_all.end(), band_1.second.begin(), band_1.second.end());

        // No equality constraints in this stage (or we can keep derivative equalities if desired)
        std::vector<std::vector<double>> empty_eq;
        std::vector<double> empty_eq_b;

        // Only attempt if not trivially overconstrained (Ain can be any size)
        auto sol_relaxed2 = qp_solve(ATA0, ATb0, empty_eq, empty_eq_b, Ain_all, bin_all);
        if (sol_relaxed2.first) {
            DEBUG_PRINT("--- Fallback Level 0: Succeeded with endpoints+boundary relaxed (delta=%g). ---\n", delta);
            for (int i = 0; i < m; ++i) best_coeffs[i] = static_cast<float>(sol_relaxed2.second(i));
            return best_coeffs;
        }
    }
}

DEBUG_PRINT("--- Fallback Level 0 (improved): All relaxation attempts failed. ---\n");
