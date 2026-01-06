Nice — done. I replaced the numerical L² solver with the analytic shifted-Legendre projection and left everything else intact so the rest of your code (including the L¹ IRLS warm start) now uses the analytic projection by default.

Below are the exact edits to make. You can paste these into your repo.

1) Header change

In AdvancedPolynomialFitter.hpp add this declaration (near other public fitters):

// Analytical L2 projection (Lebesgue L2) using shifted-Legendre basis on domain hull
std::vector<float> fitPolynomialLebesgueL2_AnalyticProjection(const std::vector<double>& x,
                                                              const std::vector<float>& y,
                                                              int degree);

// Backwards-compatible wrapper (keeps old API; ridge param ignored for analytic projection)
std::vector<float> fitPolynomialLebesgueL2(const std::vector<double>& x,
                                           const std::vector<float>& y,
                                           int degree,
                                           double ridge = 1e-14);

2) Implementation: replace old fitPolynomialLebesgueL2 with a thin wrapper that calls the analytic projection

Find the previous implementation of fitPolynomialLebesgueL2 (the version that built ATA/ATb and called robustSymmetricSolve) and replace it with this code. Also ensure you have the fitPolynomialLebesgueL2_AnalyticProjection(...) implementation pasted earlier (the long function I supplied that computes the analytic projection via shifted Legendre). If you haven't yet pasted that analytic function, paste it first, then the wrapper below.

// Wrapper: call analytic projection by default (keeps old signature)
std::vector<float> AdvancedPolynomialFitter::fitPolynomialLebesgueL2(const std::vector<double>& x,
                                                                     const std::vector<float>& y,
                                                                     int degree,
                                                                     double /*ridge*/) {
    // Use analytic projection (shifted-Legendre) by default for Lebesgue L2.
    // Ridge parameter is ignored because analytic projection does not solve a normal system.
    return fitPolynomialLebesgueL2_AnalyticProjection(x, y, degree);
}

3) Ensure the analytic function is present (repeat if needed)

If you haven't already added it, paste the fitPolynomialLebesgueL2_AnalyticProjection(...) function I provided in the earlier message (the one that:

computes Iy = ∫ x^j y_lin(x) dx via double-double accumulation,

uses getShiftedLegendreMonomialCoeffs(degree),

computes inner products, divides by norms scale/(2k+1),

converts back to monomial basis with dd).


(That full function was long; keep it in the same translation unit as the wrapper and dd helpers.)

4) Notes & rationale (short)

The wrapper preserves your API (so callers like the L¹ IRLS warm-start still work).

The analytic projection is numerically stable (orthogonal basis) and Lebesgue-correct for the piecewise-linear interpolant of your samples.

If you still want a ridge fallback for extremely ill-conditioned pathological cases, we can add a tiny regularization by shrinking high-order Legendre coefficients (Tikhonov in Legendre basis) — straightforward to add.


5) Quick smoke test

Add a small unit test in your test harness or main():

// simple smoke
std::vector<double> xs = {0.0, 0.3, 0.6, 1.0};
std::vector<float> ys = {1.0f, 0.8f, 0.5f, 0.0f};
int deg = 3;
auto coeffs = fitter.fitPolynomialLebesgueL2(xs, ys, deg);
for (size_t i = 0; i < coeffs.size(); ++i) printf("c[%zu]=%g\n", i, coeffs[i]);

Compare results quickly against the old numerical L2 (if you kept a copy) and the L1-IRLS warm-start behavior.


---

If you want, I can:

Add a tiny Legendre-diagonal Tikhonov option (damps high-degree coefficients),

Integrate a piecewise-constant weighted analytic projection to be used inside IRLS (so IRLS no longer solves normal eqns),

Or run through and produce a small patch / git diff for you.


Which would you like next?
