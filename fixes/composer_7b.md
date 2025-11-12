AdvancedPolynomialFitter::composePolynomials — Detailed documentation


---

1. High-level summary

AdvancedPolynomialFitter::composePolynomials builds a single polynomial approximation (degree = degree) that best blends two input polynomials defined on subintervals of [0,1]:

p1 is defined on [0, w1] (mapped from normalized u ∈ [0,1] via x = w1 * u)

p2 is defined on [w1, 1] (mapped from normalized u ∈ [0,1] via x = w1 + w2 * u)


The routine minimizes a weighted squared error (L2 objective) between the composite target function (piecewise p1/p2) and the fitted polynomial, with optional second-derivative sign constraints at selected extrema and at the boundary. It uses many candidate regularization weights (alpha) and different numbers of enforced interior extrema to find the best constrained quadratic programming (QP) solution. If constrained solves fail, a sequence of fallbacks is attempted, finishing with an unconstrained least-squares solve.

The code is written for numerical robustness and portability (Arduino/desktop) and assumes existence of:

dd double-double arithmetic helpers: dd(...), dd_add_double(dd, double), dd_to_double(dd).

robustSymmetricSolve(const Eigen::MatrixXd&, const Eigen::VectorXd&) as a stable solver for symmetric positive semidefinite systems.

Eigen 3 (Eigen/Core, Eigen/QR, Eigen/Eigenvalues) and the symbol AdvancedPolynomialFitter with this member function.



---

2. Function signature / parameters

std::vector<float> AdvancedPolynomialFitter::composePolynomials(
    const float* p1_coeffs, // pointer to degree+1 coefficients of p1, index k is coefficient of x^k in p1(u)
    double p1_delta,        // length (weight) of p1 subinterval proportional to w1
    const float* p2_coeffs, // pointer to degree+1 coefficients of p2, index k is coefficient of x^k in p2(u)
    double p2_delta,        // length (weight) of p2 subinterval proportional to w2
    int degree              // polynomial degree (m = degree + 1 coefficients returned)
);

Returns: std::vector<float> of length degree + 1 containing the fitted polynomial coefficients (coefficient i is for x^i). Returns empty vector on critical failure.



---

3. Constants and tolerances

Located in an anonymous namespace and used across the routine:

kEps = 1e-15 — tiny epsilon to avoid division by zero and treat near-zero as zero.

kRootTol = 1e-12 — tolerance used to consider two roots equal and to accept roots slightly outside [0,1].

kImagTol = 1e-9 — acceptable imaginary part magnitude when interpreting companion-matrix eigenvalues as real roots.

kLeadCoeffTol = 1e-30 — threshold below which a polynomial's leading coefficient is treated as numerically zero (skip companion matrix).

kFuzzyBoundary = 1e-15 — small fuzz to choose which piece (p1/p2) a point belongs to.

kFeasTol = 1e-8 — allowed final inequality constraint violation after QP (feasibility tolerance).


These are chosen conservatively; you can tune them if you expect different arithmetic precision (e.g., long double or double-double throughout).


---

4. Mathematical objective

The fitted polynomial q(x) minimizes a weighted integral error over [0,1]:

Objective
Eα(q) = ∫_0^1 (q(x) - f(x))^2 dx  +  α ∫_0^1 (q''(matched-scale))^2 dx

where:

f(x) is the piecewise function: p1(u) on [0,w1] (with u = x/w1) and p2(u) on [w1,1] (with u = (x - w1)/w2).

The code forms the quadratic objective in coefficient space: 0.5 * c^T ATA c - ATb^T c (but implemented as c^T ATA c - 2 ATb^T c + const when computing error).

alpha is a regularization parameter (varied across a candidate set) that weights the second-derivative penalty (the code uses an analytic derivation of the integral contributions).


All integrals are computed analytically as rational expressions 1/(n+1), 1/(n+2), 1/(n+3) etc., after mapping coordinates.


---

5. Numerical techniques & helpers

5.1 Coefficient conversions

Inputs p1_coeffs / p2_coeffs are copied into std::vector<double> arrays (p1d, p2d) to perform double arithmetic for higher precision.

5.2 dd accumulators

Parts of ATb (right-hand vector of quadratic objective) and F2_const (exact ∫ f(x)^2 dx) use dd (double-double) accumulation to reduce rounding error when summing many small contributions. After accumulation, results are converted with dd_to_double(...).

5.3 Combinatorics helper

A small comb(n,k) function calculates binomial coefficients iteratively with stability measures and symmetry.

5.4 Polynomial evaluation and derivatives

eval_poly(c, u) — Horner evaluation of polynomial coefficients in vector c at value u.

deriv_coeffs(c) — returns coefficient vector for first derivative.

second_deriv_coeffs(c) — returns coefficient vector for second derivative.


Precomputed vectors: p1d_deriv, p1d_second, p2d_deriv, p2d_second.

5.5 Mappings & scaling

To evaluate piecewise f, f', and f'' at a global x ∈ [0,1], the code maps to local u and divides by the appropriate scale factors (w1 or w2, and squares for second derivatives). Boundary fuzzing uses kFuzzyBoundary to decide piece ownership when x is exactly w1.


---

6. Candidate extrema detection and scoring

To select interior points where sign-of-curvature constraints may be useful, the code:

1. Finds roots in [0,1] of the derivative polynomial of each input piece using companion matrix eigenvalues via Eigen.

For degree 1, uses closed-form root.

Skips companion matrix if leading coefficient < kLeadCoeffTol.

Accepts only nearly-real eigenvalues (abs(imag) < kImagTol), clipped into [0,1] with kRootTol tolerance.



2. Converts roots in local u into global x and collects endpoints and w1.


3. Scores each candidate x by combining:

|f''(x)| (magnitude of second derivative at the point),

a “prominence” metric — how different the value at the point is compared to neighborhoods u±hu (hu = 1e-4).



4. Sorts scored candidates descending by score and uses the top keepCount for constraints.



This provides a heuristic to enforce concavity/convexity constraints at the most significant extrema.


---

7. Constraint construction

Equality constraints (Aeq / be)

Endpoint values q(0) = f(0) and q(1) = f(1).

Optionally preserve endpoint derivative q'(0) = 0 and q'(1) = 0 if f' at those endpoints is nearly zero.

Boundary continuity at x = w1:

value constraint q(w1) = w1 * left_val + w2 * right_val (blended target).

derivative constraint q'(w1) = w1*left_d + w2*right_d (blended derivative).


For each selected interior extrema (up to keepCount), adds:

value equality q(x) = f(x),

derivative equality q'(x) = 0 (if degree m >= 2) to enforce stationary point.



If the count of equality constraints reaches or exceeds m (number of unknown coefficients), the candidate is skipped (insufficient DOF left).

Inequality constraints (Ain / bin)

For boundary and chosen interior extrema, a second-derivative sign inequality is placed:

If original f''(x) >= 0 at the point, we enforce q''(x) >= 0 (implemented as -q''(x) <= 0 if code expects Ain * c <= bin).

Otherwise q''(x) <= 0.



The build_second_deriv_row(x) builds the row of second derivative values for polynomial coefficients (for constraint purposes).


---

8. QP solver (primal-dual interior-point) — qp_solve

Purpose

Solve the quadratic program of the form:

Minimize: 0.5 * c^T ATA c - ATb^T c
Subject to: Aeq * c = be, Ain * c <= bin

(Implementation uses the KKT system and primal-dual interior-point iterations.)

Key features and protections

If there are no inequalities (nin == 0), it solves equality-constrained KKT system directly (single linear solve).

Attempts an initial KKT-based guess; if that fails, uses robustSymmetricSolve(ATA, ATb) as fallback for initial c.

Initializes positive slack s and dual variables z safely; ensures s and z remain positive via small clamping minima to prevent division-by-zero.

Constructs a reduced symmetric system M = ATA + Σ (w_i * a_i^T a_i) where w_i = (SinvZ)_i, and solves augmented KKT linear system for primal-dual steps.

Performs step-length backtracking based on negative components of the primal/dual directions and uses tau = 0.995 to stay interior.

Stops when residual and complementarity (mu) are below tolerance or after a maximum number (max_iters = 60) iterations.

After convergence, checks inequality feasibility and rejects solution if maximum violation exceeds kFeasTol.


Return values

std::pair<bool, Eigen::VectorXd>:

first = success / feasible solution found,

second = coefficient vector c (length m) on success; otherwise empty Eigen::VectorXd().




---

9. Search strategy (alpha × keepCount)

The code loops over a candidate set of alpha values that weight the penalty (roughness) and a set of numbers of interior extrema to keep (keepCount from 0 to m/2). For each pair:

1. Build ATA_alpha and ATb_alpha via build_weighted_ATA_ATb(alpha...).


2. Add a tiny ridge ridge = 1e-14 * max(1, |trace|) to diagonal entries of ATA for stability.


3. Construct equality & inequality constraints as described.


4. If number of equalities >= m, skip (overconstrained).


5. Call qp_solve. If successful, compute exact error err = c^T ATA c - 2 ATb^T c + F2_const and keep best.



A small early-exit heuristic stops the search if best_err becomes extremely small relative to F2_const.


---

10. Fallback hierarchy

If no candidate produces a feasible solution, a cascade of fallback attempts is used:

1. Fallback Level 0 — Solve with only C0 endpoints (q(0)=f(0), q(1)=f(1)) and value continuity at boundary q(w1)=boundary_val_target (no inequalities). If feasible, return.


2. Fallback Level 1 — Solve with only C0 endpoints (q(0)=f(0), q(1)=f(1)).


3. Final Fallback — Unconstrained least-squares solution using robustSymmetricSolve(ATA0, ATb0) where ATA0 is the alpha=0 (pure L2) ATA. If this yields finite coefficients, return them.


4. Absolute failure — return empty vector and emit debug message.



All fallback steps add the same tiny ridge to ATA0 for numerical stability.


---

11. Output format

Returns std::vector<float> length m = degree + 1. Coefficient i corresponds to x^i power.

If everything fails, the function returns an empty vector {}.


---

12. Complexity

Computing ATA is O(m^2) entries where each entry computes small rational terms -> roughly O(m^2).

Building ATb has two nested loops over r up to m and k up to degree and sometimes inner t loop up to r for p2 integrals -> asymptotically O(m^3) in the worst case for ATb build.

Root-finding via companion matrix for degree d costs eigenvalue decomposition of deg×deg companion matrix: O(deg^3) but applied to degree up to degree - 1.

QP solve: each interior iteration requires solving linear system of size m + neq. Using colPivHouseholderQr has typical cost O((m+neq)^3) per solve. Iterations up to max_iters (=60 but practically <10–20 usually).

The outer candidate search multiplies by number of alpha values (~5) and keepCount values (~m/2).


In short: this is moderately expensive. For degree up to ~8–12 on desktop it’s fine; for embedded, consider a fixed-size or simplified approach.


---

13. Portability / integration notes

Arduino: printf is not universally supported. Code defines a DEBUG_PRINT macro using Serial.printf when ARDUINO is defined and falls back to printf. (Ensure Serial is initialized before logging on Arduino).

Eigen: dynamic-sized Eigen::MatrixXd / VectorXd are used. On small microcontrollers this will be memory-heavy. Options:

Replace dynamic Eigen with fixed-size Eigen::Matrix<double, N, N> if you know maximum degree.

Use a small hand-rolled linear algebra solver for small m.


dd double-double: The code assumes a dd type implements accurate accumulation. If dd is not available, replacing with long double reduces precision but may be acceptable for many cases.

robustSymmetricSolve: referenced as a fallback; ensure it’s implemented and returns an Eigen::VectorXd.



---

14. Safety / numerical robustness features (what was done)

tiny diagonal ridge to guard against singular ATA matrices

double-double accumulators for ATb and F2 integrals

companion matrix root-finding guarded by tiny lead threshold

clamping of slack s and dual z to avoid division-by-zero

final feasibility check for inequality violations with kFeasTol

early-exit when error is extremely small relative to F2_const

minimal debug printing macro that adapts between Arduino and desktop



---

15. Suggested unit tests / verification suite

Create tests that evaluate:

1. Trivial tests

p1 == p2 == constant → result should be that constant polynomial exactly (or near machine epsilon).

p1 == p2 == linear with no kink → fit should reproduce linear exactly.



2. Boundary-only information

p1 and p2 different but both simple (linear). Verify continuity/derivative blending behavior.



3. High-curvature interior

Make p1 or p2 cubic with pronounced interior extrema; verify solver enforces curvature signs when keepCount large.



4. Small w1 or w2

p1_delta very small → the w1 mapping is near-zero. Ensure no NaNs, and solution remains stable.



5. Pathological coefficients

Inputs with alternating large/small coefficients to test kLeadCoeffTol branch.



6. Constraint failures

Construct a case where constraints become overconstrained (C_eq.size() >= m) to ensure code correctly skips candidate and falls back.



7. Randomized Monte Carlo

Generate many random polynomials and verify that error decreases vs trivial baseline (e.g., constant) and that non-empty output is produced.




Each test should:

check coefficient finite-ness (std::isfinite),

evaluate objective error err computed as c^T ATA c - 2 ATb^T c + F2_const and confirm it is not NaN and is reasonable,

verify inequality constraints Ain * c <= bin + kFeasTol.



---

16. Example usage snippet

// degree = 3 (cubic)
int degree = 3;
float p1_coeffs[4] = {1.0f, 0.5f, -0.3f, 0.1f}; // p1(u) = 1 + 0.5u - 0.3u^2 + 0.1u^3
float p2_coeffs[4] = {0.8f, -0.2f, 0.05f, 0.0f};
double p1_delta = 0.4; // relative length for first piece
double p2_delta = 0.6;

auto res = myFitter.composePolynomials(p1_coeffs, p1_delta, p2_coeffs, p2_delta, degree);
if (res.empty()) {
    Serial.println("Fitting failed.");
} else {
    for (int i = 0; i <= degree; ++i) {
        Serial.printf("coef[%d] = %.12g\n", i, (double)res[i]);
    }
}


---

17. Recommendations for embedded / performance tuning

If compiling for microcontrollers with limited RAM:

Replace Eigen::MatrixXd/VectorXd with Eigen::Matrix<double, N, N> and Eigen::Matrix<double, N, 1> using a compile-time MAX_M.

Restrict candidate alpha set or keepCount range to reduce QP solves.

Do not use dd everywhere — keep dd only for ATb and F2 accumulation if necessary.

Precompute and reuse as many power tables as possible (already done for many loops).

Avoid eigenvalue companion matrix method on-board; if degree is small use closed-form or a stable polynomial root-finder suited for embedded platforms.


If you need determinism and speed over numerical optimality, prefer fewer alpha and keepCount combinations and a robust unconstrained least-squares as default.



---

18. Caveats / known limitations

The run-time scales poorly with polynomial degree due to cubic solves and repeated QP solves — not suited for very large degree on constrained hardware.

The companion-matrix root finder relies on Eigen and double precision; when lead is tiny the root-finder is skipped to avoid nonsense.

The heuristic selection of interior extrema and how many are enforced (keepCount) is empirical — may not be globally optimal for every problem.

dd must be present and correct; otherwise precision will degrade — if this is missing, replace dd with long double and test.



---

19. Debugging tips

Enable debug printing for run-time diagnostics by compiling with ARDUINO and initializing Serial.begin(...) before calling.

Add extra prints to display ATA trace, ATb norm, and early errors when qp_solve returns false to understand failure reasons.

Check ATA condition number; if extremely large, increase ridge or regularization alpha.

When qp_solve returns false, test intermediate solves: compute KKT.colPivHouseholderQr().solve(rhs) and inspect residuals.



---

20. Extension ideas

Provide an API flag to allow the caller to specify the set of alpha and keepCount values (instead of hard-coded).

Add logging hooks (callback) instead of DEBUG_PRINT for integration with test harnesses.

Implement a warm-start strategy for QP across alphas (use previous alpha's solution as initial guess).

Provide fixed-size template variant composePolynomialsFixed<N> for embedded systems.

Add a deterministic seed to candidate root ordering for reproducible results.



---

21. Quick checklist before shipping

Ensure robustSymmetricSolve is provided and returns well-conditioned results.

Confirm dd type and functions are linked and tested.

Decide whether to keep dynamic Eigen or convert to fixed-size for embedded targets.

Add unit tests (sections above) and run stress tests on pathological polynomials.



---

If you want, I can:

produce a formatted README-style Markdown file (ready for your repo),

generate a compact changelog/gist explaining the most important guardrails,

produce a fixed-size Eigen variant for a target maximum degree (e.g., degree ≤ 6),

or generate a small battery of unit tests in C++ (GoogleTest or tiny custom harness) that exercise the fallbacks and constraints.


Which of those would you like next?
