# Evolution of the `composePolynomials` Function

This document details the progression of the `composePolynomials` function, tracking its development from a basic implementation to a robust, feature-rich solution. Each version introduces new techniques to address challenges related to numerical stability, shape preservation, and constraint satisfaction.

### `composer.cpp`
The initial version implements `composePolynomials` using a projection onto a Legendre basis. While functional, this approach is susceptible to numerical precision issues, especially when dealing with ill-conditioned polynomials.

### `composer_3.cpp`
This version introduces a significant change in methodology, replacing the Legendre projection with a Quadratic Programming (QP) solver. This allows for the imposition of constraints to preserve the shape of the combined polynomial, such as enforcing minima and maxima.

### `composer_4.cpp`
Building on the QP solver, this version adds a candidate search mechanism. It explores different weighting schemes and constraint combinations to find an optimal fit that balances accuracy and shape preservation.

### `composer_5.cpp`
This version replaces the active-set QP solver with a more sophisticated primal-dual interior-point method. This change improves the robustness and accuracy of the solver, particularly for complex constraint sets.

### `composer_6.cpp`
A critical bug fix is introduced in this version to address overconstrained systems. Previously, the solver would fail if the number of constraints exceeded the number of variables. The fix ensures that such cases are handled gracefully, preventing crashes and improving stability.

### `composer_7.cpp`
This version adds iterative fallback logic to the QP solver. If the primary solver fails to find a solution, it systematically relaxes the constraints in a multi-stage process, increasing the likelihood of finding a valid, if less precise, solution.

### `composer_7b.cpp`
This version introduces several micro-optimizations and portability improvements. These changes enhance the function's performance and ensure its compatibility with different platforms, including Arduino.

### `composer_7b_0fbfix.cpp`
The final version incorporates a bug fix in the fallback logic, further improving the solver's reliability. This version represents the most advanced and stable implementation of `composePolynomials`, making it suitable for integration into the main project.
