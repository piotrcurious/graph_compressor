Notes, recommendations and caveats
This is exact (modulo FP rounding) for the continuous L² projection of the piecewise-linear interpolant onto polynomials. No iterative linear solves required.
For numerical robustness I used your dd accumulators where we compute integrals and final monomial coefs.
Complexity:
Integral computation cost is � to produce Iy up to power �, plus � arithmetic to form inner products & expand basis. This is typically cheaper and more stable than solving a potentially ill-conditioned normal matrix when degree is small relative to n.
Integration domain mapping: I used shifted Legendre polynomials on � and mapped �. If you prefer mapping to � or using Chebyshev basis, the same scheme applies with different monomial coefficient matrices and norms.
If you want the weighted L² projection (e.g. IRLS weights), you can still use an analogous analytic approach only if the weight is piecewise-constant (per interval). Then you compute weighted Iy_j with weight multiplied into interval integrals and proceed identical. For arbitrary continuous weight � you need to compute weight-armored integrals � — but if � is known analytically (e.g. polynomial or piecewise-constant) it remains analytical.
Integration of very high powers may overflow — you already guard with dd; consider scaling and centering domain to reduce dynamic range (map to [0,1] already helps).
