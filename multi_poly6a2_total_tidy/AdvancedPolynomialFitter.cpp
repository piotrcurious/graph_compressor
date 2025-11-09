#include "AdvancedPolynomialFitter.hpp"
#include <algorithm>
#include <cmath>
//#include <ArduinoEigenDense.h>

using namespace Eigen;

    // calculate using Welford's method
    double AdvancedPolynomialFitter::calculateMSE(const std::vector<float>& coeffs, 
                   const std::vector<float>& x, 
                   const std::vector<float>& y) {
    double meanSquaredError = 0.0;
    double mean = 0.0;
    double M2 = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        float prediction = 0.0;
        for (size_t j = 0; j < coeffs.size(); ++j) {
            prediction += coeffs[j] * pow(x[i], j);
        }
        float error = prediction - y[i];
        double squaredError = error * error;
        double delta = squaredError - mean;
        mean += delta / (i + 1);
        M2 += delta * (squaredError - mean);
    }

    meanSquaredError = mean;
    //Serial.println(meanSquaredError);
    return meanSquaredError;
}
//version with X in double domain
    double AdvancedPolynomialFitter::calculateMSED(const std::vector<float>& coeffs, 
                   const std::vector<double>& x, 
                   const std::vector<float>& y) {
    double meanSquaredError = 0.0;
    double mean = 0.0;
    double M2 = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        float prediction = 0.0;
        for (size_t j = 0; j < coeffs.size(); ++j) {
            prediction += coeffs[j] * pow(x[i], j);
        }
        float error = prediction - y[i];
        double squaredError = error * error;
        double delta = squaredError - mean;
        mean += delta / (i + 1);
        M2 += delta * (squaredError - mean);
    }

    meanSquaredError = mean;
    //Serial.println(meanSquaredError);
    return meanSquaredError;
}


    // Fit a polynomial to the data using the normal equation
    std::vector<float> AdvancedPolynomialFitter::fitPolynomial(const std::vector<float>& x, const std::vector<float>& y, int degree,
    OptimizationMethod method ) {
        if (x.size() != y.size() || x.empty() || degree < 1) {
            return {};  // Invalid input
        }
        Serial.println(x[0]); Serial.println(x[x.size()-1]);

        size_t n = x.size();size_t m = degree + 1;

        // Construct the Vandermonde matrix
        std::vector<std::vector<double>> A(n, std::vector<double>(m, 0.0));
        for (size_t i = 0; i < n; ++i) {
            double xi = 1.0;
            for (size_t j = 0; j < m; ++j) {
                A[i][j] = xi;
                xi *= x[i];
            }
        }

        // Construct the normal equation: (A^T * A) * coeffs = A^T * y
        std::vector<std::vector<double>> ATA(m, std::vector<double>(m, 0.0));
        std::vector<double> ATy(m, 0.0);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                ATy[j] += A[i][j] * y[i];
                for (size_t k = 0; k < m; ++k) {
                    ATA[j][k] += A[i][j] * A[i][k];
                }
            }
        }

        // Solve the normal equation using Gaussian elimination
        std::vector<double> coeffs = solveLinearSystem(ATA, ATy);

        // Convert coefficients to float
        std::vector<float> result(coeffs.begin(), coeffs.end());

        switch (method) {
        case GRADIENT_DESCENT:
            // Implement gradient descent here if needed
            break;
        case LEVENBERG_MARQUARDT:
            result = levenbergMarquardt(result, x, y, degree);
            break;
        case NELDER_MEAD:
            // Implement Nelder-Mead here if needed
            break;
        default: // no optimization
            //result = levenbergMarquardt(x_norm, y, degree);
            break;
        } // switch(method) {
        return result;
    }
//----------- double precission timestamps
    // Fit a polynomial to the data using the normal equation
    std::vector<float> AdvancedPolynomialFitter::fitPolynomialD(const std::vector<double>& x, const std::vector<float>& y, int degree,
    OptimizationMethod method ) {
        if (x.size() != y.size() || x.empty() || degree < 1) {
            return {};  // Invalid input
        }
        Serial.println(x[0]); Serial.println(x[x.size()-1]);

        size_t n = x.size();size_t m = degree + 1;

        // Construct the Vandermonde matrix
        std::vector<std::vector<double>> A(n, std::vector<double>(m, 0.0));
        for (size_t i = 0; i < n; ++i) {
            double xi = 1.0;
            for (size_t j = 0; j < m; ++j) {
                A[i][j] = xi;
                xi *= x[i];
            }
        }

        // Construct the normal equation: (A^T * A) * coeffs = A^T * y
        std::vector<std::vector<double>> ATA(m, std::vector<double>(m, 0.0));
        std::vector<double> ATy(m, 0.0);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                ATy[j] += A[i][j] * y[i];
                for (size_t k = 0; k < m; ++k) {
                    ATA[j][k] += A[i][j] * A[i][k];
                }
            }
        }

        // Solve the normal equation using Gaussian elimination
        std::vector<double> coeffs = solveLinearSystem(ATA, ATy);

        // Convert coefficients to float
        std::vector<float> result(coeffs.begin(), coeffs.end());

        switch (method) {
        case GRADIENT_DESCENT:
            // Implement gradient descent here if needed
            break;
        case LEVENBERG_MARQUARDT:
            result = levenbergMarquardtD(result, x, y, degree);  // TODO : double timestamp precission optimization method
            break;
        case NELDER_MEAD:
            // Implement Nelder-Mead here if needed
            break;
        default: // no optimization
            //result = levenbergMarquardt(x_norm, y, degree);
            break;
        } // switch(method) {
        return result;
    }


// -------------TOTAL least SQUARES 

    std::vector<float> AdvancedPolynomialFitter::fitPolynomialD_superpos5c(const std::vector<double>& x, const std::vector<float>& y, int degree,
    OptimizationMethod method ) {

        const size_t n = x.size();
        const size_t m = degree + 1;

        // Use Eigen or Armadillo for more robust linear algebra if possible
        std::vector<std::vector<double>> ATA(m, std::vector<double>(m, 0.0));
        std::vector<double> ATy(m, 0.0);

        // Precompute powers to avoid repeated multiplication
        std::vector<std::vector<double>> xPowers(n, std::vector<double>(m, 1.0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 1; j < m; ++j) {
                xPowers[i][j] = xPowers[i][j-1] * x[i];
            }
        }

        // Compute A^T * A and A^T * y
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                ATy[j] += xPowers[i][j] * y[i];
                for (size_t k = 0; k <= j; ++k) {
                    ATA[j][k] += xPowers[i][j] * xPowers[i][k];
                }
            }
        }

        // Fill symmetric matrix
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = j + 1; k < m; ++k) {
                ATA[j][k] = ATA[k][j];
            }
        }

          std::vector<double> coeffs = solveQR(ATA, ATy);
          // Convert coefficients to float
          std::vector<float> result(coeffs.begin(), coeffs.end());
          return result; 
    }


// QR decomposition solver (previous implementation remains largely the same)
   std::vector<double> AdvancedPolynomialFitter::solveQR(
        std::vector<std::vector<double>>& A, 
        std::vector<double>& b
    ) {
        const size_t n = A.size();
        const double eps = std::numeric_limits<double>::epsilon();

        // Householder QR decomposition
        for (size_t k = 0; k < n; ++k) {
            // Compute column norm with numerical stability
            double norm_x = 0.0;
            for (size_t i = k; i < n; ++i) {
                norm_x += A[i][k] * A[i][k];
            }
            norm_x = std::sqrt(std::max(norm_x, eps));

            // Avoid potential overflow/underflow
            double alpha = (A[k][k] > 0) ? -norm_x : norm_x;
            double r = std::sqrt(std::max(0.5 * (alpha * alpha - A[k][k] * alpha), eps));

            std::vector<double> v(n, 0.0);
            v[k] = (A[k][k] - alpha) / (2 * r);
            for (size_t i = k + 1; i < n; ++i) {
                v[i] = A[i][k] / (2 * r);
            }

            // Apply Householder transformation to A and b
            for (size_t j = k; j < n; ++j) {
                double dot = 0.0;
                for (size_t i = k; i < n; ++i) {
                    dot += v[i] * A[i][j];
                }
                for (size_t i = k; i < n; ++i) {
                    A[i][j] -= 2 * v[i] * dot;
                }
            }

            double dot_b = 0.0;
            for (size_t i = k; i < n; ++i) {
                dot_b += v[i] * b[i];
            }
            for (size_t i = k; i < n; ++i) {
                b[i] -= 2 * v[i] * dot_b;
            }

            A[k][k] = alpha;
            for (size_t i = k + 1; i < n; ++i) {
                A[i][k] = 0.0;
            }

            
        }

        // Back substitution with added numerical checks
        std::vector<double> x(n);
        for (int i = n - 1; i >= 0; --i) {
            x[i] = b[i];
            for (size_t j = i + 1; j < n; ++j) {
                x[i] -= A[i][j] * x[j];
            }
            // Add a small check to prevent division by near-zero
            x[i] /= (std::abs(A[i][i]) > eps) ? A[i][i] : eps;
        }

        return x;
    }




    

//------------------------fitter method with built in normalization from 0 to maxX
    // Fit a polynomial to the data using the normal equation
    std::vector<float> AdvancedPolynomialFitter::NormalizeAndFitPolynomial(const std::vector<float>& x, const std::vector<float>& y, int degree,
    OptimizationMethod method ) {
        if (x.size() != y.size() || x.empty() || degree < 1) {
            return {};  // Invalid input
        }

       // Normalize x and y
        std::vector<float> x_norm = x;
        double x_min = *std::min_element(x.begin(), x.end());
        Serial.print("xmin:");
        Serial.print(x_min);
        double x_max = *std::max_element(x.begin(), x.end());
        Serial.print(" xmax:");
        Serial.println(x_max);

#ifdef REVERSED_NORMALIZATION       
        std::transform(x.begin(), x.end(), x_norm.begin(), [x_max](double val) { return val -x_max; });
#else  
        std::transform(x.begin(), x.end(), x_norm.begin(), [x_min](double val) { return val -x_min; });

#endif //#ifdef REVERSED_NORMALIZATION       

       // double y_max = *std::max_element(y.begin(), y.end());
//        std::transform(y.begin(), y.end(), y_norm.begin(), [y_max](double val) { return val / y_max; });

        Serial.println(x_norm[0]); 
        Serial.println(x_norm[x_norm.size()-1]);

        size_t n = x_norm.size();
        size_t m = degree + 1;

        // Construct the Vandermonde matrix
        std::vector<std::vector<double>> A(n, std::vector<double>(m, 0.0));
        for (size_t i = 0; i < n; ++i) {
            double xi = 1.0;
            for (size_t j = 0; j < m; ++j) {
                A[i][j] = xi;
                xi *= x_norm[i];
            }
        }

        // Construct the normal equation: (A^T * A) * coeffs = A^T * y
        std::vector<std::vector<double>> ATA(m, std::vector<double>(m, 0.0));
        std::vector<double> ATy(m, 0.0);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                ATy[j] += A[i][j] * y[i];
                for (size_t k = 0; k < m; ++k) {
                    ATA[j][k] += A[i][j] * A[i][k];
                }
            }
        }

        // Solve the normal equation using Gaussian elimination
        std::vector<double> coeffs = solveLinearSystem(ATA, ATy);

        // Convert coefficients to float
        std::vector<float> result(coeffs.begin(), coeffs.end());

        switch (method) {
        case GRADIENT_DESCENT:
            // Implement gradient descent here if needed
            break;
        case LEVENBERG_MARQUARDT:
            result = levenbergMarquardt(result, x_norm, y, degree);
            break;
        case NELDER_MEAD:
            // Implement Nelder-Mead here if needed
            break;
        default: // no optimization
            //result = levenbergMarquardt(x_norm, y, degree);
            break;
    }
        
        return result;
    }

// Fit segmented polynomials
std::vector<float> AdvancedPolynomialFitter::fitSegmentedPolynomials(const std::vector<float>& x, const std::vector<float>& y, int degree, int segments) {
    std::vector<float> result;
    size_t segmentSize = x.size() / segments;
    
    for (int i = 0; i < segments; ++i) {
        size_t startIdx = i * segmentSize;
        size_t endIdx = (i == segments - 1) ? x.size() : (i + 1) * segmentSize;
        
        std::vector<float> x_segment(x.begin() + startIdx, x.begin() + endIdx);
        std::vector<float> y_segment(y.begin() + startIdx, y.begin() + endIdx);
        
        std::vector<float> segmentCoeffs = fitPolynomial(x_segment, y_segment, degree);
        result.insert(result.end(), segmentCoeffs.begin(), segmentCoeffs.end());
    }
    
    return result;
}

// Implement Levenberg-Marquardt algorithm
std::vector<float> AdvancedPolynomialFitter::levenbergMarquardt(std::vector<float>& coeffs,const std::vector<float>& x, const std::vector<float>& y, int degree) {
    const int maxIterations = 200; // Maximum number of iterations
    const double lambdaInit = 0.1; // Initial lambda value
    const double lambdaFactor = 2; // Factor to increase/decrease lambda
    const double tolerance = 1e-6; // Convergence tolerance

    //std::vector<float> coeffs(degree + 1, 0.0); // Initial coefficients
    double lambda = lambdaInit;
    double prevMSE = calculateMSE(coeffs, x, y);

    for (int iter = 0; iter < maxIterations; ++iter) {
        // Compute the Jacobian matrix and residuals
        std::vector<std::vector<double>> J(x.size(), std::vector<double>(degree + 1, 0.0));
        std::vector<double> residuals(x.size(), 0.0);
        
        for (size_t i = 0; i < x.size(); ++i) {
            double xi = 1.0;
            for (int j = 0; j <= degree; ++j) {
                J[i][j] = xi;
                xi *= x[i];
            }
            double prediction = 0.0;
            for (int j = 0; j <= degree; ++j) {
                prediction += coeffs[j] * pow(x[i], j);
            }
            residuals[i] = y[i] - prediction;
        }

        // Compute the normal equations
        std::vector<std::vector<double>> JTJ(degree + 1, std::vector<double>(degree + 1, 0.0));
        std::vector<double> JTr(degree + 1, 0.0);
        
        for (size_t i = 0; i < x.size(); ++i) {
            for (int j = 0; j <= degree; ++j) {
                for (int k = 0; k <= degree; ++k) {
                    JTJ[j][k] += J[i][j] * J[i][k];
                }
                JTr[j] += J[i][j] * residuals[i];
            }
        }

        // Add the damping factor to the diagonal elements
        for (int j = 0; j <= degree; ++j) {
            JTJ[j][j] += lambda;
        }

        // Solve for the parameter update
        std::vector<double> delta = solveLinearSystem(JTJ, JTr);

        // Update the coefficients
        std::vector<float> newCoeffs = coeffs;
        for (int j = 0; j <= degree; ++j) {
            newCoeffs[j] += delta[j];
        }

        double newMSE = calculateMSE(newCoeffs, x, y);
        
        // Check for convergence
        if (std::abs(prevMSE - newMSE) < tolerance) {
            break;
        }

        // Update lambda and coefficients based on the new MSE
        if (newMSE < prevMSE) {
            lambda /= lambdaFactor;
            Serial.println(lambda);
            coeffs = newCoeffs;
            prevMSE = newMSE;
        } else {
            lambda *= lambdaFactor;
        }
    }

    return coeffs;
}

// Implement Levenberg-Marquardt algorithm (doubles in timestamps)
std::vector<float> AdvancedPolynomialFitter::levenbergMarquardtD(std::vector<float>& coeffs,const std::vector<double>& x, const std::vector<float>& y, int degree) {
    const int maxIterations = 100; // Maximum number of iterations
    const double lambdaInit = 0.1; // Initial lambda value
    const double lambdaFactor = 2; // Factor to increase/decrease lambda
    const double tolerance = 1e-6; // Convergence tolerance

    //std::vector<float> coeffs(degree + 1, 0.0); // Initial coefficients
    double lambda = lambdaInit;
    double prevMSE = calculateMSED(coeffs, x, y);

    for (int iter = 0; iter < maxIterations; ++iter) {
        // Compute the Jacobian matrix and residuals
        std::vector<std::vector<double>> J(x.size(), std::vector<double>(degree + 1, 0.0));
        std::vector<double> residuals(x.size(), 0.0);
        
        for (size_t i = 0; i < x.size(); ++i) {
            double xi = 1.0;
            for (int j = 0; j <= degree; ++j) {
                J[i][j] = xi;
                xi *= x[i];
            }
            double prediction = 0.0;
            for (int j = 0; j <= degree; ++j) {
                prediction += coeffs[j] * pow(x[i], j);
            }
            residuals[i] = y[i] - prediction;
        }

        // Compute the normal equations
        std::vector<std::vector<double>> JTJ(degree + 1, std::vector<double>(degree + 1, 0.0));
        std::vector<double> JTr(degree + 1, 0.0);
        
        for (size_t i = 0; i < x.size(); ++i) {
            for (int j = 0; j <= degree; ++j) {
                for (int k = 0; k <= degree; ++k) {
                    JTJ[j][k] += J[i][j] * J[i][k];
                }
                JTr[j] += J[i][j] * residuals[i];
            }
        }

        // Add the damping factor to the diagonal elements
        for (int j = 0; j <= degree; ++j) {
            JTJ[j][j] += lambda;
        }

        // Solve for the parameter update
        std::vector<double> delta = solveLinearSystem(JTJ, JTr);

        // Update the coefficients
        std::vector<float> newCoeffs = coeffs;
        for (int j = 0; j <= degree; ++j) {
            newCoeffs[j] += delta[j];
        }

        double newMSE = calculateMSED(newCoeffs, x, y);
        
        // Check for convergence
        if (std::abs(prevMSE - newMSE) < tolerance) {
            break;
        }

        // Update lambda and coefficients based on the new MSE
        if (newMSE < prevMSE) {
            lambda /= lambdaFactor;
            Serial.println(lambda);
            coeffs = newCoeffs;
            prevMSE = newMSE;
        } else {
            lambda *= lambdaFactor;
        }
    }

    return coeffs;
}


    // Solve a linear system using Gaussian elimination
    std::vector<double> AdvancedPolynomialFitter::solveLinearSystem(std::vector<std::vector<double>>& A, std::vector<double>& b) {
        size_t n = A.size();

        // Forward elimination
        for (size_t k = 0; k < n; ++k) {
            // Pivot for numerical stability
            for (size_t i = k + 1; i < n; ++i) {
                if (fabs(A[i][k]) > fabs(A[k][k])) {
                    std::swap(A[k], A[i]);
                    std::swap(b[k], b[i]);
                }
            }

            for (size_t i = k + 1; i < n; ++i) {
                double factor = A[i][k] / A[k][k];
                for (size_t j = k; j < n; ++j) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
            }
        }

        // Back substitution
        std::vector<double> x(n, 0.0);
        for (int i = n - 1; i >= 0; --i) {
            x[i] = b[i];
            for (size_t j = i + 1; j < n; ++j) {
                x[i] -= A[i][j] * x[j];
            }
            x[i] /= A[i][i];
        }

        return x;
    }
