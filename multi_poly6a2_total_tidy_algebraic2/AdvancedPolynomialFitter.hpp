#ifndef ADVANCED_POLYNOMIAL_FITTER_H
#define ADVANCED_POLYNOMIAL_FITTER_H

#include <Arduino.h>
//#include <Eigen/Dense>
#include <ArduinoEigenDense.h>
#include <vector>
#include <ArduinoEigen.h> 

class AdvancedPolynomialFitter {
public:
    enum OptimizationMethod {
        GRADIENT_DESCENT,
        LEVENBERG_MARQUARDT,
        NELDER_MEAD,
        NONE,
    };

    double calculateMSE(const std::vector<float>& coeffs, const std::vector<float>& x, const std::vector<float>& y);
    double calculateMSED(const std::vector<float>& coeffs, const std::vector<double>& x, const std::vector<float>& y);

    std::vector<float> fitPolynomial(const std::vector<float>& x, const std::vector<float>& y, int degree,
                                     OptimizationMethod method = GRADIENT_DESCENT);
    std::vector<float> fitPolynomialD(const std::vector<double>& x, const std::vector<float>& y, int degree,
                                     OptimizationMethod method = GRADIENT_DESCENT);
    std::vector<float> fitPolynomialD_superpos5c(const std::vector<double>& x, const std::vector<float>& y, int degree,
                                     OptimizationMethod method = GRADIENT_DESCENT);


    std::vector<float> NormalizeAndFitPolynomial(const std::vector<float>& x, const std::vector<float>& y, int degree,
                                     OptimizationMethod method = GRADIENT_DESCENT);
                                                                      
    std::vector<float> fitSegmentedPolynomials(const std::vector<float>& x, const std::vector<float>& y, int degree, int segments);
    std::vector<float> levenbergMarquardt(std::vector<float>& coeffs, const std::vector<float>& x, const std::vector<float>& y, int degree);
    std::vector<float> levenbergMarquardtD(std::vector<float>& coeffs, const std::vector<double>& x, const std::vector<float>& y, int degree);

    std::vector<float> composePolynomials(const float* p1_coeffs, double p1_delta, const float* p2_coeffs, double p2_delta, int degree);

private:
    std::vector<double> solveLinearSystem(std::vector<std::vector<double>>& A, std::vector<double>& b);
    std::vector<double> solveQR(std::vector<std::vector<double>>& A, std::vector<double>& b); 
    
};

#endif
