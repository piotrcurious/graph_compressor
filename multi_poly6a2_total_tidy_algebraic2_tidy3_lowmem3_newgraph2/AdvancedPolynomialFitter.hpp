#ifndef ADVANCED_POLYNOMIAL_FITTER_H
#define ADVANCED_POLYNOMIAL_FITTER_H

#include <Arduino.h>
#if __has_include(<ArduinoEigenDense.h>)
#include <ArduinoEigenDense.h>
#else
#include <Eigen/Dense>
#endif
#include <vector>
#if __has_include(<ArduinoEigen.h>)
#include <ArduinoEigen.h>
#endif

// Constants for composePolynomials
namespace {
    const double kEps = 1e-15;
    const double kRootTol = 1e-12;
    const double kImagTol = 1e-9;
    const double kLeadCoeffTol = 1e-30;
    const double kFuzzyBoundary = 1e-15;
    const double kFeasTol = 1e-8;
}

// Debug print macro
#if defined(ARDUINO)
#include <Arduino.h>
#define DEBUG_PRINT(...) do { if (Serial) Serial.printf(__VA_ARGS__); } while(0)
#else
#include <stdio.h>
#define DEBUG_PRINT(...) do { printf(__VA_ARGS__); } while(0)
#endif

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


    std::vector<float> fitPolynomial_lowmem(const std::vector<float>& x,
                                                                   const std::vector<float>& y,
                                                                   int degree,
                                                                   double ridge);
    std::vector<float> fitPolynomialD_lowmem(const std::vector<double>& x,
                                                                    const std::vector<float>& y,
                                                                    int degree,
                                                                    double ridge);                                                               

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
