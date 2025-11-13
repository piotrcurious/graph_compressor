#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Arduino.h"
#include "multi_poly6a2_total_tidy_algebraic2_tidy3_lowmem3_newgraph2/AdvancedPolynomialFitter.hpp"

MockSerial Serial;

// Function to read data from a file into a vector of doubles
std::vector<double> read_data(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    double value;
    while (file >> value) {
        data.push_back(value);
    }
    return data;
}

// Function to read polynomial coefficients from a file
std::vector<float> read_coeffs(const std::string& filename) {
    std::vector<float> coeffs;
    std::ifstream file(filename);
    float value;
    while (file >> value) {
        coeffs.push_back(value);
    }
    return coeffs;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <command> [args...]" << std::endl;
        return 1;
    }

    std::string command = argv[1];

    if (command == "fit") {
        if (argc != 5) {
            std::cerr << "Usage: " << argv[0] << " fit <x_data_file> <y_data_file> <degree>" << std::endl;
            return 1;
        }
        std::vector<double> x = read_data(argv[2]);
        std::vector<float> y(x.size());
        std::vector<double> y_double = read_data(argv[3]);
        for(size_t i = 0; i < y_double.size(); ++i) {
            y[i] = y_double[i];
        }

        int degree = std::stoi(argv[4]);

        AdvancedPolynomialFitter fitter;
        std::vector<float> coeffs = fitter.fitPolynomialD(x, y, degree);

        for (float c : coeffs) {
            std::cout << c << std::endl;
        }
    } else if (command == "compose") {
        if (argc != 7) {
            std::cerr << "Usage: " << argv[0] << " compose <p1_coeffs_file> <p1_delta> <p2_coeffs_file> <p2_delta> <degree>" << std::endl;
            return 1;
        }
        std::vector<float> p1_coeffs = read_coeffs(argv[2]);
        double p1_delta = std::stod(argv[3]);
        std::vector<float> p2_coeffs = read_coeffs(argv[4]);
        double p2_delta = std::stod(argv[5]);
        int degree = std::stoi(argv[6]);

        AdvancedPolynomialFitter fitter;
        std::vector<float> coeffs = fitter.composePolynomials(p1_coeffs.data(), p1_delta, p2_coeffs.data(), p2_delta, degree);

        for (float c : coeffs) {
            std::cout << c << std::endl;
        }
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        return 1;
    }

    return 0;
}
