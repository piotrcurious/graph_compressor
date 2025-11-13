# C++ Test Suite for Polynomial Fitter

This directory contains a Linux-based test suite for the `AdvancedPolynomialFitter` library, featuring a Tcl/Tk graphical frontend and Gnuplot for visualization.

## Prerequisites

- A C++ compiler (e.g., g++)
- The Eigen3 library (`sudo apt-get install libeigen3-dev`)
- Tcl/Tk (`sudo apt-get install tcl tk`)
- Gnuplot (`sudo apt-get install gnuplot`)

## Building the Test Harness

A `Makefile` is provided in the parent directory to build the C++ test harness. To compile the code, simply run `make` from the parent directory:

```bash
make
```

This will produce an executable file named `run_test`.

## Running the Test Suite

To launch the interactive test suite, run the `test_frontend.tcl` script:

```bash
./test_frontend.tcl
```

### Testing `fitPolynomial`

1.  **X Data File:** Enter the path to a file containing space-separated X-axis data points.
2.  **Y Data File:** Enter the path to a file containing space-separated Y-axis data points.
3.  **Degree:** Enter the desired degree of the polynomial to fit.
4.  Click **Run Fit**.

The C++ harness will be executed, and the resulting polynomial coefficients will be displayed in the output window. A Gnuplot window will also appear, showing a plot of the original data points and the fitted polynomial curve.

### Testing `composePolynomials`

1.  **P1 Coeffs File:** Enter the path to a file containing the coefficients of the first polynomial.
2.  **P1 Delta:** Enter the delta value for the first polynomial.
3.  **P2 Coeffs File:** Enter the path to a file containing the coefficients of the second polynomial.
4.  **P2 Delta:** Enter the delta value for the second polynomial.
5.  **Degree:** Enter the degree of the polynomials.
6.  Click **Run Compose**.

The C++ harness will compute the composed polynomial, and its coefficients will be displayed in the output window. A Gnuplot window will appear, showing plots of the two original polynomials and the final composed polynomial.

## Sample Data

Sample data files (`x_data.dat`, `y_data.dat`, `p1_coeffs.dat`, `p2_coeffs.dat`) are provided in the parent directory to demonstrate the functionality of the test suite.
