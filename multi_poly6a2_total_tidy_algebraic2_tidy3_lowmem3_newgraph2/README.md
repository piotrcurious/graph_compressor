# Arduino Data Compressor

This Arduino project provides a library for compressing time-series data into a series of polynomial functions. It's designed to be easily included in other Arduino projects for efficient data logging and visualization.

## Features

- **Data Compression:** Compresses raw time-series data into a more compact format using polynomial fitting.
- **Portable:** The core logic is encapsulated in `DataCompressor` and `AdvancedPolynomialFitter` classes, which can be easily included in any Arduino sketch.
- **Optional Low-Memory Fitter:** Includes an alternative polynomial fitter that is optimized for low-memory environments. This can be enabled with a preprocessor directive.
- **Example Sketch:** A basic usage example is provided in the main `.ino` file to demonstrate how to use the library.

## How to Use

1.  **Include the Library:** Copy the `AdvancedPolynomialFitter.hpp`, `AdvancedPolynomialFitter.cpp`, `DataCompressor.hpp`, and `DataCompressor.cpp` files into your Arduino sketch folder.
2.  **Instantiate the Compressor:** Create an instance of the `DataCompressor` class in your sketch:
    ```cpp
    #include "DataCompressor.hpp"
    DataCompressor dataCompressor;
    ```
3.  **Log Data:** In your `loop()` function, log your sampled data by calling the `logSampledData` method:
    ```cpp
    float sampledData = sampleScalarData(currentTimestamp);
    dataCompressor.logSampledData(sampledData, currentTimestamp);
    ```
4.  **Access Compressed Data:** The compressed data is stored in a buffer of `PolynomialSegment` structs, which can be accessed for visualization or storage. The provided example sketch demonstrates how to draw the compressed data to a TFT display.

## Low-Memory Fitter

For memory-constrained devices, an alternative low-memory polynomial fitter is available. To use it, open `DataCompressor.hpp` and uncomment the following line:

```cpp
// #define USE_LOW_MEMORY_FITTER
```

This will switch the polynomial fitting algorithm to a version that uses less memory, with a potential trade-off in accuracy.
