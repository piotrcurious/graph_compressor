#ifndef DATA_COMPRESSOR_H
#define DATA_COMPRESSOR_H

#include <Arduino.h>
#include <vector>
#include "AdvancedPolynomialFitter.hpp"

// =================================================================================================
// Configuration Settings
// These macros define the structure and behavior of the data compression.
// =================================================================================================

// Uncomment the following line to use the low-memory polynomial fitter.
// This is useful for memory-constrained devices, but may have a slight impact on accuracy.
// #define USE_LOW_MEMORY_FITTER

#define NUM_DATA_SERIES 2           // The number of data series to be processed.
#define POLY_COUNT 8                // Number of polynomials in each segment.
#define SEGMENTS 2                  // Total number of segments in the circular buffer.
#define POLY_DEGREE 5               // The degree of the polynomials used for storing compressed data.
#define SUB_FIT_POLY_DEGREE 3       // Degree for the supplementary fitter, used to improve boundary transitions.
#define BOUNDARY_MARGIN3 5          // Margin for the sub-fitter, helps in creating a smoother fit.
#define BOUNDARY_DELTA3 10          // Time window of the margin for the sub-fitter.
#define BOUNDARY_MARGIN 5           // Margin for the main fitter.
#define BOUNDARY_DELTA 10           // Time window of the margin for the main fitter.
#define LOG_BUFFER_POINTS_PER_POLY 60 // Number of raw data points to accumulate before fitting a new polynomial.
#define TIME_PRECISION_DIVIDER 10     // Divisor to reduce the precision of time deltas.

// =================================================================================================
// Data Structures
// =================================================================================================

// Represents a segment of the compressed data, containing multiple polynomials.
struct PolynomialSegment {
    float coefficients[POLY_COUNT][NUM_DATA_SERIES][POLY_DEGREE + 1]; // Coefficients for each polynomial.
    uint16_t timeDeltas[POLY_COUNT];                   // Time duration of each polynomial in milliseconds.
};

// =================================================================================================
// DataCompressor Class
// =================================================================================================

/**
 * @class DataCompressor
 * @brief Encapsulates the logic for compressing time-series data into polynomial segments.
 *
 * This class manages the process of receiving raw data points, buffering them,
 * fitting them to polynomials, and storing these polynomials in a circular buffer of segments.
 * It also handles the recompression of older segments to free up space.
 */
class DataCompressor {
public:
    /**
     * @brief Constructor for the DataCompressor class.
     */
    DataCompressor();

    /**
     * @brief Logs a new data sample.
     * When enough samples have been collected, it triggers the polynomial fitting process.
     * @param data An array of float values representing the data samples.
     * @param currentTimestamp The timestamp of the data sample in milliseconds.
     */
    void logSampledData(const float* data, uint32_t currentTimestamp);

    // --- Getters for accessing compressed data ---
    // These are primarily used for visualization and debugging.

    const PolynomialSegment* getSegmentBuffer() const { return segmentBuffer; }
    uint8_t getSegmentCount() const { return segmentCount; }
    uint16_t getCurrentPolyIndex() const { return currentPolyIndex; }
    uint32_t getRawLogDelta() const { return raw_log_delta; }
    uint32_t getLastTimestamp() const { return lastTimestamp; }
    uint8_t getRolloverCount() const { return rollover_count; }


private:
    // --- Internal methods for data management ---

    void addSegment(const PolynomialSegment& newSegment);
    bool isBufferFull() const;
    void getOldestSegments(PolynomialSegment& oldest, PolynomialSegment& secondOldest) const;
    void removeOldestTwo();
    void processBufferedData();
    void compressDataToSegment(uint8_t seriesIndex, const uint32_t* timestamps, uint16_t dataSize, float* coefficients, uint32_t& timeDelta);
    void combinePolynomials(const PolynomialSegment& oldest, const PolynomialSegment& secondOldest, PolynomialSegment& recompressedSegment);
    void recompressSegments();

    // --- Member Variables ---

    PolynomialSegment segmentBuffer[SEGMENTS]; // Circular buffer of polynomial segments.
    uint8_t segmentCount;                      // The number of active segments in the buffer.
    uint16_t currentPolyIndex;                 // The index of the current polynomial being written to in the latest segment.
    uint32_t lastTimestamp;                    // The timestamp of the last logged data point.
    uint32_t raw_log_delta;                    // Time delta since the last polynomial was created.
    uint8_t head;                              // The index of the oldest segment in the circular buffer.
    uint8_t rollover_count;                    // Counter for timestamp rollovers.

    // Buffer for accumulating incoming raw data before fitting.
    float rawDataBuffer[LOG_BUFFER_POINTS_PER_POLY][NUM_DATA_SERIES];
    uint32_t timestampsBuffer[LOG_BUFFER_POINTS_PER_POLY];
    uint16_t dataIndex;
    float lastDataPoint[NUM_DATA_SERIES];
};

#endif // DATA_COMPRESSOR_H