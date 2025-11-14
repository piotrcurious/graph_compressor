#ifndef DATA_VISUALIZER_H
#define DATA_VISUALIZER_H

#include <TFT_eSPI.h>
#include "DataCompressor.hpp"

class DataVisualizer {
public:
    DataVisualizer(TFT_eSPI& tft, DataCompressor& compressor, const float (*rawData)[NUM_DATA_SERIES], const uint32_t* rawTimestamps, const uint16_t& rawDataIndex);

    void drawCompoundGraph(int rx, int ry, int rw, int rh, uint64_t windowEndAbs, uint32_t windowDurationMs);

private:
    TFT_eSPI& tft;
    DataCompressor& compressor;
    const float (*rawData)[NUM_DATA_SERIES];
    const uint32_t* rawTimestamps;
    const uint16_t& rawDataIndex;

    uint64_t getCompressedEndAbs() const;
    bool rawInterpolatedValueAt(uint64_t timestamp, uint8_t seriesIndex, double& outValue) const;
    bool compressedValueAt_inRange(uint64_t ts, uint8_t seriesIndex, double& outValue) const;
    void computeWindowMinMaxDirect(uint64_t wStart, uint64_t wEnd, float& outMin, float& outMax) const;
    int buildPolyStarts(uint64_t starts[], int maxStarts) const;

    static inline double evaluatePolynomialNormalized(const float* coefficients, uint8_t degree, double tNorm);
};

#endif // DATA_VISUALIZER_H