#ifndef DATA_VISUALIZER_H
#define DATA_VISUALIZER_H

#include <TFT_eSPI.h>
#include "DataCompressor.hpp"

class DataVisualizer {
public:
    DataVisualizer(TFT_eSPI& tft, DataCompressor& compressor, const float* rawData, const uint32_t* rawTimestamps, const uint16_t& rawDataIndex);

    void drawCompoundGraph(int rx, int ry, int rw, int rh, uint32_t windowEndAbs, uint32_t windowDurationMs);

private:
    TFT_eSPI& tft;
    DataCompressor& compressor;
    const float* rawData;
    const uint32_t* rawTimestamps;
    const uint16_t& rawDataIndex;

    uint32_t getCompressedEndAbs() const;
    bool rawInterpolatedValueAt(uint32_t timestamp, double& outValue) const;
    bool compressedValueAt_inRange(uint32_t ts, double& outValue) const;
    void computeWindowMinMaxDirect(uint32_t wStart, uint32_t wEnd, float& outMin, float& outMax) const;
    int buildPolyStarts(uint32_t starts[], int maxStarts) const;

    static inline double evaluatePolynomialNormalized(const float* coefficients, uint8_t degree, double tNorm);
};

#endif // DATA_VISUALIZER_H