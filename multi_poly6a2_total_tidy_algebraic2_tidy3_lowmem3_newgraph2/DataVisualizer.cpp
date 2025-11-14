#include "DataVisualizer.hpp"

// =================================================================================================
// Constructor
// =================================================================================================

DataVisualizer::DataVisualizer(TFT_eSPI& tft, DataCompressor& compressor, const float (*rawData)[NUM_DATA_SERIES], const uint32_t* rawTimestamps, const uint16_t& rawDataIndex)
    : tft(tft), compressor(compressor), rawData(rawData), rawTimestamps(rawTimestamps), rawDataIndex(rawDataIndex) {}

// =================================================================================================
// Private Helper Functions
// =================================================================================================

inline double DataVisualizer::evaluatePolynomialNormalized(const float* coefficients, uint8_t degree, double tNorm) {
    double result = 0.0;
    double tPower = 1.0;
    for (uint8_t i = 0; i < degree; ++i) {
        result += (double)coefficients[i] * tPower;
        tPower *= tNorm;
    }
    return result;
}

uint64_t DataVisualizer::getCompressedEndAbs() const {
    uint64_t last_ts = compressor.getLastTimestamp();
    uint64_t rollovers = compressor.getRolloverCount();
    uint64_t total_millis = (rollovers << 32) | last_ts;
    return total_millis - compressor.getRawLogDelta();
}

bool DataVisualizer::rawInterpolatedValueAt(uint64_t timestamp, uint8_t seriesIndex, double &outValue) const {
    if (rawDataIndex == 0) return false;

    uint64_t current_rollovers = compressor.getRolloverCount();
    uint64_t first_ts = (current_rollovers << 32) | rawTimestamps[0];
    uint64_t last_ts = (current_rollovers << 32) | rawTimestamps[rawDataIndex - 1];

    if (timestamp <= first_ts) { outValue = rawData[0][seriesIndex]; return true; }
    if (timestamp >= last_ts) { outValue = rawData[rawDataIndex - 1][seriesIndex]; return true; }

    int lo = 0, hi = (int)rawDataIndex - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint64_t tm = (current_rollovers << 32) | rawTimestamps[mid];
        if (tm == timestamp) { outValue = rawData[mid][seriesIndex]; return true; }
        if (tm < timestamp) lo = mid + 1; else hi = mid - 1;
    }
    if (hi >= 0 && hi + 1 < rawDataIndex) {
        uint64_t t0 = (current_rollovers << 32) | rawTimestamps[hi];
        uint64_t t1 = (current_rollovers << 32) | rawTimestamps[hi + 1];
        double frac = (t1 == t0) ? 0.0 : double(timestamp - t0) / double(t1 - t0);
        outValue = rawData[hi][seriesIndex] * (1.0 - frac) + rawData[hi + 1][seriesIndex] * frac;
        return true;
    }
    return false;
}

bool DataVisualizer::compressedValueAt_inRange(uint64_t ts, uint8_t seriesIndex, double &outValue) const {
    const PolynomialSegment* segmentBuffer = compressor.getSegmentBuffer();
    uint8_t segmentCount = compressor.getSegmentCount();
    if (segmentCount == 0) return false;

    uint64_t cursor = getCompressedEndAbs();
    for (int s = (int)segmentCount - 1; s >= 0; --s) {
        const PolynomialSegment &seg = segmentBuffer[s];
        int startPoly = (s == (int)segmentCount - 1) ? (int)compressor.getCurrentPolyIndex() - 1 : (POLY_COUNT - 1);
        for (int p = startPoly; p >= 0; --p) {
            uint64_t dt = (uint64_t)seg.timeDeltas[p] * TIME_PRECISION_DIVIDER;
            if (dt == 0) continue;

            uint64_t startAbs = (cursor >= dt) ? (cursor - dt) : 0;

            if (ts >= startAbs && ts < cursor) {
                double tRel = (double)(ts - startAbs);
                double tNorm = (dt == 0 ? 0.0 : tRel / (double)dt);
                outValue = evaluatePolynomialNormalized(seg.coefficients[p][seriesIndex], POLY_DEGREE + 1, tNorm);
                return true;
            }
            cursor = startAbs;
        }
    }
    return false;
}

void DataVisualizer::computeWindowMinMaxDirect(uint64_t wStart, uint64_t wEnd, float &outMin, float &outMax) const {
    outMin = INFINITY; outMax = -INFINITY;
    if (wEnd <= wStart) { outMin = -1; outMax = 1; return; }
    const uint16_t SAMPLES = 80;
    uint64_t compressedEnd = getCompressedEndAbs();
    for (uint8_t series = 0; series < NUM_DATA_SERIES; ++series) {
        for (uint16_t i = 0; i <= SAMPLES; ++i) {
            uint64_t ts = wStart + i * (wEnd - wStart) / SAMPLES;
            double v;
            if (ts < compressedEnd && compressedValueAt_inRange(ts, series, v)) {
                outMin = min(outMin, (float)v); outMax = max(outMax, (float)v);
            } else if (rawInterpolatedValueAt(ts, series, v)) {
                outMin = min(outMin, (float)v); outMax = max(outMax, (float)v);
            }
        }
        for (uint16_t i = 0; i < rawDataIndex; ++i) {
            uint64_t t = rawTimestamps[i];
            if (t < wStart || t > wEnd) continue;
            outMin = min(outMin, rawData[i][series]); outMax = max(outMax, rawData[i][series]);
        }
    }
    if (isinf(outMin) || isinf(outMax)) { outMin = -1; outMax = 1; }
    float r = outMax - outMin;
    if (r <= 0.0001f) { outMin -= 1; outMax += 1; } else { outMin -= r*0.05f; outMax += r*0.05f; }
}

int DataVisualizer::buildPolyStarts(uint64_t starts[], int maxStarts) const {
    const PolynomialSegment* segmentBuffer = compressor.getSegmentBuffer();
    uint8_t segmentCount = compressor.getSegmentCount();
    if (segmentCount == 0) return 0;

    uint64_t cursor = getCompressedEndAbs();
    int count = 0;
    for (int s = (int)segmentCount - 1; s >= 0; --s) {
        const PolynomialSegment &seg = segmentBuffer[s];
        int startPoly = (s == (int)segmentCount - 1) ? (int)compressor.getCurrentPolyIndex() - 1 : (POLY_COUNT - 1);
        for (int p = startPoly; p >= 0; --p) {
            uint64_t dt = (uint64_t)seg.timeDeltas[p] * TIME_PRECISION_DIVIDER;
            if (dt == 0) continue;

            uint64_t startAbs = (cursor >= dt) ? (cursor - dt) : 0;
            if (count < maxStarts) starts[count++] = startAbs;
            cursor = startAbs;
        }
    }
    return count;
}

// =================================================================================================
// Main Drawing Function
// =================================================================================================

void DataVisualizer::drawCompoundGraph(int rx, int ry, int rw, int rh, uint64_t windowEndAbs, uint32_t windowDurationMs) {
    if (rw <= 0 || rh <= 0) return;

    uint64_t wEnd = windowEndAbs;
    uint64_t wStart = (windowDurationMs == 0 || windowDurationMs > wEnd) ? 0 : (wEnd - windowDurationMs);
    uint64_t compressedEnd = getCompressedEndAbs();

    float vmin, vmax;
    computeWindowMinMaxDirect(wStart, wEnd, vmin, vmax);

    tft.fillRect(rx, ry, rw, rh, TFT_BLACK);
    for (int i=0;i<=4;i++){ tft.drawFastHLine(rx, ry + (i*rh)/4, rw, 0x0821); tft.drawFastVLine(rx + (i*rw)/4, ry, rh, 0x0821);}

    auto tsToX = [&](uint64_t ts) {
        if (wEnd == wStart) return rx;
        double f = double(ts - wStart) / double(wEnd - wStart);
        return rx + (int)round(f * (rw - 1));
    };

    auto valueToY = [&](double val) {
        double c = val;
        if (c < vmin) c = vmin; if (c > vmax) c = vmax;
        return ry + (int)round((1.0 - (c - vmin) / (vmax - vmin)) * (rh - 1));
    };

    if (wEnd > compressedEnd) {
        int xStart = tsToX(compressedEnd);
        int xEnd = tsToX(wEnd);
        if (xEnd > xStart) tft.fillRect(xStart, ry, (xEnd - xStart), rh, 0x0821);
    }

    const int MAXS = SEGMENTS * POLY_COUNT;
    uint64_t starts[MAXS];
    int sc = buildPolyStarts(starts, MAXS);
    for (int i = sc - 1; i >= 0; --i) {
        uint64_t startAbs = starts[i];
        if (startAbs < wStart || startAbs > wEnd) continue;
        int x = tsToX(startAbs);
        tft.drawFastVLine(x, ry, rh, TFT_ORANGE);
    }

    if (compressedEnd >= wStart && compressedEnd <= wEnd) {
        int xb = tsToX(compressedEnd);
        tft.drawFastVLine(xb, ry, rh, TFT_MAGENTA);
    }

    uint16_t colors[] = {TFT_YELLOW, TFT_GREEN};
    for (uint8_t series = 0; series < NUM_DATA_SERIES; ++series) {
        if (compressor.getSegmentCount() > 0) {
            int lastY = -1;
            int compressedRightX = tsToX(min(wEnd, compressedEnd));
            for (int px = 0; px <= compressedRightX - rx; ++px) {
                uint64_t ts = wStart + px * (wEnd - wStart) / max(1, rw - 1);
                if (ts >= compressedEnd) break;
                double v;
                if (!compressedValueAt_inRange(ts, series, v)) continue;
                int x = rx + px;
                int y = valueToY(v);
                if (lastY >= 0) tft.drawLine(x - 1, lastY, x, y, colors[series]);
                else tft.drawPixel(x, y, colors[series]);
                lastY = y;
            }
        }

        if (rawDataIndex > 0) {
            int prevX = -1, prevY = -1;
            uint64_t current_rollovers = compressor.getRolloverCount();
            for (uint16_t i = 0; i < rawDataIndex; ++i) {
                uint64_t t = (current_rollovers << 32) | rawTimestamps[i];
                if (t < wStart || t > wEnd) continue;
                int x = tsToX(t);
                int y = valueToY(rawData[i][series]);
                tft.drawPixel(x, y, colors[series]);
                if (prevX >= 0) tft.drawLine(prevX, prevY, x, y, colors[series]);
                prevX = x; prevY = y;
            }
        }
    }

    tft.drawRect(rx, ry, rw, rh, TFT_WHITE);
}