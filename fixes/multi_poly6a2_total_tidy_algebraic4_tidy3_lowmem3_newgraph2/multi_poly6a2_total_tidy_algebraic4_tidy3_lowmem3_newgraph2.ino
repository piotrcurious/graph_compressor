#include <TFT_eSPI.h>
TFT_eSPI tft = TFT_eSPI(); // Create TFT instance
#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 240
#define GRAPH_HEIGHT 240
#define RAW_GRAPH_Y 0
#define COMPRESSED_GRAPH_Y 0

#include "AdvancedPolynomialFitter.hpp" // Include the advanced fitter
#include <vector>
#include <Arduino.h>
#include <algorithm>
#include <cmath>

//for debug
#define MAX_RAW_DATA 1440 // 1440 points each minute log is approx 24h
#define LOG_BUFFER_POINTS_PER_POLY 60 // how many points are accumulated in normal log buffer before fitting to polynomial

static float    raw_Data[MAX_RAW_DATA];
static uint32_t raw_timestamps[MAX_RAW_DATA];
static uint16_t raw_dataIndex = 0; 
static float raw_graphMinY = 0;
static float raw_graphMaxY = 0;
static uint32_t raw_log_delta = 0 ; // delta to last logged data, for the offset the compressed graph

 // Min/Max values for Y-axis scaling of the polynomial graph
static float minValue = INFINITY, maxValue = -INFINITY;

#include <stdint.h>

#define POLY_COUNT 8 // Number of polynomials in each segment
#define SEGMENTS 2    // Total number of segments
//const uint8_t POLYS_TO_COMBINE = 2;  // Number of polynomials to combine into one when recompression is triggered

#define POLY_DEGREE 5 // poly degree used for storage
#define SUB_FIT_POLY_DEGREE 3 //  supplementary fitter to extend boundary transitions .  
                    // the simpler the poly the better prediction , but the more complex data in the window the more chances it will cause misfit. 
                    // choose this up to your data. 
    #define BOUNDARY_MARGIN3 5 // duplicate data across margin for better fit, multiple of 2
    #define BOUNDARY_DELTA3 10 // time window of margin.                  
    #define BOUNDARY_MARGIN 5 // duplicate data across margin for better fit, multiple of 2
    #define BOUNDARY_DELTA 10 // time window of margin.

// Storage structure
struct PolynomialSegment {
    float coefficients[POLY_COUNT][POLY_DEGREE+1]; // 5th degree (6 coefficients), full resolution
    uint32_t timeDeltas[POLY_COUNT];   // 32-bit time deltas
};

// Data buffer
static PolynomialSegment segmentBuffer[SEGMENTS];
static uint8_t segmentCount = 0;
static uint32_t lastTimestamp = 0;
static uint16_t currentPolyIndex = 0;



// ---------- small helpers ----------
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline double normalizeTime(double t, double tMax) { return t / tMax; }
static inline float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
    if (in_max == in_min) return (out_min + out_max) * 0.5f;
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}
static inline uint32_t mapUint(uint32_t x, uint32_t in_min, uint32_t in_max, uint32_t out_min, uint32_t out_max) {
    if (in_max == in_min) return out_min;
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}



// Rolling buffer indices
static uint8_t head = 0; // Points to the oldest segment
//uint8_t tail = 0; // Points to the newest segment
//uint8_t segmentCount = 0; // Number of valid segments in buffer


//static float denormalizeTime(float tNorm, float tMax) {
//    return tNorm * tMax;
//}


// ---------- Direct (no-copy) drawing helpers ----------

static inline double evaluatePolynomialNormalized(const float *coefficients, uint8_t degree, double tNorm) {
    double result = 0.0;
    double tPower = 1.0;
    for (uint8_t i = 0; i < degree; ++i) {
        result += (double)coefficients[i] * tPower;
        tPower *= tNorm;
    }
    return result;
}

static uint32_t getCompressedEndAbs() {
    return (raw_dataIndex > 0) ? (raw_timestamps[raw_dataIndex - 1] - raw_log_delta) : 0u;
}


// Evaluate compressed representation at absolute timestamp 'ts' by scanning segmentBuffer in-place.
// Returns true if some polynomial evaluated (including extrapolation), false if no polynomials present.
static bool compressedValueAtDirect(uint32_t ts, double &outValue) {
    if (segmentCount == 0) return false;
    // newest polynomial end (absolute)
    uint32_t cursor = getCompressedEndAbs();
    // walk newest -> oldest
    for (int s = (int)segmentCount - 1; s >= 0; --s) {
        PolynomialSegment &seg = segmentBuffer[s];
        int startPoly = (s == (int)segmentCount - 1) ? (int)currentPolyIndex : (POLY_COUNT - 1);
        for (int p = startPoly; p >= 0; --p) {
            uint32_t dt = seg.timeDeltas[p];
            if (dt == 0) continue;
            uint32_t startAbs = (cursor >= dt) ? (cursor - dt) : 0u;
            if (ts >= startAbs && ts < cursor) {
                double tRel = (double)(ts - startAbs);
                double tNorm = (dt == 0 ? 0.0 : tRel / (double)dt);
                outValue = evaluatePolynomialNormalized(seg.coefficients[p], POLY_DEGREE + 1, tNorm);
                return true;
            }
            cursor = startAbs;
        }
    }
    // not inside any poly: extrapolate with nearest (oldest or newest)
    // newest poly (if any)
    // find newest valid poly
    uint32_t newestEnd = getCompressedEndAbs();
    for (int s = (int)segmentCount - 1; s >= 0; --s) {
        PolynomialSegment &seg = segmentBuffer[s];
        int startPoly = (s == (int)segmentCount - 1) ? (int)currentPolyIndex : (POLY_COUNT - 1);
        for (int p = startPoly; p >= 0; --p) {
            uint32_t dt = seg.timeDeltas[p];
            if (dt == 0) continue;
            // newest found -> extrapolate forward/backward relative to its start
            uint32_t endAbs = newestEnd;             // end of newest
            uint32_t startAbs = (endAbs >= dt) ? (endAbs - dt) : 0u;
            double tRel = (double)(ts - startAbs);
            double tNorm = (dt == 0 ? 0.0 : tRel / (double)dt);
            outValue = evaluatePolynomialNormalized(seg.coefficients[p], POLY_DEGREE + 1, tNorm);
            return true;
        }
    }
    return false;
}

// Evaluate compressed polynomials IN-PLACE. Returns true only if ts falls inside some stored polynomial
// (no extrapolation). This keeps compressed curve limited to available compressed data.
static bool compressedValueAt_inRange(uint32_t ts, double &outValue) {
    if (segmentCount == 0) return false;
    uint32_t cursor = getCompressedEndAbs(); // newest polynomial end
    for (int s = (int)segmentCount - 1; s >= 0; --s) {
        PolynomialSegment &seg = segmentBuffer[s];
        int startPoly = (s == (int)segmentCount - 1) ? (int)currentPolyIndex : (POLY_COUNT - 1);
        for (int p = startPoly; p >= 0; --p) {
            uint32_t dt = seg.timeDeltas[p];
            if (dt == 0) continue;
            uint32_t startAbs = (cursor >= dt) ? (cursor - dt) : 0u;
            if (ts >= startAbs && ts < cursor) {
                double tRel = (double)(ts - startAbs);
                double tNorm = (dt == 0 ? 0.0 : tRel / (double)dt);
                outValue = evaluatePolynomialNormalized(seg.coefficients[p], POLY_DEGREE + 1, tNorm);
                return true;
            }
            cursor = startAbs;
        }
    }
    return false; // not inside compressed region
}


// Binary-search raw interpolation (no copy)
static bool rawInterpolatedValueAt(uint32_t timestamp, double &outValue) {
    if (raw_dataIndex == 0) return false;
    if (timestamp <= raw_timestamps[0]) { outValue = raw_Data[0]; return true; }
    if (timestamp >= raw_timestamps[raw_dataIndex - 1]) { outValue = raw_Data[raw_dataIndex - 1]; return true; }

    int lo = 0, hi = (int)raw_dataIndex - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint32_t tm = raw_timestamps[mid];
        if (tm == timestamp) { outValue = raw_Data[mid]; return true; }
        if (tm < timestamp) lo = mid + 1; else hi = mid - 1;
    }
    if (hi >= 0 && hi + 1 < raw_dataIndex) {
        uint32_t t0 = raw_timestamps[hi], t1 = raw_timestamps[hi + 1];
        double frac = (t1 == t0) ? 0.0 : double(timestamp - t0) / double(t1 - t0);
        outValue = raw_Data[hi] * (1.0 - frac) + raw_Data[hi + 1] * frac;
        return true;
    }
    return false;
}

// Compute min/max across window by sampling (direct evaluation; no copies)
static void computeWindowMinMaxDirect_extrapolate(uint32_t windowStart, uint32_t windowEnd, float &outMin, float &outMax) {
    outMin = INFINITY; outMax = -INFINITY;
    if (windowEnd <= windowStart) {
        outMin = raw_graphMinY; outMax = raw_graphMaxY;
        if (isinf(outMin) || isinf(outMax)) { outMin = -1.0f; outMax = 1.0f; }
        return;
    }
    const uint16_t SAMPLE_POINTS = 80;
    for (uint16_t i = 0; i <= SAMPLE_POINTS; ++i) {
        uint32_t ts = windowStart + (uint64_t)i * (uint64_t)(windowEnd - windowStart) / SAMPLE_POINTS;
        double v;
        if (compressedValueAtDirect(ts, v)) {
            outMin = min(outMin, (float)v);
            outMax = max(outMax, (float)v);
        } else if (rawInterpolatedValueAt(ts, v)) {
            outMin = min(outMin, (float)v);
            outMax = max(outMax, (float)v);
        }
    }
    // also check raw samples inside window for exact extrema
    for (uint16_t i = 0; i < raw_dataIndex; ++i) {
        uint32_t t = raw_timestamps[i];
        if (t < windowStart || t > windowEnd) continue;
        outMin = min(outMin, raw_Data[i]);
        outMax = max(outMax, raw_Data[i]);
    }
    if (isinf(outMin) || isinf(outMax)) {
        outMin = raw_graphMinY; outMax = raw_graphMaxY;
        if (isinf(outMin) || isinf(outMax)) { outMin = -1.0f; outMax = 1.0f; }
    }
    float range = outMax - outMin;
    if (range <= 0.0001f) { outMin -= 1.0f; outMax += 1.0f; } else { outMin -= range * 0.05f; outMax += range * 0.05f; }
}

// Compute window min/max by sampling both compressed (only where available) and raw data (no copies)
static void computeWindowMinMaxDirect(uint32_t wStart, uint32_t wEnd, float &outMin, float &outMax) {
    outMin = INFINITY; outMax = -INFINITY;
    if (wEnd <= wStart) { outMin = raw_graphMinY; outMax = raw_graphMaxY; if (isinf(outMin)||isinf(outMax)){outMin=-1;outMax=1;} return; }
    const uint16_t SAMPLES = 80;
    uint32_t compressedEnd = getCompressedEndAbs();
    for (uint16_t i = 0; i <= SAMPLES; ++i) {
        uint32_t ts = wStart + (uint64_t)i * (uint64_t)(wEnd - wStart) / SAMPLES;
        double v;
        // prefer compressed when available
        if (ts < compressedEnd && compressedValueAt_inRange(ts, v)) {
            outMin = min(outMin, (float)v); outMax = max(outMax, (float)v);
        } else if (rawInterpolatedValueAt(ts, v)) {
            outMin = min(outMin, (float)v); outMax = max(outMax, (float)v);
        }
    }
    // scan raw points strictly inside window for precise extrema
    for (uint16_t i = 0; i < raw_dataIndex; ++i) {
        uint32_t t = raw_timestamps[i];
        if (t < wStart || t > wEnd) continue;
        outMin = min(outMin, raw_Data[i]); outMax = max(outMax, raw_Data[i]);
    }
    if (isinf(outMin) || isinf(outMax)) { outMin = raw_graphMinY; outMax = raw_graphMaxY; if (isinf(outMin)||isinf(outMax)){outMin=-1;outMax=1;} }
    float r = outMax - outMin;
    if (r <= 0.0001f) { outMin -= 1; outMax += 1; } else { outMin -= r*0.05f; outMax += r*0.05f; }
}


// build per-poly starts (stack-only) newest->oldest then reverse when drawing boundaries
static int buildPolyStarts(uint32_t starts[], int maxStarts) {
    if (segmentCount == 0) return 0;
    uint32_t cursor = getCompressedEndAbs();
    int count = 0;
    for (int s = (int)segmentCount - 1; s >= 0; --s) {
        PolynomialSegment &seg = segmentBuffer[s];
        int startPoly = (s == (int)segmentCount - 1) ? (int)currentPolyIndex : (POLY_COUNT - 1);
        for (int p = startPoly; p >= 0; --p) {
            uint32_t dt = seg.timeDeltas[p];
            if (dt == 0) continue;
            uint32_t startAbs = (cursor >= dt) ? (cursor - dt) : 0u;
            if (count < maxStarts) starts[count++] = startAbs;
            cursor = startAbs;
        }
    }
    return count; // newest-first
}


// Add a new segment to the buffer
static void addSegment(const PolynomialSegment &newSegment) {
    segmentBuffer[segmentCount] = newSegment;
//    tail = (tail + 1) % SEGMENTS;
//      tail = tail+1; 
      segmentCount++;
    if (segmentCount <= SEGMENTS) {
    } else {
        // Buffer full, advance head to discard the oldest segment
//        head = (head + 1) % SEGMENTS;
    }
}

// Check if the buffer is full
static bool isBufferFull() {
    return segmentCount == SEGMENTS;
}

// Retrieve the oldest and second-oldest segments
static void getOldestSegments(PolynomialSegment &oldest, PolynomialSegment &secondOldest) {
    oldest = segmentBuffer[head];
    secondOldest = segmentBuffer[(head + 1) % SEGMENTS];
}

// Remove the oldest two segments
static void removeOldestTwo() {
    //head = (head + 2) % SEGMENTS;
    segmentCount -= 2;
}

#include <math.h>

static void compressDataToSegment(const PolynomialSegment *segments, uint8_t count, uint16_t polyindex,
                                  const float *rawData, const uint32_t *timestamps, uint16_t dataSize,
                                  float *coefficients, uint32_t &timeDelta) {
    AdvancedPolynomialFitter fitter;
    int8_t segmentIndex = count - 1;
    int16_t polyIndex = (int16_t)polyindex;

    const int preMargin3 = BOUNDARY_MARGIN3;
    const int preMargin = BOUNDARY_MARGIN / 2;
    const int postMargin = BOUNDARY_MARGIN / 2;

    // compute totalTime as sum of timestamps (absolute units)
    uint64_t totalTimeU64 = 0;
    for (uint16_t j = 0; j < dataSize; ++j) totalTimeU64 += timestamps[j];
    double totalTime = double(totalTimeU64);
    if (totalTime <= 0.0) totalTime = 1.0; // avoid zero

    // helper: convert absolute-time monomial coeffs c_k (P(x)=sum c_k x^k)
    // into normalized-domain coeffs c'_k for u in [0..1], where x = u * totalTime:
    // P(u) = sum (c_k * totalTime^k) * u^k
    auto convertAbsToNormalized = [&](const std::vector<float>& absCoeffs)->std::vector<float> {
        int m = (int)absCoeffs.size();
        std::vector<float> out(m, 0.0f);
        double scalePow = 1.0; // totalTime^k
        for (int k = 0; k < m; ++k) {
            out[k] = static_cast<float>(double(absCoeffs[k]) * scalePow);
            scalePow *= totalTime;
        }
        return out;
    };

    // safe neighbor evaluators (map absolute time -> neighbor's normalized domain)
    auto eval_previous_poly_at_abs = [&](double abs_t, float fallback, int prev_seg_idx, int prev_poly_idx)->float {
        if (prev_seg_idx < 0 || prev_seg_idx >= (int)count) return fallback;
        if (prev_poly_idx < 0 || prev_poly_idx >= POLY_COUNT) return fallback;
        uint32_t prev_duration = segments[prev_seg_idx].timeDeltas[prev_poly_idx];
        if (prev_duration == 0) return fallback;
        double t_local = double(prev_duration) + abs_t; // for negative abs_t map to tail of previous poly
        if (t_local < 0.0) t_local = 0.0;
        if (t_local > double(prev_duration)) t_local = double(prev_duration);
        float tNorm = normalizeTime((uint32_t)std::lround(t_local), prev_duration);
        return evaluatePolynomial(segments[prev_seg_idx].coefficients[prev_poly_idx], POLY_DEGREE+1, tNorm);
    };

    auto eval_next_poly_at_abs = [&](double abs_t, float fallback, int next_seg_idx, int next_poly_idx)->float {
        if (next_seg_idx < 0 || next_seg_idx >= (int)count) return fallback;
        if (next_poly_idx < 0 || next_poly_idx >= POLY_COUNT) return fallback;
        uint32_t next_duration = segments[next_seg_idx].timeDeltas[next_poly_idx];
        if (next_duration == 0) return fallback;
        double t_local = abs_t; // map post-boundary offset into next poly start
        if (t_local < 0.0) t_local = 0.0;
        if (t_local > double(next_duration)) t_local = double(next_duration);
        float tNorm = normalizeTime((uint32_t)std::lround(t_local), next_duration);
        return evaluatePolynomial(segments[next_seg_idx].coefficients[next_poly_idx], POLY_DEGREE+1, tNorm);
    };

    // ---------------- SUB-FIT (seed polynomial) ----------------
    {
        size_t Nsub = (size_t)dataSize + preMargin3;
        std::vector<double> absTimes(Nsub);
        std::vector<float> values(Nsub);

        double preStart = -double(preMargin3 * BOUNDARY_DELTA3);
        for (int i = 0; i < preMargin3; ++i) absTimes[i] = preStart + double(i * BOUNDARY_DELTA3);

        double tAbs = 0.0;
        for (uint16_t j = 0; j < dataSize; ++j) {
            tAbs += double(timestamps[j]);
            absTimes[j + preMargin3] = tAbs;
            values[j + preMargin3] = rawData[j];
        }

        // evaluate pre-boundary values: blend neighbor & fallback (Hann window)
        int prev_seg_idx = (segmentIndex > 0) ? (segmentIndex - 1) : -1;
        int prev_poly_idx = POLY_COUNT - 1;
        for (int i = 0; i < preMargin3; ++i) {
            double at = absTimes[i]; // negative
            float fallback = rawData[0];
            float neighborVal;
            if (polyIndex == 0) neighborVal = eval_previous_poly_at_abs(at, fallback, prev_seg_idx, prev_poly_idx);
            else               neighborVal = eval_previous_poly_at_abs(at, fallback, segmentIndex, polyIndex - 1);

            double t = (preMargin3 <= 1) ? 0.0 : (double(i) / double(preMargin3 - 1));
            double w = 0.5 * (1.0 - cos(M_PI * (1.0 - t))); // reversed Hann (furthest -> neighbor)
            values[i] = float(w * neighborVal + (1.0 - w) * fallback);
        }

        // Fit in ABSOLUTE time domain (fitter expects absolute times)
        std::vector<float> fitted_sub_abs = fitter.fitPolynomialD_lowmem(absTimes, values, SUB_FIT_POLY_DEGREE, AdvancedPolynomialFitter::NONE);

        // Convert to NORMALIZED domain coefficients for evaluatePolynomial usage and store into 'coefficients' buffer
        std::vector<float> fitted_sub_norm = convertAbsToNormalized(fitted_sub_abs);
        for (size_t k = 0; k < fitted_sub_norm.size() && k < (SUB_FIT_POLY_DEGREE + 1); ++k) {
            coefficients[k] = fitted_sub_norm[k];
        }
    }

    // ---------------- MAIN FIT ----------------
    {
        size_t Nmain = (size_t)dataSize + preMargin + postMargin;
        std::vector<double> absTimes(Nmain);
        std::vector<float> values(Nmain);

        double preStart = -double(preMargin * BOUNDARY_DELTA);
        for (int i = 0; i < preMargin; ++i) absTimes[i] = preStart + double(i * BOUNDARY_DELTA);

        double tAbs = 0.0;
        for (uint16_t j = 0; j < dataSize; ++j) {
            tAbs += double(timestamps[j]);
            absTimes[j + preMargin] = tAbs;
            values[j + preMargin] = rawData[j];
        }
        double dataEndAbs = tAbs;

        for (int i = 0; i < postMargin; ++i)
            absTimes[dataSize + preMargin + i] = dataEndAbs + double((i + 1) * BOUNDARY_DELTA);

        // pre-boundary blending (neighbor vs seed/sub-fit fallback)
        int prev_seg_idx = (segmentIndex > 0) ? (segmentIndex - 1) : -1;
        int prev_poly_idx = POLY_COUNT - 1;
        for (int i = 0; i < preMargin; ++i) {
            double at = absTimes[i];
            uint32_t t_clamped = (at <= 0.0) ? 0U : ( (at >= totalTime) ? (uint32_t)std::lround(totalTime) : (uint32_t)std::lround(at) );
            float fallback = evaluatePolynomial(coefficients, SUB_FIT_POLY_DEGREE+1, normalizeTime(t_clamped, (uint32_t)std::lround(totalTime)));
            float neighborVal;
            if (polyIndex == 0) neighborVal = eval_previous_poly_at_abs(at, fallback, prev_seg_idx, prev_poly_idx);
            else               neighborVal = eval_previous_poly_at_abs(at, fallback, segmentIndex, polyIndex - 1);

            double t = double(preMargin - i) / double(preMargin + 1);
            double w = 0.5 * (1.0 - cos(M_PI * t)); // Hann
            values[i] = float(w * neighborVal + (1.0 - w) * fallback);
        }

        // post-boundary blending (neighbor vs seed/sub-fit fallback)
        int next_seg_idx = (segmentIndex + 1 < count) ? (segmentIndex + 1) : -1;
        int next_poly_idx = 0;
        for (int i = 0; i < postMargin; ++i) {
            size_t idx = dataSize + preMargin + i;
            double at = absTimes[idx];
            uint32_t t_clamped = (at <= 0.0) ? 0U : ( (at >= totalTime) ? (uint32_t)std::lround(totalTime) : (uint32_t)std::lround(at) );
            float fallback = evaluatePolynomial(coefficients, SUB_FIT_POLY_DEGREE+1, normalizeTime(t_clamped, (uint32_t)std::lround(totalTime)));
            float neighborVal;
            if (polyIndex == (POLY_COUNT - 1)) neighborVal = eval_next_poly_at_abs(at, fallback, next_seg_idx, next_poly_idx);
            else                                neighborVal = eval_next_poly_at_abs(at, fallback, segmentIndex, polyIndex + 1);

            double t = double(i + 1) / double(postMargin + 1);
            double w = 0.5 * (1.0 - cos(M_PI * t)); // Hann
            values[idx] = float(w * neighborVal + (1.0 - w) * fallback);
        }

        // Fit in ABSOLUTE time domain (fitter expects absolute times)
        std::vector<float> fitted_abs = fitter.fitPolynomialD_lowmem(absTimes, values, POLY_DEGREE, AdvancedPolynomialFitter::NONE);

        // Convert to normalized-domain coefficients and store to output buffer
        std::vector<float> fitted_norm = convertAbsToNormalized(fitted_abs);
        for (size_t k = 0; k < fitted_norm.size() && k < (POLY_DEGREE + 1); ++k) {
            coefficients[k] = fitted_norm[k];
        }

        timeDelta = (uint32_t)std::lround(dataEndAbs);
    }
}





 // on some systems this is faster
static double evaluatePolynomial(const float *coefficients, uint8_t degree, double t) {
    // t is already in milliseconds within the segment's time delta range
    double result = 0.0;
    double tPower = 1.0;  // t^0 = 1
    
    for (int i = 0; i < degree; i++) {
        result += coefficients[i] * tPower;
        tPower *= t;  // More efficient than pow()
    }
   // Serial.println(result);
    return result;
}


// Modified evaluation function to handle normalized time
//static double evaluatePolynomialDelta(const float *coefficients, uint8_t degree, double t) {
//    // Normalize t to [0,1] range using the segment's time delta
//    double tNorm = t / segmentBuffer[segmentCount-1].timeDeltas[currentPolyIndex];
//    
//   double result = 0.0;
//    double tPower = 1.0;
//    
//    for (int i = 0; i < degree; i++) {
//        result += coefficients[i] * tPower;
//        tPower *= tNorm;
//    }
//    return result;
//}


// Modified combinePolynomials function
static void combinePolynomials(const PolynomialSegment &oldest, const PolynomialSegment &secondOldest, PolynomialSegment &recompressedSegment) {
    AdvancedPolynomialFitter fitter;
    uint16_t currentPolyIndex = 0;

    for (uint16_t i = 0; i < POLY_COUNT; i = i + 2) {
        if (oldest.timeDeltas[i] == 0 || oldest.timeDeltas[i+1] == 0) break;

        double combinedTimeDelta = oldest.timeDeltas[i] + oldest.timeDeltas[i+1];

        std::vector<float> newCoefficients = fitter.composePolynomials(oldest.coefficients[i], oldest.timeDeltas[i], oldest.coefficients[i+1], oldest.timeDeltas[i+1], POLY_DEGREE);

        // Store coefficients
        for (uint8_t j = 0; j < newCoefficients.size() && j < POLY_DEGREE + 1; j++) {
            recompressedSegment.coefficients[currentPolyIndex][j] = newCoefficients[j];
        }

        recompressedSegment.timeDeltas[currentPolyIndex] = combinedTimeDelta;
        currentPolyIndex++;
    }

    // Handle second oldest segment similarly
    for (uint16_t i = 0; i < POLY_COUNT; i = i + 2) {
        if (secondOldest.timeDeltas[i] == 0 || secondOldest.timeDeltas[i+1] == 0) break;
        
        double combinedTimeDelta = secondOldest.timeDeltas[i] + secondOldest.timeDeltas[i+1];

        std::vector<float> newCoefficients = fitter.composePolynomials(secondOldest.coefficients[i], secondOldest.timeDeltas[i], secondOldest.coefficients[i+1], secondOldest.timeDeltas[i+1], POLY_DEGREE);

        for (uint8_t j = 0; j < newCoefficients.size() && j < POLY_DEGREE + 1; j++) {
            recompressedSegment.coefficients[currentPolyIndex][j] = newCoefficients[j];
        }

        recompressedSegment.timeDeltas[currentPolyIndex] = combinedTimeDelta;
        currentPolyIndex++;
    }
}

static void recompressSegments() {
    if (segmentCount < 2) return;

    PolynomialSegment oldest, secondOldest;
    getOldestSegments(oldest, secondOldest);
    
    // Create recompressed segment
    PolynomialSegment recompressedSegment;
    for (uint16_t i = 0; i < POLY_COUNT; i++) {
        recompressedSegment.timeDeltas[i] = 0;
    }
    
    combinePolynomials(oldest, secondOldest, recompressedSegment);
  
    // Add recompressed segment at the correct position (head)
    uint8_t insertPos = head;
    //head = (head + 1) % SEGMENTS;  // Update head to next position
    segmentBuffer[insertPos] = recompressedSegment;
    // Insert recompressed segment

    // Shift existing segments if needed
    for (uint8_t i = insertPos+1 ; i < segmentCount-1; i++) {
        segmentBuffer[i] = segmentBuffer[i+1];
    }
    
    Serial.print("Recompressed. New segment count: ");
    Serial.println(segmentCount-1);
    Serial.print("poly size: ");
    Serial.print(sizeof(segmentBuffer));
    Serial.print(" raw size: ");
    Serial.println(sizeof(raw_Data)+sizeof(raw_timestamps));    
}

// Sample scalar data (simulated random data for now)
static float sampleScalarData(uint32_t timestamp) {
    float scalar = random(0, 1000*sin((float)timestamp * 0.0002)) / 100.0; // background noise
    scalar = scalar + 10 * sin((float)timestamp * 0.001)+20*sin((float)timestamp * 0.0001);
    return scalar; // Random data
}

void logSampledData(float data, uint32_t currentTimestamp) {
    static float rawData[LOG_BUFFER_POINTS_PER_POLY];
    static uint32_t timestamps[LOG_BUFFER_POINTS_PER_POLY];
    static uint16_t dataIndex = 0;

    // Calculate time delta
    uint32_t timeDelta = (currentTimestamp - lastTimestamp);
    lastTimestamp = currentTimestamp;

    // Store the data and timestamp
    rawData[dataIndex] = data;
    timestamps[dataIndex] = timeDelta;
    dataIndex++;
    raw_log_delta += timeDelta;
  
   if (dataIndex >= LOG_BUFFER_POINTS_PER_POLY) {

         // Initialize first segment if needed
        if (segmentCount == 0) {
           addSegment(PolynomialSegment());
            currentPolyIndex = 0 ;  
            // Initialize new segment's timeDeltas
            for (uint16_t i = 0; i < POLY_COUNT; i++) {
                segmentBuffer[segmentCount-1].timeDeltas[i] = 0;
            }
            segmentBuffer[segmentCount-1].timeDeltas[currentPolyIndex]=1; // initalize first poly. it acts as extra storage for oldest data.       
        } 
     currentPolyIndex++;

        // If current segment is full, prepare for next segment
        if (currentPolyIndex >= POLY_COUNT) {
            
            if (segmentCount < SEGMENTS) {
                // Create new segment
                addSegment(PolynomialSegment());
                // Initialize timeDeltas for new segment
                for (uint16_t i = 0; i < POLY_COUNT; i++) {
                    segmentBuffer[segmentCount-1].timeDeltas[i] = 0;
                }
                currentPolyIndex = 0;
                Serial.print("Created new segment ");
                Serial.println(segmentCount-1);
            } else {
                // Trigger recompression when buffer is full
                recompressSegments();
                // initalize time deltas for freshly cleared segment
                for (uint16_t i = 0; i < POLY_COUNT; i++) {
                    segmentBuffer[segmentCount-1].timeDeltas[i] = 0;
                }
                currentPolyIndex = 0;
            }
        }

        // Fit polynomial to current data chunk
        float new_coefficients[POLY_DEGREE+1];
        uint32_t new_timeDelta;
        compressDataToSegment(segmentBuffer,segmentCount,currentPolyIndex,rawData, timestamps, dataIndex, new_coefficients, new_timeDelta);

        // Store the polynomial in current segment
        for (uint8_t i = 0; i < POLY_DEGREE+1; i++) {
            segmentBuffer[segmentCount-1].coefficients[currentPolyIndex][i] = new_coefficients[i];
        }
        segmentBuffer[segmentCount-1].timeDeltas[currentPolyIndex] = new_timeDelta;

        raw_log_delta = 0; 
        Serial.print("Added polynomial ");
        Serial.print(currentPolyIndex);
        Serial.print(" to segment ");
        Serial.println(segmentCount-1);

        // Reset data buffer
        dataIndex = 0;
    }
}

// Log sampled data into the current segment
static void raw_logSampledData(float data, uint32_t currentTimestamp) {
    // Check if the current segment is full
    if (raw_dataIndex >= MAX_RAW_DATA - 1) {
        raw_graphMinY = raw_graphMaxY;
        raw_graphMaxY = 0;
        for (uint16_t i = 0; i < raw_dataIndex; i++) {
            raw_Data[i] = raw_Data[i + 1];
            raw_timestamps[i] = raw_timestamps[i + 1];
            if (raw_Data[i] > raw_graphMaxY) {
                raw_graphMaxY = raw_Data[i];
            }
            if (raw_Data[i] < raw_graphMinY) {
                raw_graphMinY = raw_Data[i];
            }
        }
        raw_Data[raw_dataIndex] = data;
        raw_timestamps[raw_dataIndex] = currentTimestamp;
    } else {
        // Store the data and timestamp
        raw_Data[raw_dataIndex] = data;
        raw_timestamps[raw_dataIndex] = currentTimestamp;
        raw_graphMinY = raw_graphMaxY;
        raw_graphMaxY = 0;
        for (uint16_t i = 0; i < raw_dataIndex; i++) {
            if (raw_Data[i] > raw_graphMaxY) {
                raw_graphMaxY = raw_Data[i];
            }
            if (raw_Data[i] < raw_graphMinY) {
                raw_graphMinY = raw_Data[i];
            }
        }
        raw_dataIndex++;
    }
}

void setup() {
    Serial.begin(115200);
    randomSeed(analogRead(0));

    // TFT initialization
    tft.init();
    tft.setRotation(1);
    tft.initDMA();
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_WHITE);
    tft.setTextSize(1);
}

static void drawRawGraph() {
    // Time window for alignment
    uint32_t windowStart = raw_timestamps[0];
    uint32_t windowEnd = raw_timestamps[raw_dataIndex -1] ;

      uint32_t xMin = windowStart, xMax = windowEnd - raw_log_delta;    
    uint16_t SWdelta = mapFloat(raw_log_delta+xMin, xMin, windowEnd, 0, SCREEN_WIDTH-1);
    uint16_t Swidth = SCREEN_WIDTH-SWdelta;
    uint32_t lastDataX = xMax;
    
    if(SWdelta){tft.fillRect(SCREEN_WIDTH-SWdelta,0,SWdelta,SCREEN_HEIGHT,0x0821);}
   // if(SWdelta){tft.drawRect(SCREEN_WIDTH-1-SWdelta,0,SWdelta,SCREEN_HEIGHT-1,TFT_RED);}   
    
    // Draw the new point
    for (uint16_t i = 0; i < raw_dataIndex; i++) {
        uint16_t y = mapFloat(raw_Data[i], raw_graphMinY, raw_graphMaxY, SCREEN_HEIGHT - 1, 0);
        uint16_t x = mapFloat(raw_timestamps[i], raw_timestamps[0], raw_timestamps[raw_dataIndex - 1], 0, SCREEN_WIDTH);
        tft.drawPixel(x, y, TFT_BLUE);
    }
}




void updateMinMax(const PolynomialSegment *segments, uint8_t count, uint16_t polyindex,uint32_t windowStart, uint32_t windowEnd, bool clear_under, bool draw_lines) {
    // Initialize tracking indices
    int16_t segmentIndex = count - 1;
    int16_t polyIndex = polyindex;
    // First pass: Calculate min/max values
    uint32_t tCurrent = windowEnd;
    minValue = INFINITY;
    maxValue = -INFINITY;
 
    for (int16_t i = 0; i < count; ++i) {
        const PolynomialSegment &segment = segments[segmentIndex];
        for (int16_t j = (i == 0 ? polyIndex : POLY_COUNT - 1); j >= 0; --j) {
            uint32_t tDelta = segment.timeDeltas[j];
            if (tDelta == 0) continue;

            uint32_t numSteps = min(100UL, tDelta);
            uint32_t stepSize = tDelta / numSteps;

            for (uint32_t t = stepSize; t <= tDelta; t += stepSize) {
                uint32_t actualTime = tCurrent - t;
                if (actualTime < windowStart || actualTime > windowEnd) break;

                float value = evaluatePolynomial(segment.coefficients[j],POLY_DEGREE+1, t);
                minValue = min(minValue, value);
                maxValue = max(maxValue, value);
            }
            tCurrent -= tDelta;
            if (tCurrent < windowStart) break;
        }
        if (--segmentIndex < 0) break;
    }
 // use min/max value from previous graph pass 
 
    if (isinf(minValue) || isinf(maxValue)) {
        minValue = raw_graphMinY;
        maxValue = raw_graphMaxY;
    }

    // Add margin for aesthetics
 //   float valueRange = maxValue - minValue;
 //   minValue -= valueRange * 0.05f;
 //   maxValue += valueRange * 0.05f;
    
}


void updateCompressedGraphBackwardsFastOpt(const PolynomialSegment *segments, uint8_t count, uint16_t polyindex, bool clear_under, bool draw_lines) {
    if (count == 0) return;

    uint32_t windowStart = raw_timestamps[0];
    uint32_t windowEnd = raw_timestamps[raw_dataIndex - 1];
    float new_minValue = INFINITY, new_maxValue = -INFINITY;
    
    uint32_t xMin = windowStart, xMax = windowEnd - raw_log_delta;
    int16_t segmentIndex = count - 1;
    int16_t polyIndex = polyindex; 
    uint32_t lastDataX = xMax;
    uint16_t SWdelta = mapFloat(raw_log_delta+windowStart, windowStart, windowEnd, 0, SCREEN_WIDTH);
    uint16_t Swidth = SCREEN_WIDTH-SWdelta-1;
 
    if(SWdelta) {
        tft.drawRect(SCREEN_WIDTH-SWdelta, 0, SWdelta, SCREEN_HEIGHT-1, TFT_RED);
    }   
   
    int16_t lastY = -1;
    for (int x = Swidth; x >= 0; --x) {
        double dataX = mapFloat(x, +0.0, Swidth, xMin, xMax);  
        double tDelta = segments[segmentIndex].timeDeltas[polyIndex] - (lastDataX - dataX);   
        
        if(clear_under) {
            tft.drawFastVLine(x, 0, SCREEN_HEIGHT, TFT_BLACK);
        }
        
        while (segmentIndex >= 0 && ((lastDataX - dataX) >= segments[segmentIndex].timeDeltas[polyIndex])) {
            tft.drawFastVLine(x, 0, SCREEN_HEIGHT, 0x0821);
            lastDataX -= segments[segmentIndex].timeDeltas[polyIndex];
            if (--polyIndex < 0) {
                polyIndex = POLY_COUNT - 1;
                if (--segmentIndex < 0) break;
                tft.drawFastVLine(x, 0, SCREEN_HEIGHT, TFT_RED);
            }
            tDelta = segments[segmentIndex].timeDeltas[polyIndex] - (lastDataX - dataX);
        }

        // Normalize tDelta before evaluation
        double tNorm = normalizeTime(tDelta, segments[segmentIndex].timeDeltas[polyIndex]);
        //Serial.println(tNorm);

        double yFitted = 0.0f;
        double tPower = 1.0;
        
        for (uint8_t i = 0; i < POLY_DEGREE+1; i++) {
            yFitted += segments[segmentIndex].coefficients[polyIndex][i] * tPower;
 //Serial.println(yFitted);
            tPower *= tNorm;
 //Serial.println(tPower);
        }
        new_minValue = min(new_minValue, (float)yFitted);
        new_maxValue = max(new_maxValue, (float)yFitted);

        uint16_t y = 0;
        if (!isnan(yFitted)) {
            y = mapFloat(yFitted, minValue, maxValue, SCREEN_HEIGHT - 1, 0);
            if (y < SCREEN_HEIGHT) {
                if(draw_lines && lastY > 0) {
                    tft.drawLine(x, y, x+1, lastY, TFT_YELLOW);  
                } else {
                    tft.drawPixel(x, y, TFT_WHITE);  
                }                
            }
        }
        lastY = y;
    }
    minValue = new_minValue;
    maxValue = new_maxValue;
}


// draw helpers for compactness
static inline int tsToX(int rx, int rw, uint32_t wStart, uint32_t wEnd, uint32_t ts) {
    if (wEnd == wStart) return rx;
    double f = double(ts - wStart) / double(wEnd - wStart);
    return rx + (int)round(f * (rw - 1));
}
static inline int valueToY(int ry, int rh, float vmin, float vmax, double val) {
    double c = val;
    if (c < vmin) c = vmin; if (c > vmax) c = vmax;
    return ry + (int)round((1.0 - (c - vmin) / (vmax - vmin)) * (rh - 1));
}

// --------------- Main compact drawing function (aligned, no extrapolation) ---------------
static void drawCompoundGraphAligned(TFT_eSPI &tft,
                                     int rx, int ry, int rw, int rh,
                                     uint32_t windowEndAbs,
                                     uint32_t windowDurationMs,
                                     bool drawCompressed = true,
                                     bool drawRaw = true,
                                     bool clearArea = true,
                                     bool drawGrid = true,
                                     bool drawLines = true) {
    if (rw <= 0 || rh <= 0) return;

    uint32_t wEnd = windowEndAbs;
    uint32_t wStart = (windowDurationMs == 0 || windowDurationMs > wEnd) ? 0u : (wEnd - windowDurationMs);
    uint32_t compressedEnd = getCompressedEndAbs();

    // compute vmin/vmax aligned to compressed availability (compressed used when available)
    float vmin, vmax;
    computeWindowMinMaxDirect(wStart, wEnd, vmin, vmax);

    if (clearArea) tft.fillRect(rx, ry, rw, rh, TFT_BLACK);
    if (drawGrid) { for (int i=0;i<=4;i++){ tft.drawFastHLine(rx, ry + (i*rh)/4, rw, 0x0821); tft.drawFastVLine(rx + (i*rw)/4, ry, rh, 0x0821);} }

    // Draw uncompressed tail FIRST (shaded), so it won't erase curves
    if (wEnd > compressedEnd) {
        int xStart = tsToX(rx, rw, wStart, wEnd, compressedEnd);
        int xEnd = tsToX(rx, rw, wStart, wEnd, wEnd);
        if (xEnd > xStart) tft.fillRect(xStart, ry, (xEnd - xStart), rh, 0x0821);
    }

    // Draw per-poly boundaries (oldest->newest). Build starts into small stack then draw reversed.
    const int MAXS = SEGMENTS * POLY_COUNT;
    uint32_t starts[MAXS];
    int sc = buildPolyStarts((uint32_t*)starts, MAXS); // newest-first
    for (int i = sc - 1; i >= 0; --i) { // draw oldest->newest
        uint32_t startAbs = starts[i];
        if (startAbs < wStart || startAbs > wEnd) continue;
        int x = tsToX(rx, rw, wStart, wEnd, startAbs);
        tft.drawFastVLine(x, ry, rh, TFT_ORANGE);
        // optionally draw thicker/colored mark at segment boundaries — could detect seg changes if desired
    }
    // marker at compressedEnd
    if (compressedEnd >= wStart && compressedEnd <= wEnd) {
        int xb = tsToX(rx, rw, wStart, wEnd, compressedEnd);
        tft.drawFastVLine(xb, ry, rh, TFT_MAGENTA);
    }

    // Draw compressed curve ONLY where compressed data exists (ts < compressedEnd)
    if (drawCompressed && segmentCount > 0) {
        int lastY = -1;
        // compute rightmost pixel for compressed region
        int compressedRightX = tsToX(rx, rw, wStart, wEnd, min(wEnd, compressedEnd));
        for (int px = 0; px <= compressedRightX - rx; ++px) {
            uint32_t ts = wStart + (uint64_t)px * (uint64_t)(wEnd - wStart) / (uint32_t)max(1, rw - 1);
            if (ts >= compressedEnd) break; // safety
            double v;
            if (!compressedValueAt_inRange(ts, v)) continue;
            int x = rx + px;
            int y = valueToY(ry, rh, vmin, vmax, v);
            if (drawLines && lastY >= 0) tft.drawLine(x-1, lastY, x, y, TFT_YELLOW);
            else tft.drawPixel(x, y, TFT_YELLOW);
            lastY = y;
        }
    }

    // Overlay raw samples & connecting lines (entire raw timeline inside window)
    if (drawRaw && raw_dataIndex > 0) {
        int prevX = -1, prevY = -1;
        for (uint16_t i = 0; i < raw_dataIndex; ++i) {
            uint32_t t = raw_timestamps[i];
            if (t < wStart || t > wEnd) continue;
            int x = tsToX(rx, rw, wStart, wEnd, t);
            int y = valueToY(ry, rh, vmin, vmax, raw_Data[i]);
            tft.drawPixel(x, y, TFT_CYAN);
            if (drawLines && prevX >= 0) tft.drawLine(prevX, prevY, x, y, TFT_CYAN);
            prevX = x; prevY = y;
        }
    }

    // bounding rectangle last
    tft.drawRect(rx, ry, rw, rh, TFT_WHITE);
}

// Draw compound graph directly from segmentBuffer & raw arrays (no copies)
static void drawCompoundGraph(TFT_eSPI &tft,
                                   int rx, int ry, int rw, int rh,
                                   uint32_t windowEndAbs,
                                   uint32_t windowDurationMs,
                                   bool drawCompressed = true,
                                   bool drawRaw = true,
                                   bool clearArea = true,
                                   bool drawGrid = true,
                                   bool drawLines = true) {
    if (rw <= 0 || rh <= 0) return;

    uint32_t windowEnd = windowEndAbs;
    uint32_t windowStart = (windowDurationMs == 0 || windowDurationMs > windowEnd) ? 0u : (windowEnd - windowDurationMs);
    uint32_t compressedEnd = getCompressedEndAbs();

    // compute min/max by direct evaluation
    float vmin, vmax;
    computeWindowMinMaxDirect(windowStart, windowEnd, vmin, vmax);

    if (clearArea) tft.fillRect(rx, ry, rw, rh, TFT_BLACK);
    if (drawGrid) {
        const int GRID_H = 4, GRID_V = 4;
        for (int i = 0; i <= GRID_H; ++i) {
            int yy = ry + (i * rh) / GRID_H;
            tft.drawFastHLine(rx, yy, rw, 0x0821);
        }
        for (int i = 0; i <= GRID_V; ++i) {
            int xx = rx + (i * rw) / GRID_V;
            tft.drawFastVLine(xx, ry, rh, 0x0821);
        }
    }

    auto tsToX = [&](uint32_t ts)->int {
        if (windowEnd == windowStart) return rx;
        double frac = double(ts - windowStart) / double(windowEnd - windowStart);
        return rx + (int)round(frac * (rw - 1));
    };
    auto valueToY = [&](double val)->int {
        double clamped = val;
        if (clamped < vmin) clamped = vmin;
        if (clamped > vmax) clamped = vmax;
        return ry + (int)round((1.0 - (clamped - vmin) / (vmax - vmin)) * (rh - 1));
    };

    // Draw uncompressed tail FIRST so it won't overwrite curves
    if (windowEnd > compressedEnd) {
        int xs = tsToX(compressedEnd);
        int xe = tsToX(windowEnd);
        if (xe > xs) tft.fillRect(xs, ry, (xe - xs), rh, 0x0821);
    }

    // Draw per-polynomial boundaries by scanning segments/polys and accumulating cursor (oldest->newest)
    // We'll compute absolute start times on the fly, no copy:
    {
        // find absolute time of newest end
        uint32_t cursor = compressedEnd;
        // collect boundaries newest->oldest into small stack to draw oldest->newest in order
        // but we can draw on the fly if ordering doesn't matter; here draw as we discover (oldest->newest requires storing starts)
        // to avoid allocation we'll iterate twice: first compute total counts, second draw while using a small fixed array of starts limited to SEGMENTS*POLY_COUNT
        uint32_t starts[SEGMENTS * POLY_COUNT];
        int starts_count = 0;
        // newest->oldest fill starts[]
        for (int s = (int)segmentCount - 1; s >= 0; --s) {
            PolynomialSegment &seg = segmentBuffer[s];
            int startPoly = (s == (int)segmentCount - 1) ? (int)currentPolyIndex : (POLY_COUNT - 1);
            for (int p = startPoly; p >= 0; --p) {
                uint32_t dt = seg.timeDeltas[p];
                if (dt == 0) continue;
                uint32_t startAbs = (cursor >= dt) ? (cursor - dt) : 0u;
                // store startAbs (newest-first)
                if (starts_count < (int)(SEGMENTS * POLY_COUNT)) starts[starts_count++] = startAbs;
                cursor = startAbs;
            }
        }
        // draw in reverse (oldest->newest)
        for (int i = starts_count - 1; i >= 0; --i) {
            uint32_t startAbs = starts[i];
            if (startAbs < windowStart || startAbs > windowEnd) continue;
            int x = tsToX(startAbs);
            // color: change color at segment boundaries approximated by comparing adjacent starts (coarse)
            uint16_t col = TFT_ORANGE;
            // map i to segment change detection not done here (keeps code single-pass and no extra metadata)
            tft.drawFastVLine(x, ry, rh, col);
        }
        // boundary at compressedEnd
        if (compressedEnd >= windowStart && compressedEnd <= windowEnd) {
            int xb = tsToX(compressedEnd);
            tft.drawFastVLine(xb, ry, rh, TFT_MAGENTA);
        }
    }

    // Draw compressed curve by sampling per-pixel (direct evaluation)
    if (drawCompressed && segmentCount > 0) {
        int lastY = -1;
        for (int px = 0; px < rw; ++px) {
            uint32_t ts = windowStart + (uint64_t)px * (uint64_t)(windowEnd - windowStart) / (uint32_t)max(1, rw - 1);
            double v;
            if (!compressedValueAtDirect(ts, v)) continue;
            int x = rx + px;
            int y = valueToY(v);
            if (drawLines && lastY >= 0) tft.drawLine(x - 1, lastY, x, y, TFT_YELLOW);
            else tft.drawPixel(x, y, TFT_YELLOW);
            lastY = y;
        }
    }

    // Draw raw samples overlay (direct)
    if (drawRaw && raw_dataIndex > 0) {
        int prevX = -1, prevY = -1;
        for (uint16_t i = 0; i < raw_dataIndex; ++i) {
            uint32_t t = raw_timestamps[i];
            if (t < windowStart || t > windowEnd) continue;
            int x = tsToX(t);
            int y = valueToY(raw_Data[i]);
            tft.drawPixel(x, y, TFT_CYAN);
            if (drawLines && prevX >= 0) tft.drawLine(prevX, prevY, x, y, TFT_CYAN);
            prevX = x; prevY = y;
        }
    }

    tft.drawRect(rx, ry, rw, rh, TFT_WHITE);
}



//==================================================================

void loop() {
    // Simulate sampling at random intervals
    delay(random(10, 200)); // Random delay between 50 ms to 500 ms
    uint32_t currentTimestamp = millis();

    // Sample scalar data
    float sampledData = sampleScalarData(currentTimestamp);

    // Log the sampled data
    logSampledData(sampledData, currentTimestamp/1.0);

    // Log in the raw form (for debug purposes)
    raw_logSampledData(sampledData, currentTimestamp/1.0);

/* old drawing system 
     // tft.fillScreen(TFT_BLACK);
     uint32_t windowStart = raw_timestamps[0];
     uint32_t windowEnd = raw_timestamps[raw_dataIndex -1] ;
    // updateMinMax(segmentBuffer, segmentCount,currentPolyIndex,windowStart,windowEnd,false,true);
     // Update the raw data graph
     drawRawGraph();
     
    // Update the compressed data graph
//     updateCompressedGraphBackwards(segmentBuffer, segmentCount,currentPolyIndex);
     updateCompressedGraphBackwardsFastOpt(segmentBuffer, segmentCount,currentPolyIndex,true,true); 
                                          // buffer_segment[n].coeffs[degree],segment_count,poly_index_in_last_seg, if_clear_under, if_draw_lines 
     drawRawGraph();
     updateCompressedGraphBackwardsFastOpt(segmentBuffer, segmentCount,currentPolyIndex,false,true); // again to reduce flicker

*/ 
        // For demonstration: assume raw_timestamps[] has been updated by raw_logSampledData()
    // choose window size (e.g. 60s = 60000 ms). Use 0 to mean entire available span.
//    uint32_t windowDurationMs = 120000; // 60 seconds; change as needed
    uint32_t windowDurationMs = 0; // 0 requests all time

    uint32_t windowEnd = (raw_dataIndex > 0) ? raw_timestamps[raw_dataIndex - 1] : millis();
    // Draw graph at arbitrary position/size
/*
    drawCompoundGraph(tft,
                      8, 8, SCREEN_WIDTH - 16, SCREEN_HEIGHT - 16,
                      windowEnd,
                      windowDurationMs,
                      true,   // drawCompressed
                      true,   // drawRaw
                      true,   // clearArea
                      true,   // drawGrid
                      true    // drawLines
                      );
*/
    drawCompoundGraphAligned(tft,
                      8, 8, SCREEN_WIDTH - 16, SCREEN_HEIGHT - 16,
                      windowEnd,
                      windowDurationMs,
                      true,   // drawCompressed
                      true,   // drawRaw
                      true,   // clearArea
                      true,   // drawGrid
                      true    // drawLines
                      );
 
                      
 
}
