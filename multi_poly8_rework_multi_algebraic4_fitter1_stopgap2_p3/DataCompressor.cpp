#include "DataCompressor.hpp"
#include "AdvancedPolynomialFitter.hpp"

// Small helper function, can be moved to a common utility file if needed
static inline double normalizeTime(double t, double tMax) { return t / tMax; }

// Constructor
DataCompressor::DataCompressor() : segmentCount(0), currentPolyIndex(0), lastTimestamp(0), raw_log_delta(0), head(0), dataIndex(0) {
    // Initialize buffer
}

// Add a new segment to the buffer
void DataCompressor::addSegment(const PolynomialSegment &newSegment) {
    uint8_t tail = (head + segmentCount) % SEGMENTS;
    segmentBuffer[tail] = newSegment;
    if (segmentCount < SEGMENTS) {
        segmentCount++;
    } else {
        // The buffer is full, overwrite the oldest segment
        head = (head + 1) % SEGMENTS;
    }
}

// Check if the buffer is full
bool DataCompressor::isBufferFull() const {
    return segmentCount == SEGMENTS;
}

// Retrieve the oldest and second-oldest segments
void DataCompressor::getOldestSegments(PolynomialSegment &oldest, PolynomialSegment &secondOldest) const {
    oldest = segmentBuffer[head];
    secondOldest = segmentBuffer[(head + 1) % SEGMENTS];
}

// Remove the oldest two segments
void DataCompressor::removeOldestTwo() {
    if (segmentCount >= 2) {
        head = (head + 2) % SEGMENTS;
        segmentCount -= 2;
    }
}


void DataCompressor::logSampledData(const float* data, uint32_t currentTimestamp) {
    uint32_t timeDelta = (currentTimestamp - lastTimestamp);
    lastTimestamp = currentTimestamp;

    for (int i = 0; i < NUM_DATA_SERIES; ++i) {
        rawDataBuffer[dataIndex][i] = data[i];
    }
    timestampsBuffer[dataIndex] = timeDelta;
    dataIndex++;
    raw_log_delta += timeDelta;

   if (dataIndex >= LOG_BUFFER_POINTS_PER_POLY) {
        if (segmentCount == 0) {
           addSegment(PolynomialSegment());
            currentPolyIndex = 0 ;
            for (uint16_t i = 0; i < POLY_COUNT; i++) {
                uint8_t tail = (head + segmentCount - 1) % SEGMENTS;
                segmentBuffer[tail].timeDeltas[i] = 0;
            }
        }

        if (currentPolyIndex >= POLY_COUNT) {
            if (isBufferFull()) {
                recompressSegments();
            } else {
                Serial.print("Created new segment ");
                Serial.println(segmentCount);
            }
            addSegment(PolynomialSegment());
            currentPolyIndex = 0;
            for (uint16_t i = 0; i < POLY_COUNT; i++) {
                uint8_t tail = (head + segmentCount - 1) % SEGMENTS;
                segmentBuffer[tail].timeDeltas[i] = 0;
            }
        }

        uint8_t tail = (head + segmentCount - 1) % SEGMENTS;
        uint32_t new_timeDelta = 0;
        for (int i = 0; i < NUM_DATA_SERIES; ++i) {
            float new_coefficients[POLY_DEGREE + 1];
            compressDataToSegment(i, timestampsBuffer, dataIndex, new_coefficients, new_timeDelta);
            for (int j = 0; j < POLY_DEGREE + 1; ++j) {
                segmentBuffer[tail].coefficients[currentPolyIndex][i][j] = new_coefficients[j];
            }
        }
        segmentBuffer[tail].timeDeltas[currentPolyIndex] = new_timeDelta;

        Serial.print("Added polynomial ");
        Serial.print(currentPolyIndex);
        Serial.print(" to segment ");
        Serial.println(segmentCount - 1);
        //delay(10000);// debug - simulate compression delay to produce timestamp skew
        currentPolyIndex++;
        raw_log_delta = 0;       
        dataIndex = 0;
        //duplicate current data point at the beginning of new series
            for (int i = 0; i < NUM_DATA_SERIES; ++i) {
        rawDataBuffer[dataIndex][i] = data[i];
            }
        timestampsBuffer[dataIndex] = 0; // 0 time delta because it is last point from last series

        dataIndex++ ; // data index is 1 now. 
            
    }
}

// This function needs access to evaluatePolynomial, which should also be part of the class or a utility.
// For now, we'll keep it as a static function inside this file.
static double evaluatePolynomial(const float *coefficients, uint8_t degree, double t) {
    double result = 0.0;
    double tPower = 1.0;
    for (int i = 0; i < degree; i++) {
        result += coefficients[i] * tPower;
        tPower *= t;
    }
    return result;
}

void DataCompressor::compressDataToSegment(uint8_t seriesIndex, const uint32_t* timestamps, uint16_t dataSize, float* coefficients, uint32_t& timeDelta) {
    AdvancedPolynomialFitter fitter;
    int8_t segmentIndex = segmentCount - 1;
    int16_t polyIndex = (int16_t)currentPolyIndex;

    const int preMargin3 = BOUNDARY_MARGIN3;
    const int preMargin = BOUNDARY_MARGIN / 2;
    const int postMargin = BOUNDARY_MARGIN / 2;

    uint64_t totalTimeU64 = 0;
    for (uint16_t j = 0; j < dataSize; ++j) totalTimeU64 += timestamps[j];
    double totalTime = double(totalTimeU64);
    if (totalTime <= 0.0) totalTime = 1.0;

    auto convertAbsToNormalized = [&](const std::vector<float>& absCoeffs)->std::vector<float> {
        int m = (int)absCoeffs.size();
        std::vector<float> out(m, 0.0f);
        double scalePow = 1.0;
        for (int k = 0; k < m; ++k) {
            out[k] = static_cast<float>(double(absCoeffs[k]) * scalePow);
            scalePow *= totalTime;
        }
        return out;
    };

    auto eval_previous_poly_at_abs = [&](double abs_t, float fallback, int prev_seg_idx, int prev_poly_idx)->float {
        if (prev_seg_idx < 0 || prev_seg_idx >= (int)segmentCount) return fallback;
        if (prev_poly_idx < 0 || prev_poly_idx >= POLY_COUNT) return fallback;
        uint32_t prev_duration = segmentBuffer[prev_seg_idx].timeDeltas[prev_poly_idx];
        if (prev_duration == 0) return fallback;
        double t_local = double(prev_duration) + abs_t;
        if (t_local < 0.0) t_local = 0.0;
        if (t_local > double(prev_duration)) t_local = double(prev_duration);
        float tNorm = normalizeTime((uint32_t)std::lround(t_local), prev_duration);
        return evaluatePolynomial(segmentBuffer[prev_seg_idx].coefficients[prev_poly_idx][seriesIndex], POLY_DEGREE+1, tNorm);
    };

    auto eval_next_poly_at_abs = [&](double abs_t, float fallback, int next_seg_idx, int next_poly_idx)->float {
        if (next_seg_idx < 0 || next_seg_idx >= (int)segmentCount) return fallback;
        if (next_poly_idx < 0 || next_poly_idx >= POLY_COUNT) return fallback;
        uint32_t next_duration = segmentBuffer[next_seg_idx].timeDeltas[next_poly_idx];
        if (next_duration == 0) return fallback;
        double t_local = abs_t;
        if (t_local < 0.0) t_local = 0.0;
        if (t_local > double(next_duration)) t_local = double(next_duration);
        float tNorm = normalizeTime((uint32_t)std::lround(t_local), next_duration);
        return evaluatePolynomial(segmentBuffer[next_seg_idx].coefficients[next_poly_idx][seriesIndex], POLY_DEGREE+1, tNorm);
    };

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
            values[j + preMargin3] = rawDataBuffer[j][seriesIndex];
        }
        int prev_seg_idx = (segmentIndex > 0) ? (segmentIndex - 1) : -1;
        int prev_poly_idx = POLY_COUNT - 1;
        for (int i = 0; i < preMargin3; ++i) {
            double at = absTimes[i];
            float fallback = rawDataBuffer[0][seriesIndex];
            float neighborVal;
            if (polyIndex == 0) neighborVal = eval_previous_poly_at_abs(at, fallback, prev_seg_idx, prev_poly_idx);
            else neighborVal = eval_previous_poly_at_abs(at, fallback, segmentIndex, polyIndex - 1);
            double t = (preMargin3 <= 1) ? 0.0 : (double(i) / double(preMargin3 - 1));
            double w = 0.5 * (1.0 - cos(M_PI * (1.0 - t)));
            values[i] = float(w * neighborVal + (1.0 - w) * fallback);
        }
#ifdef USE_LOW_MEMORY_FITTER
        std::vector<float> fitted_sub_abs = fitter.fitPolynomialD_lowmem(absTimes, values, SUB_FIT_POLY_DEGREE, 0.0);
#else
        std::vector<float> fitted_sub_abs = fitter.fitPolynomialD_superpos5c(absTimes, values, SUB_FIT_POLY_DEGREE, AdvancedPolynomialFitter::NONE);
#endif
        std::vector<float> fitted_sub_norm = convertAbsToNormalized(fitted_sub_abs);
        for (size_t k = 0; k < fitted_sub_norm.size() && k < (SUB_FIT_POLY_DEGREE + 1); ++k) {
            coefficients[k] = fitted_sub_norm[k];
        }
    }

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
            values[j + preMargin] = rawDataBuffer[j][seriesIndex];
        }
        double dataEndAbs = tAbs;
        for (int i = 0; i < postMargin; ++i)
            absTimes[dataSize + preMargin + i] = dataEndAbs + double((i + 1) * BOUNDARY_DELTA);
        int prev_seg_idx = (segmentIndex > 0) ? (segmentIndex - 1) : -1;
        int prev_poly_idx = POLY_COUNT - 1;
        for (int i = 0; i < preMargin; ++i) {
            double at = absTimes[i];
            uint32_t t_clamped = (at <= 0.0) ? 0U : ( (at >= totalTime) ? (uint32_t)std::lround(totalTime) : (uint32_t)std::lround(at) );
            float fallback = evaluatePolynomial(coefficients, SUB_FIT_POLY_DEGREE+1, normalizeTime(t_clamped, (uint32_t)std::lround(totalTime)));
            float neighborVal;
            if (polyIndex == 0) neighborVal = eval_previous_poly_at_abs(at, fallback, prev_seg_idx, prev_poly_idx);
            else neighborVal = eval_previous_poly_at_abs(at, fallback, segmentIndex, polyIndex - 1);
            double t = double(preMargin - i) / double(preMargin + 1);
            double w = 0.5 * (1.0 - cos(M_PI * t));
            values[i] = float(w * neighborVal + (1.0 - w) * fallback);
        }
        int next_seg_idx = (segmentIndex + 1 < segmentCount) ? (segmentIndex + 1) : -1;
        int next_poly_idx = 0;
        for (int i = 0; i < postMargin; ++i) {
            size_t idx = dataSize + preMargin + i;
            double at = absTimes[idx];
            uint32_t t_clamped = (at <= 0.0) ? 0U : ( (at >= totalTime) ? (uint32_t)std::lround(totalTime) : (uint32_t)std::lround(at) );
            float fallback = evaluatePolynomial(coefficients, SUB_FIT_POLY_DEGREE+1, normalizeTime(t_clamped, (uint32_t)std::lround(totalTime)));
            float neighborVal;
            if (polyIndex == (POLY_COUNT - 1)) neighborVal = eval_next_poly_at_abs(at, fallback, next_seg_idx, next_poly_idx);
            else neighborVal = eval_next_poly_at_abs(at, fallback, segmentIndex, polyIndex + 1);
            double t = double(i + 1) / double(postMargin + 1);
            double w = 0.5 * (1.0 - cos(M_PI * t));
            values[idx] = float(w * neighborVal + (1.0 - w) * fallback);
        }
#ifdef USE_LOW_MEMORY_FITTER
        std::vector<float> fitted_abs = fitter.fitPolynomialD_lowmem(absTimes, values, POLY_DEGREE, 0.0);
#else
        std::vector<float> fitted_abs = fitter.fitPolynomialD_superpos5c(absTimes, values, POLY_DEGREE, AdvancedPolynomialFitter::NONE);
#endif
        std::vector<float> fitted_norm = convertAbsToNormalized(fitted_abs);
        for (size_t k = 0; k < fitted_norm.size() && k < (POLY_DEGREE + 1); ++k) {
            coefficients[k] = fitted_norm[k];
        }
        timeDelta = (uint32_t)std::lround(dataEndAbs);
    }
}


void DataCompressor::combinePolynomials(const PolynomialSegment &oldest, const PolynomialSegment &secondOldest, PolynomialSegment &recompressedSegment) {
    AdvancedPolynomialFitter fitter;
    uint16_t currentPolyIndex_local = 0;

    for (uint16_t i = 0; i < POLY_COUNT; i = i + 2) {
        if (i + 1 >= POLY_COUNT || oldest.timeDeltas[i] == 0 || oldest.timeDeltas[i+1] == 0) break;

        double combinedTimeDelta = oldest.timeDeltas[i] + oldest.timeDeltas[i+1];
        for (int series = 0; series < NUM_DATA_SERIES; ++series) {
            std::vector<float> newCoefficients = fitter.composePolynomials(oldest.coefficients[i][series], oldest.timeDeltas[i], oldest.coefficients[i+1][series], oldest.timeDeltas[i+1], POLY_DEGREE);
            for (uint8_t j = 0; j < newCoefficients.size() && j < POLY_DEGREE + 1; j++) {
                recompressedSegment.coefficients[currentPolyIndex_local][series][j] = newCoefficients[j];
            }
        }
        recompressedSegment.timeDeltas[currentPolyIndex_local] = combinedTimeDelta;
        currentPolyIndex_local++;
    }

    for (uint16_t i = 0; i < POLY_COUNT; i = i + 2) {
        if (i + 1 >= POLY_COUNT || secondOldest.timeDeltas[i] == 0 || secondOldest.timeDeltas[i+1] == 0) break;

        double combinedTimeDelta = secondOldest.timeDeltas[i] + secondOldest.timeDeltas[i+1];
        for (int series = 0; series < NUM_DATA_SERIES; ++series) {
            std::vector<float> newCoefficients = fitter.composePolynomials(secondOldest.coefficients[i][series], secondOldest.timeDeltas[i], secondOldest.coefficients[i+1][series], secondOldest.timeDeltas[i+1], POLY_DEGREE);
            for (uint8_t j = 0; j < newCoefficients.size() && j < POLY_DEGREE + 1; j++) {
                recompressedSegment.coefficients[currentPolyIndex_local][series][j] = newCoefficients[j];
            }
        }
        recompressedSegment.timeDeltas[currentPolyIndex_local] = combinedTimeDelta;
        currentPolyIndex_local++;
    }
}

void DataCompressor::recompressSegments() {
    if (segmentCount < 2) return;

    PolynomialSegment oldest, secondOldest;
    getOldestSegments(oldest, secondOldest);

    PolynomialSegment recompressedSegment;
    for (uint16_t i = 0; i < POLY_COUNT; i++) {
        recompressedSegment.timeDeltas[i] = 0;
    }

    combinePolynomials(oldest, secondOldest, recompressedSegment);

    // Shift the remaining segments to the left
    for (uint8_t i = 0; i < segmentCount - 2; i++) {
        segmentBuffer[(head + i) % SEGMENTS] = segmentBuffer[(head + i + 2) % SEGMENTS];
    }

    // Add the recompressed segment at the end of the valid data
    segmentBuffer[(head + segmentCount - 2) % SEGMENTS] = recompressedSegment;

    // Decrease the segment count by one
    segmentCount--;
    Serial.print("Recompressed. New segment count: ");
    Serial.println(segmentCount);
    Serial.print("poly size: ");
    Serial.print(sizeof(segmentBuffer));
}
