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
#define SUB_FIT_POLY_DEGREE 4 //  supplementary fitter to extend boundary transitions .  
                    // the simpler the poly the better prediction , but the more complex data in the window the more chances it will cause misfit. 
                    // choose this up to your data. 
    #define BOUNDARY_MARGIN3 2 // duplicate data across margin for better fit, multiple of 2
    #define BOUNDARY_DELTA3 10 // time window of margin.                  
    #define BOUNDARY_MARGIN 2 // duplicate data across margin for better fit, multiple of 2
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

// Rolling buffer indices
static uint8_t head = 0; // Points to the oldest segment
//uint8_t tail = 0; // Points to the newest segment
//uint8_t segmentCount = 0; // Number of valid segments in buffer


//---- various helper functions
static float mapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

static uint32_t mapUint(uint32_t x, uint32_t in_min, uint32_t in_max, uint32_t out_min, uint32_t out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

//// Add normalization helper functions
//static float normalizeTime(float t, float tMax) {
//    return t / tMax;
//}

// Add normalization helper functions
static double normalizeTime(double t, double tMax) {
    return t / tMax;
}


static float denormalizeTime(float tNorm, float tMax) {
    return tNorm * tMax;
}

// Add helper function for already normalized time evaluation
static double evaluatePolynomialNormalized(const float *coefficients, uint8_t degree, double tNorm) {
    double result = 0.0;
    double tPower = 1.0;  // tNorm^0 = 1
    
    for (int i = 0; i < degree; i++) {
        result += coefficients[i] * tPower;
        tPower *= tNorm;  // More efficient than pow()
    }
    return result;
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


static void compressDataToSegment(const PolynomialSegment *segments, uint8_t count, uint16_t polyindex, const float *rawData, const uint32_t *timestamps, uint16_t dataSize, float *coefficients, uint32_t &timeDelta) {
    AdvancedPolynomialFitter fitter;
    int8_t segmentIndex = count - 1;
    int16_t polyIndex = polyindex; 
    
   // #define BOUNDARY_MARGIN3 16
    //#define BOUNDARY_DELTA3 100
    std::vector<double> x3(dataSize + BOUNDARY_MARGIN3);
    std::vector<float> y3(dataSize + BOUNDARY_MARGIN3);

    float timestamp_absolute = 0;
    
    // Calculate total time span for normalization
    float totalTime = 0;
    for (uint16_t j = 0; j < dataSize; j++) {
        totalTime += timestamps[j];
    }

    // Add boundary points with normalized timestamps
    for (uint8_t i = 0; i < BOUNDARY_MARGIN3; i++) {
        x3[i] = normalizeTime(-(((BOUNDARY_MARGIN3)-i)*BOUNDARY_DELTA3), totalTime);
        if(currentPolyIndex == 0) {
            if (segmentIndex > 0) {
                uint32_t previous_poly_range = segments[segmentIndex-1].timeDeltas[POLY_COUNT-1];
                float tNorm = normalizeTime(previous_poly_range + x3[i] * totalTime, previous_poly_range);
                y3[i] = evaluatePolynomial(segments[segmentIndex-1].coefficients[POLY_COUNT-1], POLY_DEGREE+1, tNorm);
            } else {
                y3[i] = rawData[0];
            }
        } else {
            uint32_t previous_poly_range = segments[segmentIndex].timeDeltas[polyIndex-1];
            float tNorm = normalizeTime(previous_poly_range + x3[i] * totalTime, previous_poly_range);
            y3[i] = evaluatePolynomial(segments[segmentIndex].coefficients[polyIndex-1], POLY_DEGREE+1, tNorm);
        }
    }

    // Add main data points with normalized timestamps
    for (uint16_t j = 0; j < dataSize; j++) {
        timestamp_absolute += timestamps[j];
        x3[j + BOUNDARY_MARGIN3] = normalizeTime(timestamp_absolute, totalTime);
        y3[j + BOUNDARY_MARGIN3] = rawData[j];
    }

    // Fit polynomial with normalized x values
//    std::vector<float> fitted_coefficients3 = fitter.fitPolynomial(x3, y3, SUB_FIT_POLY_DEGREE, AdvancedPolynomialFitter::NONE);
    std::vector<float> fitted_coefficients3 = fitter.fitPolynomialD_superpos5c(x3, y3, SUB_FIT_POLY_DEGREE, AdvancedPolynomialFitter::NONE);


    // Store coefficients (these are now for normalized domain)
    for (uint8_t j = 0; j < fitted_coefficients3.size() && j < SUB_FIT_POLY_DEGREE+1; j++) {
        coefficients[j] = fitted_coefficients3[j];
    }

    // Now handle the main polynomial fitting with normalization
    //#define BOUNDARY_MARGIN 16
    //#define BOUNDARY_DELTA 100
    std::vector<double> x(dataSize + BOUNDARY_MARGIN);
    std::vector<float> y(dataSize + BOUNDARY_MARGIN);

    timestamp_absolute = 0;

    // Add boundary points with normalized timestamps
    for (uint8_t i = 0; i < BOUNDARY_MARGIN/2; i++) {
        x[i] = normalizeTime(-(((BOUNDARY_MARGIN/2)-i)*BOUNDARY_DELTA), totalTime);
        if(currentPolyIndex == 0) {
            if (segmentIndex > 0) {
                uint32_t previous_poly_range = segments[segmentIndex-1].timeDeltas[POLY_COUNT-1];
                float tNorm = normalizeTime(previous_poly_range + x[i] * totalTime, previous_poly_range);
                y[i] = evaluatePolynomial(segments[segmentIndex-1].coefficients[POLY_COUNT-1], POLY_DEGREE+1, tNorm);
            } else {
                y[i] = evaluatePolynomial(coefficients, SUB_FIT_POLY_DEGREE+1, x[i]);
            }
        } else {
            uint32_t previous_poly_range = segments[segmentIndex].timeDeltas[polyIndex-1];
            float tNorm = normalizeTime(previous_poly_range + x[i] * totalTime, previous_poly_range);
            y[i] = evaluatePolynomial(segments[segmentIndex].coefficients[polyIndex-1], POLY_DEGREE+1, tNorm);
        }
    }

    // Add main data points with normalized timestamps
    timestamp_absolute = 0;
    for (uint16_t j = 0; j < dataSize; j++) {
        timestamp_absolute += timestamps[j];
        x[j + BOUNDARY_MARGIN/2] = normalizeTime(timestamp_absolute, totalTime);
        y[j + BOUNDARY_MARGIN/2] = rawData[j];
    }

    // Add end boundary points
    for (uint8_t i = 0; i < BOUNDARY_MARGIN/2; i++) {
        x[dataSize + BOUNDARY_MARGIN/2 + i] = normalizeTime(timestamp_absolute + (((BOUNDARY_MARGIN/2)*i)*BOUNDARY_DELTA), totalTime);
        y[dataSize + BOUNDARY_MARGIN/2 + i] = evaluatePolynomial(coefficients, SUB_FIT_POLY_DEGREE+1, x[dataSize + BOUNDARY_MARGIN/2 + i]);
    }

    // Fit polynomial with normalized x values
//    std::vector<float> fitted_coefficients = fitter.fitPolynomial(x, y, POLY_DEGREE, AdvancedPolynomialFitter::NONE);
    std::vector<float> fitted_coefficients = fitter.fitPolynomialD_superpos5c(x, y, POLY_DEGREE, AdvancedPolynomialFitter::NONE);


    // Store coefficients (these are now for normalized domain)
    for (uint8_t j = 0; j < fitted_coefficients.size() && j < POLY_DEGREE+1; j++) {
        coefficients[j] = fitted_coefficients[j];
    }
//    Serial.println(timestamp_absolute);
    timeDelta = timestamp_absolute;
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
    float scalar = random(0, 1000*sin((float)timestamp * 0.00002)) / 100.0; // background noise
    scalar = scalar + 10 * sin((float)timestamp * 0.0001)+20*sin((float)timestamp * 0.00001);
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

//==================================================================

void loop() {
    // Simulate sampling at random intervals
    delay(random(100, 2000)); // Random delay between 50 ms to 500 ms
    uint32_t currentTimestamp = millis();

    // Sample scalar data
    float sampledData = sampleScalarData(currentTimestamp);

    // Log the sampled data
    logSampledData(sampledData, currentTimestamp/1.0);

    // Log in the raw form (for debug purposes)
    raw_logSampledData(sampledData, currentTimestamp/1.0);

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
 
}
