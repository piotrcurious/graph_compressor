#include <TFT_eSPI.h>
#include "DataCompressor.hpp"
#include "DataVisualizer.hpp"

// =================================================================================================
// Global Objects and Definitions
// =================================================================================================

TFT_eSPI tft = TFT_eSPI();
DataCompressor dataCompressor;

#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 240
#define MAX_RAW_DATA 1440 // Defines the maximum number of raw data points to store for visualization.

// Buffer to hold raw data for the purpose of displaying it on the screen alongside the compressed data.
static float    raw_Data[MAX_RAW_DATA][NUM_DATA_SERIES];
static uint32_t raw_timestamps[MAX_RAW_DATA];
static uint16_t raw_dataIndex = 0;

DataVisualizer dataVisualizer(tft, dataCompressor);


// =================================================================================================
// Helper Functions
// =================================================================================================

/**
 * @brief Logs raw data into a circular buffer for visualization.
 */
static void raw_logSampledData(const float* data, uint32_t currentTimestamp) {
    if (raw_dataIndex >= MAX_RAW_DATA) {
        for (uint16_t i = 0; i < MAX_RAW_DATA - 1; i++) {
            for (int j = 0; j < NUM_DATA_SERIES; ++j) {
                raw_Data[i][j] = raw_Data[i + 1][j];
            }
            raw_timestamps[i] = raw_timestamps[i + 1];
        }
        raw_dataIndex = MAX_RAW_DATA - 1;
    }
    for (int j = 0; j < NUM_DATA_SERIES; ++j) {
        raw_Data[raw_dataIndex][j] = data[j];
    }
    raw_timestamps[raw_dataIndex] = currentTimestamp;
    raw_dataIndex++;
}

/**
 * @brief Generates simulated sample data.
 */
static void sampleVectorData(uint32_t timestamp, float* outData) {
    outData[0] = sin((float)timestamp * 0.00002) * 50.0 + sin((float)timestamp * 0.0001) * 20.0 + sin((float)timestamp * 0.0005) * 10.0;
    outData[1] = cos((float)timestamp * 0.00003) * 40.0 + sin((float)timestamp * 0.00015) * 15.0 + cos((float)timestamp * 0.0006) * 15.0;
}

// =================================================================================================
// Arduino Setup and Loop
// =================================================================================================

void drawRawDataDebug(int rx, int ry, int rw, int rh, uint32_t wStart, uint32_t wEnd, float vmin, float vmax) {
    auto tsToX = [&](uint32_t ts) {
        if (wEnd == wStart) return rx;
        double f = double(ts - wStart) / double(wEnd - wStart);
        return rx + (int)round(f * (rw - 1));
    };

    auto valueToY = [&](double val) {
        double c = val;
        if (c < vmin) c = vmin; if (c > vmax) c = vmax;
        return ry + (int)round((1.0 - (c - vmin) / (vmax - vmin)) * (rh - 1));
    };

    uint16_t colors[] = {TFT_CYAN, TFT_ORANGE};
    for (uint8_t series = 0; series < NUM_DATA_SERIES; ++series) {
        if (raw_dataIndex > 0) {
            int prevX = -1, prevY = -1;
            for (uint16_t i = 0; i < raw_dataIndex; ++i) {
                uint32_t t = raw_timestamps[i];
                if (t < wStart || t > wEnd) continue;
                int x = tsToX(t);
                int y = valueToY(raw_Data[i][series]);
                if (prevX != -1) {
                    tft.drawLine(prevX, prevY, x, y, colors[series]);
                } else {
                    tft.drawPixel(x, y, colors[series]);
                }
                prevX = x; prevY = y;
            }
        }
    }
}

void drawDebugGraph(int rx, int ry, int rw, int rh, uint32_t windowEndAbs, uint32_t windowDurationMs) {
    dataVisualizer.drawCompoundGraph(rx, ry, rw, rh, windowEndAbs, windowDurationMs);

    uint32_t wEnd = windowEndAbs;
    uint32_t wStart = (windowDurationMs == 0 || windowDurationMs > wEnd) ? 0u : (wEnd - windowDurationMs);

    float vmin = -100, vmax = 100;

    drawRawDataDebug(rx, ry, rw, rh, wStart, wEnd, vmin, vmax);
}

void setup() {
    Serial.begin(115200);
    randomSeed(analogRead(0));
    tft.init();
    tft.initDMA();
    tft.setRotation(1);
    tft.fillScreen(TFT_BLACK);
}

void loop() {
    // Simulate sampling data at random intervals.
    delay(random(50, 200));
    uint32_t currentTimestamp = millis();
    float sampledData[NUM_DATA_SERIES];
    sampleVectorData(currentTimestamp, sampledData);

    // Also log the raw data for visualization.
    raw_logSampledData(sampledData, currentTimestamp);
    // Log the data with the compressor.
    dataCompressor.logSampledData(sampledData, currentTimestamp);


    // Define the time window for the graph.
    uint32_t windowDurationMs = 60000*60;
   // uint32_t windowEnd = (raw_dataIndex > 0) ? raw_timestamps[raw_dataIndex - 1] : millis();
    uint32_t windowEnd = (raw_dataIndex > 0) ? raw_timestamps[raw_dataIndex - 1] : currentTimestamp;

    // Draw the production graph (compressed + recent raw data).
    //dataVisualizer.drawCompoundGraph(8, 8, SCREEN_WIDTH - 16, SCREEN_HEIGHT - 16, windowEnd, windowDurationMs);

    // Uncomment the following line to draw the debug graph (production graph + full raw data overlay).
     drawDebugGraph(8, 8, SCREEN_WIDTH - 16, SCREEN_HEIGHT - 16, windowEnd, windowDurationMs);
}
