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
static float    raw_Data[MAX_RAW_DATA];
static uint32_t raw_timestamps[MAX_RAW_DATA];
static uint16_t raw_dataIndex = 0;

DataVisualizer dataVisualizer(tft, dataCompressor, raw_Data, raw_timestamps, raw_dataIndex);


// =================================================================================================
// Helper Functions
// =================================================================================================

/**
 * @brief Logs raw data into a circular buffer for visualization.
 */
static void raw_logSampledData(float data, uint32_t currentTimestamp) {
    if (raw_dataIndex >= MAX_RAW_DATA) {
        for (uint16_t i = 0; i < MAX_RAW_DATA - 1; i++) {
            raw_Data[i] = raw_Data[i + 1];
            raw_timestamps[i] = raw_timestamps[i + 1];
        }
        raw_dataIndex = MAX_RAW_DATA - 1;
    }
    raw_Data[raw_dataIndex] = data;
    raw_timestamps[raw_dataIndex] = currentTimestamp;
    raw_dataIndex++;
}

/**
 * @brief Generates simulated sample data.
 */
static float sampleScalarData(uint32_t timestamp) {
    float scalar = sin((float)timestamp * 0.0002) * 50.0;
    scalar += sin((float)timestamp * 0.001) * 20.0;
    scalar += sin((float)timestamp * 0.005) * 10.0;
    return scalar;
}

// =================================================================================================
// Arduino Setup and Loop
// =================================================================================================

void setup() {
    Serial.begin(115200);
    randomSeed(analogRead(0));
    tft.init();
    tft.setRotation(1);
    tft.fillScreen(TFT_BLACK);
}

void loop() {
    // Simulate sampling data at random intervals.
    delay(random(50, 200));
    uint32_t currentTimestamp = millis();
    float sampledData = sampleScalarData(currentTimestamp);

    // Log the data with the compressor.
    dataCompressor.logSampledData(sampledData, currentTimestamp);
    // Also log the raw data for visualization.
    raw_logSampledData(sampledData, currentTimestamp);

    // Define the time window for the graph.
    uint32_t windowDurationMs = 60000;
    uint32_t windowEnd = (raw_dataIndex > 0) ? raw_timestamps[raw_dataIndex - 1] : millis();

    // Draw the graph using the visualizer.
    dataVisualizer.drawCompoundGraph(8, 8, SCREEN_WIDTH - 16, SCREEN_HEIGHT - 16, windowEnd, windowDurationMs);
}