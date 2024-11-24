# Ride-Pooling System

## Overview
This system implements a sophisticated ride-pooling algorithm that matches and combines rides to improve transportation efficiency. The system analyzes ride requests and attempts to combine compatible trips while maintaining service quality and minimizing detours.

## Core Concepts

### Ride Pooling
Ride pooling is the concept of combining multiple ride requests into a single vehicle trip. For example, if two passengers are traveling in similar directions at similar times, they can share a ride, reducing the total number of vehicles needed and potentially lowering costs.

### Matching
Matching is the process of finding compatible ride requests that can be combined. The system considers several factors:
- Spatial proximity (pickup and dropoff locations)
- Temporal compatibility (pickup times)
- Directional similarity (travel directions)
- Trip characteristics (distance, duration)

### Detour
A detour is the additional distance or time added to a trip when combining rides. For example:
- Original Trip A: 10km
- Original Trip B: 8km
- Combined Trip: 20km
- Detour Ratio = 20/(10+8) = 1.11 (11% increase)

## Key Components

### 1. Data Processing
- Loads ride request data from JSON files
- Converts timestamps and coordinates
- Validates and filters requests

### 2. Demand Analysis
The `DemandAnalyzer` class:
- Creates spatial demand heat maps
- Analyzes temporal patterns
- Calculates demand scores for locations and times

### 3. Matching System
The matching algorithm considers:
```python
max_detour_ratio = 1.5      # Maximum allowed detour
max_pickup_distance = 2000  # Maximum distance between pickups (meters)
max_wait_time = 900        # Maximum wait time (seconds)
max_time_window = 15       # Maximum time difference (minutes)
```

### 4. Routing
- Uses Valhalla routing engine for accurate route calculations
- Implements route caching for performance
- Calculates optimal pickup and dropoff sequences

## Matching Criteria

### Basic Feasibility
1. Pickup Distance
   - Must be within `max_pickup_distance`
   - Calculated using haversine formula

2. Time Window
   - Pickup times must be within `max_time_window`
   - Prevents excessive waiting

3. Direction Compatibility
   - Maximum 60-degree difference in travel directions
   - Prevents significant detours

### Scoring Factors
The system calculates a matching score (0-1) based on:
1. Distance Efficiency (30%)
   - How much extra distance is added by combining trips
2. Time Compatibility (25%)
   - How well the pickup times align
3. Pickup Convenience (20%)
   - How close the pickup locations are
4. Demand Patterns (15%)
   - Based on historical demand in the area
5. Direction Compatibility (10%)
   - How similar the travel directions are

## Results and Visualizations

### 1. Temporal Analysis
- Distribution of successful pools by hour
- Success rates throughout the day
- Peak pooling periods

Example visualization:
```
Hour of Day | Success Rate | Number of Pools
    8-9     |    45%      |      156
    9-10    |    38%      |      143
    ...
```

### 2. Spatial Analysis
- Heatmap of successful pooling locations
- Cluster analysis of popular routes
- Interactive maps showing:
  - Pickup points (green)
  - Dropoff points (red)
  - Combined routes (blue)

### 3. Performance Metrics
The system tracks:
1. Pooling Rate
   - Percentage of rides successfully pooled
2. Average Detour Ratio
   - How much extra distance is added
3. Time Savings
   - Reduction in total vehicle hours
4. Distance Savings
   - Reduction in total vehicle kilometers

### 4. Distribution Analysis
- Detour ratio distribution
- Time difference distribution
- Pickup distance distribution
- Matching score distribution

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running the System
```bash
python pooling-system.py \
  --data your_data.json \
  --output results_dir \
  --max-detour 1.5 \
  --max-pickup-distance 2000
```

### Configuration Parameters
```python
class PoolingSystem:
    def __init__(
        self,
        max_detour_ratio=1.5,      # Maximum allowed detour
        max_pickup_distance=2000,   # Maximum pickup distance (meters)
        max_wait_time=900,         # Maximum wait time (seconds)
        max_time_window=15,        # Maximum time window (minutes)
        min_savings_threshold=0.2   # Minimum required savings
    )
```

## Output Files

### 1. Analysis Report (analysis_report.json)
Contains comprehensive statistics and analysis results.

### 2. Visualizations
- pooling_heatmap.html: Interactive map
- temporal_patterns.png: Time-based analysis
- pooling_statistics.png: Distribution plots

### 3. Matching Results (pooling_results.json)
Detailed information about each successful match.

## Common Patterns and Insights

1. Peak Hours
- Higher pooling success rates during rush hours
- More opportunities for matching

2. Distance Effects
- Short trips (<1km) rarely pool successfully
- Medium trips (2-10km) have highest pooling rates
- Long trips (>20km) have lower pooling rates

3. Geographical Patterns
- Dense urban areas show higher pooling rates
- Suburban areas need larger time windows
- Major corridors have better matching opportunities

## Implementation Considerations

1. Performance Optimization
- Route caching
- Spatial indexing
- Parallel processing

2. Quality Control
- Strict matching criteria
- Buffer zones for constraints
- Comprehensive validation

3. Scalability
- Batch processing
- Memory management
- Cache size limits