Ride Pooling System
Introduction
This project implements a ride pooling system designed to optimize transportation efficiency by matching ride requests for potential pooling opportunities. The system analyzes ride requests, calculates priority scores, and attempts to pool rides based on spatial and temporal proximity, demand patterns, and other factors.

Key Concepts
Ride Pooling
Ride pooling refers to combining two or more ride requests into a single trip, allowing passengers traveling in the same direction to share a vehicle. This approach reduces total travel distance, lowers costs for passengers and service providers, and can alleviate traffic congestion.

Matching
Matching is the process of identifying ride requests that are suitable for pooling. The system evaluates potential matches based on criteria such as pickup location proximity, time window overlap, trip similarity, and overall route efficiency.

Detour
A detour is the additional distance or time added to a trip when accommodating multiple passengers in a pooled ride compared to individual trips. Minimizing detours is crucial to ensure passenger satisfaction and maintain the efficiency of the pooling system.

Code Overview
The code is organized into several classes and functions that work together to implement the ride pooling system.

Classes
RideRequest
A data class representing an individual ride request with attributes:

trip_id: Unique identifier for the trip.
pickup_lat, pickup_lng: Pickup location coordinates.
dropoff_lat, dropoff_lng: Dropoff location coordinates.
pickup_time: Scheduled pickup time as a datetime object.
trip_distance: Distance of the trip.
week_id, date_id, time_id: Temporal identifiers.
straight_line_distance: Direct distance between pickup and dropoff points.
demand_score: Score representing demand at the pickup location and time.
priority_score: Calculated priority for pooling consideration.
PoolingResult
A data class representing the result of a successful pooling attempt, including:

Identifiers for the two rides (ride1_id, ride2_id).
Routing and efficiency metrics such as total_distance, detour_ratio, and estimated_savings.
Details about the pooling order (order_type) and reasoning (pooling_reason).
Additional scores like passenger_impact_score and matching_score.
DemandAnalyzer
Analyzes spatial and temporal demand patterns to calculate demand scores for ride requests.

Builds a demand matrix using pickup locations.
Analyzes temporal patterns based on pickup times.
Provides a get_demand_score method to assign demand scores to requests.
PoolingSystem
The core class that manages the pooling process.

Initializes with parameters like maximum detour ratio, pickup distance, and time window constraints.
Contains methods for calculating priority scores, attempting to pool rides, and generating visualizations.
Utilizes caching for route calculations to improve performance.
Functions
haversine_distance
Calculates the great-circle distance between two points on the Earth's surface, used to estimate distances between pickups and dropoffs.

load_and_process_data
Loads ride request data from a JSON file, filters, and processes it into a list of RideRequest objects.

find_pooling_opportunities
Implements the pooling algorithm by:

Maintaining an active window of ride requests based on time constraints.
Using spatial indexing to find nearby ride requests.
Attempting to pool rides by calling try_pool_rides.
main
The entry point of the script that:

Parses command-line arguments.
Loads and processes data.
Initializes the pooling system.
Analyzes demand patterns.
Calculates priority scores.
Finds pooling opportunities.
Generates visualizations and logs results.
How the Code Works
Data Loading and Processing

The script starts by loading ride request data from a specified JSON file using the load_and_process_data function. It filters out invalid or incomplete data and converts each valid record into a RideRequest object.

Initializing the Pooling System

An instance of PoolingSystem is created with configurable parameters like maximum detour ratio, maximum pickup distance, and time windows. This system will manage the pooling process.

Analyzing Demand Patterns

The DemandAnalyzer analyzes the loaded ride requests to identify spatial and temporal demand patterns. It creates a demand matrix based on pickup locations and time slots, which helps in calculating demand scores for new requests.

Calculating Priority Scores

For each ride request, the calculate_priority_score method computes a priority score. This score is based on factors such as:

Trip distance (longer trips may have higher pooling potential).
Demand score (areas or times with high demand).
Time of day (peak hours may have higher priority).
Straight-line efficiency (ratio of straight-line distance to trip distance).
Finding Pooling Opportunities

The find_pooling_opportunities function processes ride requests sequentially, maintaining an active window of requests that are eligible for pooling based on the time window constraint.

It uses spatial indexing to quickly find nearby ride requests.
For each request, it looks for potential matches among nearby requests.
It sorts potential matches by priority score to attempt pooling higher-priority requests first.
Attempting to Pool Rides

The try_pool_rides method tries to pool two ride requests by:

Checking feasibility based on pickup distance and time difference constraints.
Generating different routing options (e.g., P1→P2→D1→D2).
Calculating routes using the Valhalla routing service.
Evaluating detour ratios and estimated savings.
Calculating a matching score based on multiple factors.
If a suitable route is found, it records the pooling result.
Generating Visualizations

After processing all ride requests, the system generates visualizations to help understand the pooling results:

Heatmaps showing pickup and dropoff locations of successful pools.
Temporal analysis plots displaying the distribution of pools over time.
Statistical plots for metrics like detour ratios, time differences, and matching scores.
Logging and Output

The script logs detailed information throughout the process, including the number of requests processed, pooling attempts, and reasons for rejected pooling attempts. Results and visualizations are saved in the specified output directory.

Results and Output
The outputs of the script include:

Pooling Results

A list of PoolingResult objects representing successful pooling matches, each containing detailed information about the pooled rides and efficiency metrics.

Visualizations

Pooling Heatmap: An interactive map (pooling_heatmap.html) showing the locations of successful pickups and dropoffs.
Temporal Analysis Plots: Images (temporal_analysis.png) illustrating the distribution of successful pools by hour and pooling success rates.
Statistics Plots: Images (pooling_statistics.png) displaying distributions of detour ratios, time differences, pickup distances, and matching scores.
Analysis Statistics

A summary of the pooling process, including:

Total ride requests processed.
Number of successful pools.
Pooling rate (percentage of rides that were successfully pooled).
Reasons for rejected pooling attempts.
All outputs are saved in the specified output directory, organized appropriately for easy access.

Running the Code
Prerequisites
Python 3.6 or higher

Required Python Packages: Install using pip

bash
Copy code
pip install requests numpy pandas folium seaborn matplotlib scikit-learn
Valhalla Routing Service: The code relies on the Valhalla routing engine. Ensure that Valhalla is installed, set up, and running locally or accessible via the specified valhalla_url.

Steps
Prepare the Ride Request Data

Ensure you have a JSON file containing ride request data with the required fields. The data should include information such as trip_id, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng, pickup_time, trip_distance, etc.

Run the Script

Execute the script from the command line:

bash
Copy code
python pooling-system.py --data path/to/ride_requests.json --output pooling_results
Command-Line Arguments:

--data: Path to the input data JSON file (required).
--output: Output directory for results and visualizations (default: pooling_results).
--max-detour: Maximum allowed detour ratio (default: 1.5).
--max-pickup-distance: Maximum pickup distance in meters (default: 2000).
--max-wait-time: Maximum wait time in seconds (default: 900).
--max-time-window: Maximum time window in minutes (default: 15).
Monitor the Process

The script will log progress and important information to the console. You can monitor the number of requests processed, pooling attempts, and any warnings or errors.

Review the Results

After completion, results will be available in the specified output directory:

Pooling Results: Detailed pooling results saved in data structures or potentially serialized files.
Visualizations: Found in the visualizations subdirectory.
pooling_heatmap.html: Open this interactive map in a web browser.
temporal_analysis.png and pooling_statistics.png: View these images to analyze pooling performance.
Understanding the Results
Pooling Heatmap

Visualizes the geographical distribution of successful pooling pickups and dropoffs. It helps identify high-density areas where pooling is more prevalent.

Temporal Analysis Plots

Show how pooling activity varies over time, highlighting peak hours and success rates.

Statistics Plots

Provide insights into key metrics like detour ratios, helping assess the efficiency and passenger impact of the pooling strategy.

Analysis Statistics

Summarize the overall performance, including total requests, successful pools, and pooling rates, giving a quick overview of the system's effectiveness.

Customization and Tuning
You can adjust the pooling parameters to optimize performance based on your specific needs or dataset characteristics:

Detour Ratio: Lowering this value will reduce allowable detours, potentially increasing passenger satisfaction but decreasing pooling opportunities.
Pickup Distance: Adjusting this affects how far the system is willing to consider pickups for pooling.
Time Windows: Modifying the maximum wait time and time window can balance between pooling efficiency and passenger wait times.
Experimenting with these parameters can help you find the optimal balance for your application.

Conclusion
This ride pooling system demonstrates a comprehensive approach to optimizing transportation through intelligent matching and routing. By considering spatial and temporal factors, demand patterns, and efficiency metrics, the system aims to maximize pooling opportunities while maintaining service quality.

The code provides a solid foundation that can be extended or integrated into larger transportation platforms, contributing to more sustainable and efficient urban mobility solutions.

Note: Ensure that all dependencies are correctly installed and that the Valhalla routing service is operational to avoid runtime errors. Always validate input data for correctness and completeness before running the pooling system.


