import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
import logging
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
import folium
from folium import plugins
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass
class RideRequest:
    trip_id: str
    pickup_lat: float
    pickup_lng: float
    dropoff_lat: float
    dropoff_lng: float
    pickup_time: datetime
    trip_distance: float  # Should be in meters
    week_id: int
    date_id: int
    time_id: float
    straight_line_distance: float  # Should be in meters
    demand_score: float = 0.0
    priority_score: float = 0.0

@dataclass
class PoolingResult:
    ride1_id: str
    ride2_id: str
    total_distance: float
    original_distance: float
    detour_ratio: float
    pickup_distance: float
    time_difference: float
    order_type: str
    pooling_reason: str
    route_summary: Dict
    estimated_savings: float
    passenger_impact_score: float
    route_efficiency: float
    matching_score: float

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in meters"""
    R = 6371000  # Earth's radius in meters

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

class DemandAnalyzer:
    def __init__(self, grid_size: int = 50):
        self.grid_size = grid_size
        self.demand_matrix = None
        self.temporal_patterns = defaultdict(list)
        self.lat_edges = None
        self.lng_edges = None

    def analyze_demand(self, requests: List[RideRequest]) -> None:
        lats = [r.pickup_lat for r in requests]
        lngs = [r.pickup_lng for r in requests]

        self.demand_matrix, self.lat_edges, self.lng_edges = np.histogram2d(
            lats, lngs, bins=self.grid_size
        )

        hour_demand = defaultdict(int)
        for request in requests:
            hour = request.pickup_time.hour
            hour_demand[hour] += 1

        self.temporal_patterns = dict(hour_demand)

    def get_demand_score(self, lat: float, lng: float, time: datetime) -> float:
        if self.demand_matrix is None:
            return 0.0

        lat_idx = np.digitize(lat, self.lat_edges) - 1
        lng_idx = np.digitize(lng, self.lng_edges) - 1

        spatial_score = (
            self.demand_matrix[lat_idx, lng_idx]
            if 0 <= lat_idx < self.grid_size and 0 <= lng_idx < self.grid_size
            else 0
        )
        temporal_score = self.temporal_patterns.get(time.hour, 0)

        return 0.7 * spatial_score + 0.3 * temporal_score

class PoolingSystem:
    def __init__(
        self,
        max_detour_ratio: float = 1.2,  # Tightened to 1.2
        max_pickup_distance: float = 500,  # In meters
        max_wait_time: int = 300,  # Reduced to 5 minutes (300 seconds)
        max_time_window: int = 5,  # Reduced to 5 minutes
        valhalla_url: str = "http://localhost:8002",
        output_dir: str = "pooling_results",
        min_savings_threshold: float = 1000  # Increased to 1,000 meters
    ):
        self.max_detour_ratio = max_detour_ratio
        self.max_pickup_distance = max_pickup_distance  # In meters
        self.max_wait_time = max_wait_time
        self.max_time_window = max_time_window  # In minutes
        self.valhalla_url = valhalla_url
        self.output_dir = Path(output_dir)
        self.min_savings_threshold = min_savings_threshold  # In meters

        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)

        self.route_cache = {}
        self.pooling_results = []
        self.rejected_pairs = []
        self.analysis_stats = defaultdict(int)
        self.all_requests = []

        self.demand_analyzer = DemandAnalyzer()

    def calculate_priority_score(self, request: RideRequest) -> float:
        """Calculate priority score for a request"""
        try:
            # 1. Trip distance score (longer trips get higher priority)
            # Normalize distance to [0,1] range, assuming most trips are under 20km (20,000 meters)
            distance_score = min(request.trip_distance / 20000.0, 1.0)

            # 2. Demand in the area
            demand_score = request.demand_score

            # 3. Time of day factor (prioritize peak hours)
            hour = request.pickup_time.hour
            peak_morning = {7, 8, 9}  # Morning peak hours
            peak_evening = {16, 17, 18, 19}  # Evening peak hours

            if hour in peak_morning:
                time_score = 1.0
            elif hour in peak_evening:
                time_score = 0.9
            else:
                time_score = 0.5

            # 4. Calculate straight-line efficiency
            if request.straight_line_distance > 0:
                efficiency_score = min(
                    request.straight_line_distance / request.trip_distance, 1.0
                )
            else:
                efficiency_score = 0.5  # Default if straight_line_distance is not available

            # Combine scores with weights
            weights = {
                'distance': 0.3,
                'demand': 0.3,
                'time': 0.2,
                'efficiency': 0.2
            }

            final_score = (
                weights['distance'] * distance_score +
                weights['demand'] * demand_score +
                weights['time'] * time_score +
                weights['efficiency'] * efficiency_score
            )

            # Normalize to [0,1] range
            return max(0.0, min(1.0, final_score))

        except Exception as e:
            logging.warning(f"Error calculating priority score for request {request.trip_id}: {e}")
            return 0.0  # Return minimum priority score in case of error

    def get_route(self, locations: List[Dict]) -> Optional[Dict]:
        """Get route from Valhalla with caching and validation"""
        cache_key = tuple((loc['lat'], loc['lon']) for loc in locations)

        if cache_key in self.route_cache:
            return self.route_cache[cache_key]

        try:
            response = requests.post(
                f"{self.valhalla_url}/route",
                json={
                    'locations': locations,
                    'costing': 'auto',
                    'directions_options': {'units': 'kilometers'}
                },
                timeout=5
            )

            if response.status_code == 200:
                route = response.json()
                # Validate route distance
                total_distance_km = route['trip']['summary']['length']
                total_distance_m = total_distance_km * 1000  # Convert to meters
                if total_distance_m <= 0:
                    logging.error(f"Invalid route distance: {total_distance_m}")
                    return None
                route['trip']['summary']['length_meters'] = total_distance_m
                self.route_cache[cache_key] = route
                return route
            else:
                logging.error(f"Routing error: Status code {response.status_code}")
                return None

        except Exception as e:
            logging.error(f"Routing error: {e}")
            return None

    def _is_feasible_match(self, pickup_distance: float, time_diff: float,
                           ride1: RideRequest, ride2: RideRequest) -> bool:
        """Check if a match is feasible based on constraints"""
        self.analysis_stats['total_attempts'] += 1  # Count every attempt

        if pickup_distance > self.max_pickup_distance:
            self.rejected_pairs.append({
                'ride1_id': ride1.trip_id,
                'ride2_id': ride2.trip_id,
                'rejection_reason': 'pickup_distance_exceeded',
                'pickup_distance': pickup_distance,
                'max_allowed': self.max_pickup_distance
            })
            self.analysis_stats['rejected_pickup_distance'] += 1
            return False

        if time_diff > self.max_time_window:
            self.rejected_pairs.append({
                'ride1_id': ride1.trip_id,
                'ride2_id': ride2.trip_id,
                'rejection_reason': 'time_window_exceeded',
                'time_difference': time_diff,
                'max_allowed': self.max_time_window
            })
            self.analysis_stats['rejected_time_window'] += 1
            return False

        return True

    def try_pool_rides(self, ride1: RideRequest, ride2: RideRequest) -> Optional[PoolingResult]:
        """Try to pool two rides together"""
        logging.debug(f"Attempting to pool rides {ride1.trip_id} and {ride2.trip_id}")

        # Quick feasibility checks
        pickup_distance = haversine_distance(
            ride1.pickup_lat, ride1.pickup_lng,
            ride2.pickup_lat, ride2.pickup_lng
        )

        time_diff = abs((ride1.pickup_time - ride2.pickup_time).total_seconds()) / 60

        if not self._is_feasible_match(pickup_distance, time_diff, ride1, ride2):
            logging.debug(f"Pooling rejected for rides {ride1.trip_id} and {ride2.trip_id} due to constraints")
            return None

        # Generate and try different route options
        best_result = None
        min_total_distance = float('inf')

        route_options = [
            {
                'sequence': [
                    {'lat': ride1.pickup_lat, 'lon': ride1.pickup_lng},
                    {'lat': ride2.pickup_lat, 'lon': ride2.pickup_lng},
                    {'lat': ride1.dropoff_lat, 'lon': ride1.dropoff_lng},
                    {'lat': ride2.dropoff_lat, 'lon': ride2.dropoff_lng}
                ],
                'type': 'P1->P2->D1->D2'
            },
            {
                'sequence': [
                    {'lat': ride1.pickup_lat, 'lon': ride1.pickup_lng},
                    {'lat': ride2.pickup_lat, 'lon': ride2.pickup_lng},
                    {'lat': ride2.dropoff_lat, 'lon': ride2.dropoff_lng},
                    {'lat': ride1.dropoff_lat, 'lon': ride1.dropoff_lng}
                ],
                'type': 'P1->P2->D2->D1'
            }
        ]

        for option in route_options:
            route = self.get_route(option['sequence'])
            if route:
                total_distance_m = route['trip']['summary']['length_meters']
                original_distance = ride1.trip_distance + ride2.trip_distance
                if original_distance <= 0:
                    logging.error(f"Invalid original distance for rides {ride1.trip_id} and {ride2.trip_id}")
                    continue
                detour_ratio = total_distance_m / original_distance

                if total_distance_m < min_total_distance and detour_ratio <= self.max_detour_ratio:
                    # Calculate efficiency metrics
                    matching_score = self._calculate_matching_score(
                        total_distance_m, original_distance,
                        pickup_distance, time_diff,
                        ride1, ride2
                    )

                    estimated_savings = original_distance - total_distance_m
                    passenger_impact = 1 - (time_diff / self.max_time_window)
                    route_efficiency = 1 - (detour_ratio - 1)

                    if estimated_savings >= self.min_savings_threshold:
                        min_total_distance = total_distance_m
                        best_result = PoolingResult(
                            ride1_id=ride1.trip_id,
                            ride2_id=ride2.trip_id,
                            total_distance=total_distance_m,
                            original_distance=original_distance,
                            detour_ratio=detour_ratio,
                            pickup_distance=pickup_distance,
                            time_difference=time_diff,
                            order_type=option['type'],
                            pooling_reason="feasible_pool",
                            route_summary=route['trip']['summary'],
                            estimated_savings=estimated_savings,
                            passenger_impact_score=passenger_impact,
                            route_efficiency=route_efficiency,
                            matching_score=matching_score
                        )

        if best_result:
            self.pooling_results.append(best_result)
            self.analysis_stats['successful_pools'] += 1
            logging.debug(f"Pooling successful for rides {ride1.trip_id} and {ride2.trip_id}")
        else:
            self.rejected_pairs.append({
                'ride1_id': ride1.trip_id,
                'ride2_id': ride2.trip_id,
                'rejection_reason': 'no_efficient_route_found',
                'min_total_distance': min_total_distance if min_total_distance != float('inf') else None
            })
            self.analysis_stats['rejected_no_efficient_route'] += 1
            logging.debug(f"No efficient route found for rides {ride1.trip_id} and {ride2.trip_id}")

        return best_result

    def _calculate_matching_score(
        self,
        total_distance: float,
        original_distance: float,
        pickup_distance: float,
        time_diff: float,
        ride1: RideRequest,
        ride2: RideRequest
    ) -> float:
        """Calculate sophisticated matching score"""
        # Distance efficiency (0-1)
        distance_efficiency = 1 - (total_distance - original_distance) / original_distance

        # Time compatibility (0-1)
        time_compatibility = 1 - (time_diff / self.max_time_window)

        # Pickup convenience (0-1)
        pickup_convenience = 1 - (pickup_distance / self.max_pickup_distance)

        # Demand compatibility (0-1)
        demand_compatibility = (ride1.demand_score + ride2.demand_score) / 2

        # Weights for different factors
        weights = {
            'distance_efficiency': 0.35,
            'time_compatibility': 0.25,
            'pickup_convenience': 0.25,
            'demand_compatibility': 0.15
        }

        # Calculate weighted score
        score = (
            weights['distance_efficiency'] * distance_efficiency +
            weights['time_compatibility'] * time_compatibility +
            weights['pickup_convenience'] * pickup_convenience +
            weights['demand_compatibility'] * demand_compatibility
        )

        return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1

    def visualize_results(self):
        """Create sophisticated visualizations of pooling results"""
        if not self.pooling_results:
            logging.warning("No pooling results to visualize")
            return

        # Create visualizations directory
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        # 1. Create heatmap of successful poolings
        self._create_pooling_heatmap(viz_dir)

        # 2. Create temporal analysis plots
        self._create_temporal_analysis(viz_dir)

        # 3. Create statistics plots
        self._create_statistics_plots(viz_dir)

    def _create_pooling_heatmap(self, viz_dir: Path):
        """Create heatmap of pooling locations"""
        m = folium.Map(location=[31.9522, 35.2332], zoom_start=12)

        # Add successful pooling points
        heat_data = []
        successful_pickups = []
        successful_dropoffs = []

        for result in self.pooling_results:
            ride1 = self._get_request_by_id(result.ride1_id)
            ride2 = self._get_request_by_id(result.ride2_id)

            if ride1 and ride2:
                # Add to heatmap data
                heat_data.extend([
                    [ride1.pickup_lat, ride1.pickup_lng],
                    [ride2.pickup_lat, ride2.pickup_lng],
                    [ride1.dropoff_lat, ride1.dropoff_lng],
                    [ride2.dropoff_lat, ride2.dropoff_lng]
                ])

                # Store for marker clusters
                successful_pickups.extend([
                    (ride1.pickup_lat, ride1.pickup_lng),
                    (ride2.pickup_lat, ride2.pickup_lng)
                ])
                successful_dropoffs.extend([
                    (ride1.dropoff_lat, ride1.dropoff_lng),
                    (ride2.dropoff_lat, ride2.dropoff_lng)
                ])

        # Add heatmap layer
        plugins.HeatMap(heat_data).add_to(m)

        # Add marker clusters
        pickup_cluster = plugins.MarkerCluster(name='Pickups')
        dropoff_cluster = plugins.MarkerCluster(name='Dropoffs')

        for lat, lng in successful_pickups:
            folium.Marker(
                [lat, lng],
                icon=folium.Icon(color='green', icon='info-sign'),
                popup='Pickup'
            ).add_to(pickup_cluster)

        for lat, lng in successful_dropoffs:
            folium.Marker(
                [lat, lng],
                icon=folium.Icon(color='red', icon='info-sign'),
                popup='Dropoff'
            ).add_to(dropoff_cluster)

        pickup_cluster.add_to(m)
        dropoff_cluster.add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        m.save(viz_dir / 'pooling_heatmap.html')

    def _create_temporal_analysis(self, viz_dir: Path):
        """Create temporal analysis plots"""
        plt.figure(figsize=(15, 10))

        # Create multiple subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # 1. Hourly distribution of successful pools
        hours = [self._get_request_by_id(r.ride1_id).pickup_time.hour
                 for r in self.pooling_results]

        sns.histplot(data=hours, bins=24, ax=ax1)
        ax1.set_title("Distribution of Successful Pools by Hour")
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Number of Pools")

        # 2. Success rate by hour
        hourly_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})

        # Count total rides per hour
        for request in self.all_requests:
            hour = request.pickup_time.hour
            hourly_stats[hour]['attempts'] += 1

        # Count successful pools
        for result in self.pooling_results:
            hour = self._get_request_by_id(result.ride1_id).pickup_time.hour
            hourly_stats[hour]['successes'] += 1  # Each pool is a pair of rides

        # Calculate success rates
        hours = sorted(hourly_stats.keys())
        success_rates = [
            (hourly_stats[hour]['successes'] * 2) / hourly_stats[hour]['attempts']
            if hourly_stats[hour]['attempts'] > 0 else 0
            for hour in hours
        ]

        ax2.bar(hours, success_rates)
        ax2.set_title("Pooling Success Rate by Hour")
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Success Rate")
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(viz_dir / 'temporal_analysis.png')
        plt.close()

    def _create_statistics_plots(self, viz_dir: Path):
        """Create detailed statistics plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # 1. Detour ratio distribution
        detour_ratios = [r.detour_ratio for r in self.pooling_results]
        sns.histplot(detour_ratios, ax=axes[0, 0])
        axes[0, 0].set_title("Distribution of Detour Ratios")
        axes[0, 0].set_xlabel("Detour Ratio")

        # 2. Time difference distribution
        time_diffs = [r.time_difference for r in self.pooling_results]
        sns.histplot(time_diffs, ax=axes[0, 1])
        axes[0, 1].set_title("Distribution of Time Differences (minutes)")
        axes[0, 1].set_xlabel("Time Difference")

        # 3. Pickup distance distribution
        pickup_dists = [r.pickup_distance for r in self.pooling_results]
        sns.histplot(pickup_dists, ax=axes[1, 0])
        axes[1, 0].set_title("Distribution of Pickup Distances (meters)")
        axes[1, 0].set_xlabel("Pickup Distance")

        # 4. Matching score distribution
        matching_scores = [r.matching_score for r in self.pooling_results]
        sns.histplot(matching_scores, ax=axes[1, 1])
        axes[1, 1].set_title("Distribution of Matching Scores")
        axes[1, 1].set_xlabel("Matching Score")

        plt.tight_layout()
        plt.savefig(viz_dir / 'pooling_statistics.png')
        plt.close()

    def _get_request_by_id(self, trip_id: str) -> Optional[RideRequest]:
        """Get request object by trip ID"""
        return next((r for r in self.all_requests if r.trip_id == trip_id), None)

def load_and_process_data(file_path: str) -> List[RideRequest]:
    """Load and process ride data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        requests = []
        logging.info(f"Processing {len(data)} records...")

        for ride in data:
            # Only process segment 3 rides
            if ride.get('segment_id') != 3:
                continue

            try:
                # Convert time_id to datetime
                # time_id is in minutes from start of day (0-1439)
                minutes = int(float(ride['time_id']))
                hours = minutes // 60
                remaining_minutes = minutes % 60

                pickup_time = datetime(
                    year=2024,  # Using a default year
                    month=1,    # Using January as default
                    day=ride['date_id'],
                    hour=hours,
                    minute=remaining_minutes
                )

                request = RideRequest(
                    trip_id=ride['trip_id'],
                    pickup_lat=float(ride['pickup_lat']),
                    pickup_lng=float(ride['pickup_lng']),
                    dropoff_lat=float(ride['dropoff_lat']),
                    dropoff_lng=float(ride['dropoff_lng']),
                    pickup_time=pickup_time,
                    trip_distance=float(ride['trip_distance']) * 1000,  # Convert km to meters
                    week_id=int(ride['week_id']),
                    date_id=int(ride['date_id']),
                    time_id=float(ride['time_id']),
                    straight_line_distance=float(ride['straight_line_distance']) * 1000  # Convert km to meters
                )
                requests.append(request)

            except (ValueError, KeyError) as e:
                logging.warning(f"Error processing ride {ride.get('trip_id', 'unknown')}: {e}")
                continue

        # Sort requests by pickup time
        requests.sort(key=lambda x: x.pickup_time)

        logging.info(f"Successfully loaded {len(requests)} valid requests")
        return requests

    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in file: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def find_pooling_opportunities(requests: List[RideRequest], pooling_system: PoolingSystem) -> Set[str]:
    """Find pooling opportunities using spatial indexing"""
    active_window = []
    pooled_trips = set()
    spatial_index = defaultdict(list)

    def get_grid_cell(lat: float, lng: float, cell_size: float = 0.002) -> Tuple[int, int]:
        return (int(lat / cell_size), int(lng / cell_size))

    for request in requests:
        if request.trip_id in pooled_trips:
            continue

        current_time = request.pickup_time

        # Clean up expired requests and spatial index
        active_window = [
            r for r in active_window
            if (current_time - r.pickup_time).total_seconds() <= pooling_system.max_time_window * 60
            and r.trip_id not in pooled_trips
        ]

        # Update spatial index
        spatial_index.clear()
        for r in active_window:
            cell = get_grid_cell(r.pickup_lat, r.pickup_lng)
            spatial_index[cell].append(r)

        # Find nearby requests
        request_cell = get_grid_cell(request.pickup_lat, request.pickup_lng)
        nearby_requests = []

        # Check only the current cell and immediate neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (request_cell[0] + dx, request_cell[1] + dy)
                nearby_requests.extend(spatial_index.get(cell, []))

        # Filter nearby requests based on actual pickup distance
        filtered_nearby_requests = []
        for active_request in nearby_requests:
            pickup_distance = haversine_distance(
                request.pickup_lat, request.pickup_lng,
                active_request.pickup_lat, active_request.pickup_lng
            )
            if pickup_distance <= pooling_system.max_pickup_distance:
                filtered_nearby_requests.append(active_request)

        # Sort by priority score
        filtered_nearby_requests.sort(key=lambda r: r.priority_score, reverse=True)

        # Try pooling
        pooled = False
        for active_request in filtered_nearby_requests:
            if active_request.trip_id in pooled_trips:
                continue
            pool_result = pooling_system.try_pool_rides(active_request, request)
            if pool_result:
                pooled_trips.add(active_request.trip_id)
                pooled_trips.add(request.trip_id)
                pooled = True
                break

        if not pooled:
            active_window.append(request)
            cell = get_grid_cell(request.pickup_lat, request.pickup_lng)
            spatial_index[cell].append(request)

    return pooled_trips

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Ride pooling system')
    parser.add_argument('--data', type=str, required=True, help='Path to input data JSON file')
    parser.add_argument('--output', type=str, default='pooling_results', help='Output directory')
    parser.add_argument('--max-detour', type=float, default=1.2, help='Maximum detour ratio')
    parser.add_argument('--max-pickup-distance', type=float, default=500, help='Maximum pickup distance (meters)')
    parser.add_argument('--max-wait-time', type=int, default=300, help='Maximum wait time (seconds)')
    parser.add_argument('--max-time-window', type=int, default=5, help='Maximum time window (minutes)')
    parser.add_argument('--min-savings', type=float, default=1000, help='Minimum savings threshold (meters)')
    args = parser.parse_args()

    # Load and process data
    logging.info("Loading data...")
    requests = load_and_process_data(args.data)
    logging.info(f"Loaded {len(requests)} valid requests")

    # Initialize pooling system
    pooling_system = PoolingSystem(
        max_detour_ratio=args.max_detour,
        max_pickup_distance=args.max_pickup_distance,
        max_wait_time=args.max_wait_time,
        max_time_window=args.max_time_window,
        min_savings_threshold=args.min_savings,
        output_dir=args.output
    )

    # Store all requests for reference
    pooling_system.all_requests = requests

    # Analyze demand patterns
    logging.info("Analyzing demand patterns...")
    pooling_system.demand_analyzer.analyze_demand(requests)

    # Calculate priority scores
    logging.info("Calculating request priorities...")
    for request in requests:
        request.demand_score = pooling_system.demand_analyzer.get_demand_score(
            request.pickup_lat, request.pickup_lng, request.pickup_time
        )
        request.priority_score = pooling_system.calculate_priority_score(request)

    # Find pooling opportunities
    logging.info("Finding pooling opportunities...")
    pooled_trips = find_pooling_opportunities(requests, pooling_system)

    # Generate visualizations and reports
    logging.info("Generating analysis and visualizations...")
    pooling_system.visualize_results()

    # Calculate pooling rate
    total_trips = len(requests)
    total_pooling_events = len(pooling_system.pooling_results)
    total_pooled_trips = total_pooling_events * 2  # Each pooling event involves 2 trips
    pooling_rate = total_pooled_trips / total_trips if total_trips > 0 else 0

    # Log results
    logging.info("\nPooling Analysis Complete:")
    logging.info(f"Total requests processed: {total_trips}")
    logging.info(f"Successfully pooled trips: {total_pooled_trips}")
    logging.info(f"Number of pooling events: {total_pooling_events}")
    logging.info(f"Pooling rate: {pooling_rate:.2%}")
    logging.info(f"Results saved in: {pooling_system.output_dir}")

if __name__ == "__main__":
    main()
