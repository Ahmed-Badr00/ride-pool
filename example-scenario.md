# Example Pooling Scenario

## Sample Rides

### Ride 1
```python
ride1 = RideRequest(
    trip_id="trip_001",
    pickup_lat=31.9522,
    pickup_lng=35.2332,
    dropoff_lat=31.9822,
    dropoff_lng=35.2532,
    pickup_time=datetime(2024, 1, 24, 8, 30),  # 8:30 AM
    trip_distance=5000,  # 5km
    straight_line_distance=4200  # 4.2km
)
```

### Ride 2
```python
ride2 = RideRequest(
    trip_id="trip_002",
    pickup_lat=31.9535,
    pickup_lng=35.2345,
    dropoff_lat=31.9835,
    dropoff_lng=35.2545,
    pickup_time=datetime(2024, 1, 24, 8, 35),  # 8:35 AM
    trip_distance=5200,  # 5.2km
    straight_line_distance=4400  # 4.4km
)
```

## Matching Analysis

1. Basic Checks:
```python
pickup_distance = 150  # meters between pickups
time_difference = 5    # minutes between pickups
direction_diff = 15    # degrees between routes
```

2. Route Options:
```
Option 1: P1 -> P2 -> D1 -> D2
Total Distance: 11.2km

Option 2: P1 -> P2 -> D2 -> D1
Total Distance: 10.8km
```

3. Scoring:
```python
distance_efficiency = 0.89    # (10.8 - 10.2)/10.2
time_compatibility = 0.92     # 1 - (5/60)
pickup_convenience = 0.93     # 1 - (150/2000)
demand_score = 0.85          # Based on historical data
direction_compatibility = 0.95  # Based on 15-degree difference

final_score = 0.91  # Weighted combination
```

## Result
This would be a successful match because:
- Pickup points are close (150m < 2000m limit)
- Times are compatible (5min < 15min limit)
- Directions are similar (15° < 60° limit)
- Total detour is acceptable (5.8% increase)
- High matching score (0.91 > 0.7 threshold)

The system would select Option 2 (P1->P2->D2->D1) as it provides the shorter total distance.