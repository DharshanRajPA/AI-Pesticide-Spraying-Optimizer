#!/usr/bin/env python3
"""
Flight planning module for AgriSprayAI.
Generates optimal flight paths and MAVLink missions for UAV pesticide application.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import yaml
from dataclasses import dataclass
from enum import Enum
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlightMode(Enum):
    """Flight mode enumeration."""
    MISSION = "mission"
    GUIDED = "guided"
    AUTO = "auto"
    LOITER = "loiter"

@dataclass
class Waypoint:
    """Represents a waypoint in the flight plan."""
    id: int
    latitude: float
    longitude: float
    altitude: float
    command: int  # MAVLink command
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    frame: int = 3  # MAV_FRAME_GLOBAL_RELATIVE_ALT
    current: int = 0
    autocontinue: int = 1

@dataclass
class SprayPoint:
    """Represents a spray point with dose information."""
    waypoint_id: int
    plant_id: int
    dose: float  # ml
    duration: float  # seconds
    start_time: float  # seconds from mission start

class FlightPlanner:
    """Main flight planning class for UAV pesticide application."""
    
    def __init__(self, config_path: str = "configs/optimizer.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.actuator_config = self.config["actuator"]
        self.flight_config = self.actuator_config["flight"]
        
        # MAVLink command constants
        self.MAV_CMD_NAV_WAYPOINT = 16
        self.MAV_CMD_NAV_TAKEOFF = 22
        self.MAV_CMD_NAV_LAND = 21
        self.MAV_CMD_DO_SET_SERVO = 183
        self.MAV_CMD_DO_CHANGE_SPEED = 178
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for flight planning."""
        log_dir = Path("logs/flight_planning")
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def cluster_plants(self, plants: List[Any], eps: float = 10.0, min_samples: int = 2) -> List[List[Any]]:
        """Cluster plants using DBSCAN for efficient flight planning."""
        
        if len(plants) < 2:
            return [plants] if plants else []
        
        # Extract plant locations
        locations = np.array([plant.location for plant in plants])
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(locations)
        
        # Group plants by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(plants[i])
        
        # Convert to list of lists
        plant_clusters = list(clusters.values())
        
        logger.info(f"Clustered {len(plants)} plants into {len(plant_clusters)} clusters")
        return plant_clusters
    
    def solve_tsp(self, points: List[Tuple[float, float]]) -> List[int]:
        """Solve Traveling Salesman Problem for optimal route through points."""
        
        if len(points) <= 2:
            return list(range(len(points)))
        
        n = len(points)
        
        # Calculate distance matrix
        distances = squareform(pdist(points))
        
        # Simple greedy TSP solution (for production, use more sophisticated algorithms)
        def tsp_objective(route):
            total_distance = 0
            for i in range(len(route) - 1):
                total_distance += distances[route[i], route[i + 1]]
            return total_distance
        
        # Try different starting points and pick the best
        best_route = None
        best_distance = float('inf')
        
        for start in range(n):
            # Greedy nearest neighbor
            route = [start]
            unvisited = set(range(n)) - {start}
            
            while unvisited:
                current = route[-1]
                nearest = min(unvisited, key=lambda x: distances[current, x])
                route.append(nearest)
                unvisited.remove(nearest)
            
            distance = tsp_objective(route)
            if distance < best_distance:
                best_distance = distance
                best_route = route
        
        return best_route
    
    def generate_waypoints(self, plants: List[Any], field_location: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate waypoints for the flight plan."""
        
        # Cluster plants
        plant_clusters = self.cluster_plants(plants)
        
        waypoints = []
        waypoint_id = 0
        
        # Add takeoff waypoint
        takeoff_waypoint = {
            "id": waypoint_id,
            "latitude": field_location["latitude"],
            "longitude": field_location["longitude"],
            "altitude": self.flight_config["altitude"],
            "command": self.MAV_CMD_NAV_TAKEOFF,
            "param1": 0.0,
            "param2": 0.0,
            "param3": 0.0,
            "param4": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": self.flight_config["altitude"],
            "frame": 3,
            "current": 0,
            "autocontinue": 1
        }
        waypoints.append(takeoff_waypoint)
        waypoint_id += 1
        
        # Process each cluster
        for cluster in plant_clusters:
            if not cluster:
                continue
            
            # Extract cluster locations
            cluster_locations = [plant.location for plant in cluster]
            
            # Solve TSP for this cluster
            tsp_route = self.solve_tsp(cluster_locations)
            
            # Generate waypoints for this cluster
            for i, plant_idx in enumerate(tsp_route):
                plant = cluster[plant_idx]
                
                # Convert local coordinates to GPS (simplified)
                # In practice, use proper coordinate transformation
                lat_offset = plant.location[0] * 0.00001  # Rough conversion
                lon_offset = plant.location[1] * 0.00001
                
                waypoint = {
                    "id": waypoint_id,
                    "latitude": field_location["latitude"] + lat_offset,
                    "longitude": field_location["longitude"] + lon_offset,
                    "altitude": self.flight_config["altitude"],
                    "command": self.MAV_CMD_NAV_WAYPOINT,
                    "param1": 0.0,  # Hold time
                    "param2": 0.0,  # Acceptance radius
                    "param3": 0.0,  # Pass radius
                    "param4": 0.0,  # Yaw
                    "x": plant.location[0],
                    "y": plant.location[1],
                    "z": self.flight_config["altitude"],
                    "frame": 3,
                    "current": 0,
                    "autocontinue": 1,
                    "plant_id": plant.id,
                    "dose": 0.0  # Will be filled by optimization
                }
                waypoints.append(waypoint)
                waypoint_id += 1
        
        # Add landing waypoint
        landing_waypoint = {
            "id": waypoint_id,
            "latitude": field_location["latitude"],
            "longitude": field_location["longitude"],
            "altitude": 0.0,
            "command": self.MAV_CMD_NAV_LAND,
            "param1": 0.0,
            "param2": 0.0,
            "param3": 0.0,
            "param4": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "frame": 3,
            "current": 0,
            "autocontinue": 1
        }
        waypoints.append(landing_waypoint)
        
        logger.info(f"Generated {len(waypoints)} waypoints for {len(plants)} plants")
        return waypoints
    
    def calculate_spray_duration(self, dose: float) -> float:
        """Calculate spray duration for given dose."""
        flow_rate = self.actuator_config["nozzle"]["flow_rate"]
        efficiency = self.actuator_config["nozzle"]["efficiency"]
        
        # duration = dose / (flow_rate * efficiency)
        duration = dose / (flow_rate * efficiency)
        
        # Apply constraints
        min_duration = self.actuator_config["nozzle"]["min_on_time"]
        max_duration = self.actuator_config["nozzle"]["max_on_time"]
        
        return max(min_duration, min(duration, max_duration))
    
    def generate_mavlink_mission(self, waypoints: List[Dict[str, Any]], doses: List[float]) -> Dict[str, Any]:
        """Generate MAVLink mission with spray commands."""
        
        # Create mission items
        mission_items = []
        
        for i, waypoint in enumerate(waypoints):
            mission_item = {
                "seq": waypoint["id"],
                "frame": waypoint["frame"],
                "command": waypoint["command"],
                "current": waypoint["current"],
                "autocontinue": waypoint["autocontinue"],
                "param1": waypoint["param1"],
                "param2": waypoint["param2"],
                "param3": waypoint["param3"],
                "param4": waypoint["param4"],
                "x": waypoint["x"],
                "y": waypoint["y"],
                "z": waypoint["z"]
            }
            mission_items.append(mission_item)
            
            # Add spray command if this waypoint has a plant
            if "plant_id" in waypoint and waypoint["plant_id"] is not None:
                plant_id = waypoint["plant_id"]
                if plant_id <= len(doses):
                    dose = doses[plant_id - 1]
                    spray_duration = self.calculate_spray_duration(dose)
                    
                    # Add servo command to activate spray
                    spray_command = {
                        "seq": waypoint["id"] + 1000,  # Offset to avoid conflicts
                        "frame": 3,
                        "command": self.MAV_CMD_DO_SET_SERVO,
                        "current": 0,
                        "autocontinue": 1,
                        "param1": 1.0,  # Servo number
                        "param2": 1500.0,  # PWM value for ON
                        "param3": 0.0,
                        "param4": 0.0,
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0
                    }
                    mission_items.append(spray_command)
                    
                    # Add delay command for spray duration
                    delay_command = {
                        "seq": waypoint["id"] + 1001,
                        "frame": 3,
                        "command": 93,  # MAV_CMD_CONDITION_DELAY
                        "current": 0,
                        "autocontinue": 1,
                        "param1": spray_duration,
                        "param2": 0.0,
                        "param3": 0.0,
                        "param4": 0.0,
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0
                    }
                    mission_items.append(delay_command)
                    
                    # Add servo command to deactivate spray
                    stop_spray_command = {
                        "seq": waypoint["id"] + 1002,
                        "frame": 3,
                        "command": self.MAV_CMD_DO_SET_SERVO,
                        "current": 0,
                        "autocontinue": 1,
                        "param1": 1.0,  # Servo number
                        "param2": 1000.0,  # PWM value for OFF
                        "param3": 0.0,
                        "param4": 0.0,
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0
                    }
                    mission_items.append(stop_spray_command)
        
        # Create mission
        mission = {
            "mission": {
                "plannedHomePosition": {
                    "latitude": waypoints[0]["latitude"],
                    "longitude": waypoints[0]["longitude"],
                    "altitude": 0.0
                },
                "items": mission_items
            },
            "geoFence": {
                "circles": [],
                "polygons": [],
                "version": 2
            },
            "rallyPoints": {
                "points": [],
                "version": 2
            }
        }
        
        logger.info(f"Generated MAVLink mission with {len(mission_items)} items")
        return mission
    
    def estimate_flight_time(self, waypoints: List[Dict[str, Any]]) -> float:
        """Estimate total flight time for the mission."""
        
        if len(waypoints) < 2:
            return 0.0
        
        total_distance = 0.0
        speed = self.flight_config["speed"]
        
        # Calculate total distance
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            
            # Calculate distance between waypoints
            lat1, lon1 = wp1["latitude"], wp1["longitude"]
            lat2, lon2 = wp2["latitude"], wp2["longitude"]
            
            # Haversine formula for distance calculation
            R = 6371000  # Earth's radius in meters
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = (math.sin(dlat/2) * math.sin(dlat/2) + 
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                 math.sin(dlon/2) * math.sin(dlon/2))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = R * c
            
            total_distance += distance
        
        # Calculate flight time
        flight_time = total_distance / speed
        
        # Add time for takeoff, landing, and spray operations
        takeoff_time = 30.0  # seconds
        landing_time = 30.0  # seconds
        spray_time = sum(self.calculate_spray_duration(0.0) for _ in waypoints)  # Placeholder
        
        total_time = flight_time + takeoff_time + landing_time + spray_time
        
        logger.info(f"Estimated flight time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        return total_time
    
    def validate_mission(self, mission: Dict[str, Any]) -> List[str]:
        """Validate the generated mission for safety and feasibility."""
        
        warnings = []
        
        # Check mission items
        mission_items = mission["mission"]["items"]
        if not mission_items:
            warnings.append("Mission has no items")
            return warnings
        
        # Check for takeoff command
        takeoff_commands = [item for item in mission_items if item["command"] == self.MAV_CMD_NAV_TAKEOFF]
        if not takeoff_commands:
            warnings.append("Mission missing takeoff command")
        
        # Check for landing command
        landing_commands = [item for item in mission_items if item["command"] == self.MAV_CMD_NAV_LAND]
        if not landing_commands:
            warnings.append("Mission missing landing command")
        
        # Check altitude consistency
        altitudes = [item["z"] for item in mission_items if item["command"] == self.MAV_CMD_NAV_WAYPOINT]
        if altitudes:
            min_alt = min(altitudes)
            max_alt = max(altitudes)
            if max_alt - min_alt > 10.0:  # 10 meter variation
                warnings.append(f"Large altitude variation: {max_alt - min_alt:.1f}m")
        
        # Check for reasonable mission length
        if len(mission_items) > 1000:
            warnings.append(f"Mission has many items: {len(mission_items)}")
        
        logger.info(f"Mission validation: {len(warnings)} warnings")
        return warnings
    
    def save_mission(self, mission: Dict[str, Any], filename: str):
        """Save mission to file."""
        
        mission_dir = Path("missions")
        mission_dir.mkdir(exist_ok=True)
        
        mission_file = mission_dir / filename
        
        with open(mission_file, 'w') as f:
            json.dump(mission, f, indent=2)
        
        logger.info(f"Saved mission to {mission_file}")
    
    def load_mission(self, filename: str) -> Dict[str, Any]:
        """Load mission from file."""
        
        mission_file = Path("missions") / filename
        
        if not mission_file.exists():
            raise FileNotFoundError(f"Mission file not found: {mission_file}")
        
        with open(mission_file, 'r') as f:
            mission = json.load(f)
        
        logger.info(f"Loaded mission from {mission_file}")
        return mission

def main():
    """Example usage of the flight planner."""
    
    # Create sample plants
    from action_engine.optimizer import PlantInstance
    
    plants = [
        PlantInstance(
            id=1, bbox=[100, 100, 50, 50], area=0.25, severity=2, confidence=0.9,
            category_id=1, location=(100, 100)
        ),
        PlantInstance(
            id=2, bbox=[200, 200, 60, 60], area=0.36, severity=1, confidence=0.8,
            category_id=2, location=(200, 200)
        ),
        PlantInstance(
            id=3, bbox=[300, 300, 40, 40], area=0.16, severity=3, confidence=0.95,
            category_id=1, location=(300, 300)
        )
    ]
    
    # Field location
    field_location = {
        "latitude": 40.7128,
        "longitude": -74.0060
    }
    
    # Initialize flight planner
    planner = FlightPlanner()
    
    # Generate waypoints
    waypoints = planner.generate_waypoints(plants, field_location)
    
    # Sample doses
    doses = [10.0, 5.0, 15.0]
    
    # Generate MAVLink mission
    mission = planner.generate_mavlink_mission(waypoints, doses)
    
    # Estimate flight time
    flight_time = planner.estimate_flight_time(waypoints)
    
    # Validate mission
    warnings = planner.validate_mission(mission)
    
    # Save mission
    planner.save_mission(mission, "sample_mission.json")
    
    print(f"Generated mission with {len(waypoints)} waypoints")
    print(f"Estimated flight time: {flight_time:.1f} seconds")
    print(f"Validation warnings: {warnings}")

if __name__ == "__main__":
    main()
