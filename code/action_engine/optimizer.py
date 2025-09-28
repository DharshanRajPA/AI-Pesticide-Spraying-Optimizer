#!/usr/bin/env python3
"""
Dose optimization engine for AgriSprayAI.
Implements convex optimization for minimal guaranteed pesticide doses
with MILP fallback for discrete actuator constraints.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cvxpy as cp
from ortools.linear_solver import pywraplp
import yaml
from dataclasses import dataclass
from enum import Enum
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolverStatus(Enum):
    """Solver status enumeration."""
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    SOLVER_ERROR = "solver_error"
    TIMEOUT = "timeout"

@dataclass
class OptimizationResult:
    """Result of dose optimization."""
    doses: List[float]
    status: SolverStatus
    objective_value: float
    solve_time: float
    diagnostics: Dict[str, Any]
    warnings: List[str]

@dataclass
class PlantInstance:
    """Represents a plant instance with detection and severity information."""
    id: int
    bbox: List[float]  # [x, y, width, height]
    area: float  # m²
    severity: int  # 0-3
    confidence: float  # 0-1
    category_id: int
    location: Tuple[float, float]  # (x, y) coordinates

class DoseCureModel:
    """Dose-cure probability model: P_cure(d, s) = sigmoid(a(s) * d + b(s))."""
    
    def __init__(self, config_path: str = "configs/optimizer.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.parameters = self.config["dose_cure_model"]["parameters"]
        self.fallback_params = self.config["dose_cure_model"]["fallback"]
    
    def get_cure_probability(self, dose: float, severity: int) -> float:
        """Calculate cure probability for given dose and severity."""
        if severity in self.parameters:
            a, b = self.parameters[severity]["a"], self.parameters[severity]["b"]
        else:
            a, b = self.fallback_params["a"], self.fallback_params["b"]
        
        # P_cure(d, s) = sigmoid(a(s) * d + b(s))
        logit = a * dose + b
        return 1.0 / (1.0 + np.exp(-logit))
    
    def get_required_dose(self, target_prob: float, severity: int) -> float:
        """Calculate required dose for target cure probability."""
        if severity in self.parameters:
            a, b = self.parameters[severity]["a"], self.parameters[severity]["b"]
        else:
            a, b = self.fallback_params["a"], self.fallback_params["b"]
        
        # Solve: target_prob = sigmoid(a * d + b)
        # d = (logit(target_prob) - b) / a
        if target_prob <= 0:
            return 0.0
        elif target_prob >= 1:
            return float('inf')
        
        logit_target = np.log(target_prob / (1 - target_prob))
        required_dose = (logit_target - b) / a
        
        return max(0.0, required_dose)

class DoseOptimizer:
    """Main dose optimization engine."""
    
    def __init__(self, config_path: str = "configs/optimizer.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.optimization_config = self.config["optimization"]
        self.constraints_config = self.config["constraints"]
        self.safety_config = self.config["safety"]
        
        # Initialize dose-cure model
        self.dose_cure_model = DoseCureModel(config_path)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for optimization results."""
        log_dir = Path(self.config["monitoring"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def solve_doses_cvxpy(self, 
                         plants: List[PlantInstance],
                         target_cure_prob: float = None,
                         max_dose_per_plant: float = None,
                         max_dose_per_area: float = None) -> OptimizationResult:
        """Solve dose optimization using CVXPY convex optimization."""
        
        if target_cure_prob is None:
            target_cure_prob = self.constraints_config["target_cure_prob"]
        if max_dose_per_plant is None:
            max_dose_per_plant = self.constraints_config["per_plant"]["max_dose"]
        if max_dose_per_area is None:
            max_dose_per_area = self.constraints_config["area"]["max_dose_per_m2"]
        
        start_time = time.time()
        warnings = []
        
        try:
            n_plants = len(plants)
            if n_plants == 0:
                return OptimizationResult(
                    doses=[], status=SolverStatus.OPTIMAL, objective_value=0.0,
                    solve_time=0.0, diagnostics={}, warnings=["No plants to optimize"]
                )
            
            # Decision variables: doses for each plant
            doses = cp.Variable(n_plants, nonneg=True)
            
            # Objective: minimize total dose
            objective = cp.Minimize(cp.sum(doses))
            
            # Constraints
            constraints = []
            
            # 1. Cure probability constraints: P_cure(d_i, s_i) >= target_cure_prob
            for i, plant in enumerate(plants):
                if plant.severity in self.dose_cure_model.parameters:
                    a = self.dose_cure_model.parameters[plant.severity]["a"]
                    b = self.dose_cure_model.parameters[plant.severity]["b"]
                else:
                    a = self.dose_cure_model.fallback_params["a"]
                    b = self.dose_cure_model.fallback_params["b"]
                
                # P_cure(d_i, s_i) >= target_cure_prob
                # sigmoid(a * d_i + b) >= target_cure_prob
                # a * d_i + b >= logit(target_cure_prob)
                logit_target = np.log(target_cure_prob / (1 - target_cure_prob))
                constraints.append(a * doses[i] + b >= logit_target)
            
            # 2. Per-plant dose limits
            constraints.append(doses <= max_dose_per_plant)
            
            # 3. Area-wide dose limits
            total_area = sum(plant.area for plant in plants)
            if total_area > 0:
                constraints.append(cp.sum(doses) <= max_dose_per_area * total_area)
            
            # 4. Regulatory limits
            max_total_dose = self.constraints_config["regulatory"]["max_total_dose"]
            constraints.append(cp.sum(doses) <= max_total_dose)
            
            # Solve the problem
            problem = cp.Problem(objective, constraints)
            
            # Choose solver
            solver_name = self.optimization_config["cvxpy"]["solver"]
            solver_kwargs = {
                "max_iter": self.optimization_config["cvxpy"]["max_iter"],
                "eps_abs": self.optimization_config["cvxpy"]["eps_abs"],
                "eps_rel": self.optimization_config["cvxpy"]["eps_rel"],
                "verbose": self.optimization_config["cvxpy"]["verbose"]
            }
            
            if solver_name == "OSQP":
                problem.solve(solver=cp.OSQP, **solver_kwargs)
            elif solver_name == "ECOS":
                problem.solve(solver=cp.ECOS, **solver_kwargs)
            elif solver_name == "SCS":
                problem.solve(solver=cp.SCS, **solver_kwargs)
            else:
                problem.solve()
            
            solve_time = time.time() - start_time
            
            # Check solution status
            if problem.status == cp.OPTIMAL:
                status = SolverStatus.OPTIMAL
                doses_solution = doses.value.tolist()
                objective_value = problem.value
                
                # Check for warnings
                if any(d > max_dose_per_plant * 0.8 for d in doses_solution):
                    warnings.append("High doses detected (>80% of max per plant)")
                
                if sum(doses_solution) > max_total_dose * 0.8:
                    warnings.append("High total dose (>80% of regulatory limit)")
                
            elif problem.status == cp.INFEASIBLE:
                status = SolverStatus.INFEASIBLE
                doses_solution = []
                objective_value = float('inf')
                warnings.append("Problem is infeasible - constraints cannot be satisfied")
                
            elif problem.status == cp.UNBOUNDED:
                status = SolverStatus.UNBOUNDED
                doses_solution = []
                objective_value = float('-inf')
                warnings.append("Problem is unbounded - no optimal solution")
                
            else:
                status = SolverStatus.SOLVER_ERROR
                doses_solution = []
                objective_value = float('nan')
                warnings.append(f"Solver error: {problem.status}")
            
            # Diagnostics
            diagnostics = {
                "solver": solver_name,
                "problem_status": problem.status,
                "solve_time": solve_time,
                "n_plants": n_plants,
                "n_constraints": len(constraints),
                "target_cure_prob": target_cure_prob,
                "max_dose_per_plant": max_dose_per_plant,
                "max_dose_per_area": max_dose_per_area,
                "total_area": total_area
            }
            
            return OptimizationResult(
                doses=doses_solution,
                status=status,
                objective_value=objective_value,
                solve_time=solve_time,
                diagnostics=diagnostics,
                warnings=warnings
            )
            
        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"CVXPY optimization failed: {e}")
            
            return OptimizationResult(
                doses=[],
                status=SolverStatus.SOLVER_ERROR,
                objective_value=float('nan'),
                solve_time=solve_time,
                diagnostics={"error": str(e)},
                warnings=[f"Optimization error: {e}"]
            )
    
    def solve_doses_milp(self, 
                        plants: List[PlantInstance],
                        waypoints: List[Tuple[float, float]],
                        target_cure_prob: float = None) -> OptimizationResult:
        """Solve dose optimization using MILP for discrete actuator constraints."""
        
        if target_cure_prob is None:
            target_cure_prob = self.constraints_config["target_cure_prob"]
        
        start_time = time.time()
        warnings = []
        
        try:
            n_plants = len(plants)
            n_waypoints = len(waypoints)
            
            if n_plants == 0 or n_waypoints == 0:
                return OptimizationResult(
                    doses=[], status=SolverStatus.OPTIMAL, objective_value=0.0,
                    solve_time=0.0, diagnostics={}, warnings=["No plants or waypoints"]
                )
            
            # Create solver
            solver = pywraplp.Solver.CreateSolver(self.optimization_config["milp"]["solver"])
            if not solver:
                raise ValueError(f"Solver {self.optimization_config['milp']['solver']} not available")
            
            # Set time limit
            solver.SetTimeLimit(self.optimization_config["milp"]["time_limit"] * 1000)  # Convert to milliseconds
            
            # Decision variables
            # x[k] = 1 if waypoint k is activated, 0 otherwise
            x = {}
            for k in range(n_waypoints):
                x[k] = solver.IntVar(0, 1, f'x_{k}')
            
            # d[i] = dose for plant i (continuous)
            d = {}
            for i in range(n_plants):
                d[i] = solver.NumVar(0, self.constraints_config["per_plant"]["max_dose"], f'd_{i}')
            
            # Objective: minimize total dose
            objective = solver.Objective()
            for i in range(n_plants):
                objective.SetCoefficient(d[i], 1)
            objective.SetMinimization()
            
            # Constraints
            # 1. Cure probability constraints
            for i, plant in enumerate(plants):
                if plant.severity in self.dose_cure_model.parameters:
                    a = self.dose_cure_model.parameters[plant.severity]["a"]
                    b = self.dose_cure_model.parameters[plant.severity]["b"]
                else:
                    a = self.dose_cure_model.fallback_params["a"]
                    b = self.dose_cure_model.fallback_params["b"]
                
                logit_target = np.log(target_cure_prob / (1 - target_cure_prob))
                constraint = solver.Constraint(logit_target - b, solver.infinity())
                constraint.SetCoefficient(d[i], a)
            
            # 2. Actuator constraints: dose depends on activated waypoints
            # This is a simplified model - in practice, you'd need more complex
            # constraints based on the specific actuator geometry
            for i in range(n_plants):
                # Each plant's dose is limited by nearby waypoints
                # This is a placeholder - implement based on your specific setup
                constraint = solver.Constraint(0, solver.infinity())
                constraint.SetCoefficient(d[i], 1)
                for k in range(n_waypoints):
                    constraint.SetCoefficient(x[k], -0.1)  # Placeholder coefficient
            
            # 3. Regulatory limits
            total_dose_constraint = solver.Constraint(0, self.constraints_config["regulatory"]["max_total_dose"])
            for i in range(n_plants):
                total_dose_constraint.SetCoefficient(d[i], 1)
            
            # Solve
            status = solver.Solve()
            solve_time = time.time() - start_time
            
            # Process results
            if status == pywraplp.Solver.OPTIMAL:
                solver_status = SolverStatus.OPTIMAL
                doses_solution = [d[i].solution_value() for i in range(n_plants)]
                objective_value = solver.Objective().Value()
                
            elif status == pywraplp.Solver.FEASIBLE:
                solver_status = SolverStatus.OPTIMAL
                doses_solution = [d[i].solution_value() for i in range(n_plants)]
                objective_value = solver.Objective().Value()
                warnings.append("Solution is feasible but may not be optimal")
                
            elif status == pywraplp.Solver.INFEASIBLE:
                solver_status = SolverStatus.INFEASIBLE
                doses_solution = []
                objective_value = float('inf')
                warnings.append("MILP problem is infeasible")
                
            elif status == pywraplp.Solver.UNBOUNDED:
                solver_status = SolverStatus.UNBOUNDED
                doses_solution = []
                objective_value = float('-inf')
                warnings.append("MILP problem is unbounded")
                
            else:
                solver_status = SolverStatus.SOLVER_ERROR
                doses_solution = []
                objective_value = float('nan')
                warnings.append(f"MILP solver error: {status}")
            
            # Diagnostics
            diagnostics = {
                "solver": self.optimization_config["milp"]["solver"],
                "status": status,
                "solve_time": solve_time,
                "n_plants": n_plants,
                "n_waypoints": n_waypoints,
                "target_cure_prob": target_cure_prob
            }
            
            return OptimizationResult(
                doses=doses_solution,
                status=solver_status,
                objective_value=objective_value,
                solve_time=solve_time,
                diagnostics=diagnostics,
                warnings=warnings
            )
            
        except Exception as e:
            solve_time = time.time() - start_time
            logger.error(f"MILP optimization failed: {e}")
            
            return OptimizationResult(
                doses=[],
                status=SolverStatus.SOLVER_ERROR,
                objective_value=float('nan'),
                solve_time=solve_time,
                diagnostics={"error": str(e)},
                warnings=[f"MILP optimization error: {e}"]
            )
    
    def solve_doses(self, 
                   plants: List[PlantInstance],
                   waypoints: List[Tuple[float, float]] = None,
                   target_cure_prob: float = None,
                   max_dose_per_plant: float = None,
                   max_dose_per_area: float = None,
                   use_milp: bool = False) -> OptimizationResult:
        """Main optimization function with automatic solver selection."""
        
        # Check if operator approval is required
        low_confidence_plants = [p for p in plants if p.confidence < self.safety_config["confidence_threshold"]]
        if low_confidence_plants:
            logger.warning(f"Low confidence plants detected: {len(low_confidence_plants)}")
        
        # Choose solver
        if use_milp or waypoints is not None:
            logger.info("Using MILP solver for discrete constraints")
            return self.solve_doses_milp(plants, waypoints or [], target_cure_prob)
        else:
            logger.info("Using CVXPY convex solver")
            return self.solve_doses_cvxpy(plants, target_cure_prob, max_dose_per_plant, max_dose_per_area)
    
    def log_optimization_result(self, result: OptimizationResult, plants: List[PlantInstance]):
        """Log optimization result for audit trail."""
        log_entry = {
            "timestamp": time.time(),
            "n_plants": len(plants),
            "status": result.status.value,
            "objective_value": result.objective_value,
            "solve_time": result.solve_time,
            "doses": result.doses,
            "diagnostics": result.diagnostics,
            "warnings": result.warnings,
            "plant_info": [
                {
                    "id": p.id,
                    "severity": p.severity,
                    "confidence": p.confidence,
                    "area": p.area
                } for p in plants
            ]
        }
        
        # Save to log file
        log_dir = Path(self.config["monitoring"]["log_dir"])
        log_file = log_dir / f"optimization_{int(time.time())}.json"
        
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        logger.info(f"Optimization result logged to {log_file}")

def main():
    """Example usage of the dose optimizer."""
    # Create sample plants
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
    
    # Initialize optimizer
    optimizer = DoseOptimizer()
    
    # Solve optimization
    result = optimizer.solve_doses(plants)
    
    # Print results
    print(f"Optimization Status: {result.status.value}")
    print(f"Objective Value: {result.objective_value:.2f}")
    print(f"Solve Time: {result.solve_time:.3f}s")
    print(f"Doses: {result.doses}")
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # Log result
    optimizer.log_optimization_result(result, plants)

if __name__ == "__main__":
    main()
