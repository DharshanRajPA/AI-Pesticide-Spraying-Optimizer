#!/usr/bin/env python3
"""
Unit tests for the dose optimizer module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add the code directory to the path
sys.path.append(str(Path(__file__).parent.parent / "code"))

from action_engine.optimizer import (
    DoseOptimizer, 
    PlantInstance, 
    DoseCureModel, 
    SolverStatus,
    OptimizationResult
)

class TestDoseCureModel:
    """Test the dose-cure probability model."""
    
    def test_get_cure_probability(self):
        """Test cure probability calculation."""
        model = DoseCureModel()
        
        # Test with known parameters
        dose = 10.0
        severity = 1
        
        prob = model.get_cure_probability(dose, severity)
        
        # Should return a probability between 0 and 1
        assert 0 <= prob <= 1
        
        # Higher dose should give higher probability
        prob_higher = model.get_cure_probability(20.0, severity)
        assert prob_higher > prob
    
    def test_get_required_dose(self):
        """Test required dose calculation."""
        model = DoseCureModel()
        
        target_prob = 0.85
        severity = 2
        
        required_dose = model.get_required_dose(target_prob, severity)
        
        # Should return a positive dose
        assert required_dose >= 0
        
        # Verify that this dose gives the target probability
        actual_prob = model.get_cure_probability(required_dose, severity)
        assert abs(actual_prob - target_prob) < 0.01
    
    def test_fallback_parameters(self):
        """Test fallback parameters for unknown severity."""
        model = DoseCureModel()
        
        # Test with unknown severity
        dose = 10.0
        severity = 99  # Unknown severity
        
        prob = model.get_cure_probability(dose, severity)
        
        # Should still return a valid probability
        assert 0 <= prob <= 1

class TestDoseOptimizer:
    """Test the main dose optimizer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = DoseOptimizer()
        
        # Create sample plants
        self.plants = [
            PlantInstance(
                id=1, bbox=[100, 100, 50, 50], area=0.25, severity=1, confidence=0.9,
                category_id=1, location=(100, 100)
            ),
            PlantInstance(
                id=2, bbox=[200, 200, 60, 60], area=0.36, severity=2, confidence=0.8,
                category_id=2, location=(200, 200)
            ),
            PlantInstance(
                id=3, bbox=[300, 300, 40, 40], area=0.16, severity=3, confidence=0.95,
                category_id=1, location=(300, 300)
            )
        ]
    
    def test_solve_doses_cvxpy_optimal(self):
        """Test CVXPY optimization with optimal solution."""
        result = self.optimizer.solve_doses_cvxpy(self.plants)
        
        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert result.status == SolverStatus.OPTIMAL
        assert len(result.doses) == len(self.plants)
        assert all(dose >= 0 for dose in result.doses)
        assert result.solve_time > 0
        assert "solver" in result.diagnostics
    
    def test_solve_doses_cvxpy_empty_plants(self):
        """Test CVXPY optimization with no plants."""
        result = self.optimizer.solve_doses_cvxpy([])
        
        assert result.status == SolverStatus.OPTIMAL
        assert result.doses == []
        assert result.objective_value == 0.0
    
    def test_solve_doses_cvxpy_constraints(self):
        """Test CVXPY optimization with custom constraints."""
        max_dose_per_plant = 20.0
        max_dose_per_area = 50.0
        
        result = self.optimizer.solve_doses_cvxpy(
            self.plants,
            max_dose_per_plant=max_dose_per_plant,
            max_dose_per_area=max_dose_per_area
        )
        
        if result.status == SolverStatus.OPTIMAL:
            # Check constraints
            assert all(dose <= max_dose_per_plant for dose in result.doses)
            
            total_area = sum(plant.area for plant in self.plants)
            total_dose = sum(result.doses)
            assert total_dose <= max_dose_per_area * total_area
    
    def test_solve_doses_milp(self):
        """Test MILP optimization."""
        waypoints = [(100, 100), (200, 200), (300, 300)]
        
        result = self.optimizer.solve_doses_milp(self.plants, waypoints)
        
        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert result.solve_time > 0
        assert "solver" in result.diagnostics
    
    def test_solve_doses_automatic_selection(self):
        """Test automatic solver selection."""
        # Test CVXPY selection
        result = self.optimizer.solve_doses(self.plants, use_milp=False)
        assert isinstance(result, OptimizationResult)
        
        # Test MILP selection
        waypoints = [(100, 100), (200, 200), (300, 300)]
        result = self.optimizer.solve_doses(self.plants, waypoints=waypoints)
        assert isinstance(result, OptimizationResult)
    
    def test_low_confidence_warning(self):
        """Test warning for low confidence plants."""
        # Create plant with low confidence
        low_confidence_plant = PlantInstance(
            id=4, bbox=[400, 400, 50, 50], area=0.25, severity=1, confidence=0.5,
            category_id=1, location=(400, 400)
        )
        
        plants_with_low_confidence = self.plants + [low_confidence_plant]
        
        result = self.optimizer.solve_doses(plants_with_low_confidence)
        
        # Should still work but may have warnings
        assert isinstance(result, OptimizationResult)
    
    def test_log_optimization_result(self):
        """Test logging of optimization results."""
        result = self.optimizer.solve_doses_cvxpy(self.plants)
        
        # Mock the log directory creation
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            with patch('builtins.open', create=True) as mock_open:
                self.optimizer.log_optimization_result(result, self.plants)
                
                # Should create log directory
                mock_mkdir.assert_called()
                
                # Should write log file
                mock_open.assert_called()

class TestPlantInstance:
    """Test the PlantInstance dataclass."""
    
    def test_plant_instance_creation(self):
        """Test PlantInstance creation."""
        plant = PlantInstance(
            id=1,
            bbox=[100, 100, 50, 50],
            area=0.25,
            severity=2,
            confidence=0.9,
            category_id=1,
            location=(100, 100)
        )
        
        assert plant.id == 1
        assert plant.bbox == [100, 100, 50, 50]
        assert plant.area == 0.25
        assert plant.severity == 2
        assert plant.confidence == 0.9
        assert plant.category_id == 1
        assert plant.location == (100, 100)
    
    def test_plant_instance_validation(self):
        """Test PlantInstance validation."""
        # Test with invalid severity
        with pytest.raises(ValueError):
            PlantInstance(
                id=1, bbox=[100, 100, 50, 50], area=0.25, severity=5, confidence=0.9,
                category_id=1, location=(100, 100)
            )
        
        # Test with invalid confidence
        with pytest.raises(ValueError):
            PlantInstance(
                id=1, bbox=[100, 100, 50, 50], area=0.25, severity=2, confidence=1.5,
                category_id=1, location=(100, 100)
            )

class TestSolverStatus:
    """Test the SolverStatus enumeration."""
    
    def test_solver_status_values(self):
        """Test SolverStatus values."""
        assert SolverStatus.OPTIMAL.value == "optimal"
        assert SolverStatus.INFEASIBLE.value == "infeasible"
        assert SolverStatus.UNBOUNDED.value == "unbounded"
        assert SolverStatus.SOLVER_ERROR.value == "solver_error"
        assert SolverStatus.TIMEOUT.value == "timeout"

class TestOptimizationResult:
    """Test the OptimizationResult dataclass."""
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation."""
        result = OptimizationResult(
            doses=[10.0, 20.0, 15.0],
            status=SolverStatus.OPTIMAL,
            objective_value=45.0,
            solve_time=1.5,
            diagnostics={"solver": "OSQP"},
            warnings=[]
        )
        
        assert result.doses == [10.0, 20.0, 15.0]
        assert result.status == SolverStatus.OPTIMAL
        assert result.objective_value == 45.0
        assert result.solve_time == 1.5
        assert result.diagnostics == {"solver": "OSQP"}
        assert result.warnings == []
    
    def test_optimization_result_with_warnings(self):
        """Test OptimizationResult with warnings."""
        result = OptimizationResult(
            doses=[10.0, 20.0, 15.0],
            status=SolverStatus.OPTIMAL,
            objective_value=45.0,
            solve_time=1.5,
            diagnostics={"solver": "OSQP"},
            warnings=["High doses detected", "Regulatory limit exceeded"]
        )
        
        assert len(result.warnings) == 2
        assert "High doses detected" in result.warnings
        assert "Regulatory limit exceeded" in result.warnings

# Integration tests
class TestOptimizerIntegration:
    """Integration tests for the optimizer."""
    
    def test_end_to_end_optimization(self):
        """Test complete optimization workflow."""
        optimizer = DoseOptimizer()
        
        # Create realistic plant data
        plants = [
            PlantInstance(
                id=i, bbox=[i*100, i*100, 50, 50], area=0.25, 
                severity=np.random.randint(1, 4), confidence=0.8 + np.random.random()*0.2,
                category_id=np.random.randint(1, 11), location=(i*100, i*100)
            )
            for i in range(1, 11)  # 10 plants
        ]
        
        # Run optimization
        result = optimizer.solve_doses(plants)
        
        # Verify results
        assert result.status in [SolverStatus.OPTIMAL, SolverStatus.INFEASIBLE]
        assert len(result.doses) == len(plants)
        assert result.solve_time > 0
        
        if result.status == SolverStatus.OPTIMAL:
            # Check that all doses are non-negative
            assert all(dose >= 0 for dose in result.doses)
            
            # Check that total dose is reasonable
            total_dose = sum(result.doses)
            assert total_dose <= 1000.0  # Regulatory limit
    
    def test_optimization_with_different_severities(self):
        """Test optimization with different severity levels."""
        optimizer = DoseOptimizer()
        
        # Create plants with different severities
        plants = [
            PlantInstance(
                id=1, bbox=[100, 100, 50, 50], area=0.25, severity=0, confidence=0.9,
                category_id=1, location=(100, 100)
            ),
            PlantInstance(
                id=2, bbox=[200, 200, 50, 50], area=0.25, severity=1, confidence=0.9,
                category_id=1, location=(200, 200)
            ),
            PlantInstance(
                id=3, bbox=[300, 300, 50, 50], area=0.25, severity=2, confidence=0.9,
                category_id=1, location=(300, 300)
            ),
            PlantInstance(
                id=4, bbox=[400, 400, 50, 50], area=0.25, severity=3, confidence=0.9,
                category_id=1, location=(400, 400)
            )
        ]
        
        result = optimizer.solve_doses(plants)
        
        if result.status == SolverStatus.OPTIMAL:
            # Higher severity should generally require higher doses
            # (though this depends on the specific parameters)
            assert len(result.doses) == 4
            assert all(dose >= 0 for dose in result.doses)

if __name__ == "__main__":
    pytest.main([__file__])
