"""
Optimization Engine Module
Suggests alternative solutions to resolve detected clashes
"""
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class OptimizationSuggestion:
    """Data class for optimization suggestions"""
    clash_id: str
    suggestion_type: str
    priority: int
    description: str
    element_to_modify: str
    modification: Dict[str, Any]
    estimated_cost: float
    feasibility_score: float
    detailed_steps: List[str]


class OptimizationEngine:
    """Generate optimization suggestions for resolving clashes"""
    
    def __init__(self, elements: List[Dict], clashes: List, config: Dict[str, Any]):
        """
        Initialize Optimization Engine
        
        Args:
            elements: List of BIM elements
            clashes: List of detected clashes
            config: Configuration dictionary
        """
        self.elements = elements
        self.clashes = clashes
        self.config = config
        self.suggestions = []
        
        # Create element lookup
        self.element_dict = {elem['guid']: elem for elem in elements}
        
        # Optimization settings
        self.opt_config = config.get('optimization', {})
        self.max_suggestions = self.opt_config.get('max_suggestions', 5)
        self.consider_cost = self.opt_config.get('consider_cost', True)
        self.consider_material = self.opt_config.get('consider_material', True)
    
    def generate_suggestions(self) -> List[OptimizationSuggestion]:
        """
        Generate optimization suggestions for all clashes
        
        Returns:
            List of optimization suggestions
        """
        logger.info("Generating optimization suggestions...")
        self.suggestions = []
        
        for clash in self.clashes:
            clash_suggestions = self._generate_clash_suggestions(clash)
            self.suggestions.extend(clash_suggestions)
        
        # Sort by priority and feasibility
        self.suggestions.sort(key=lambda x: (x.priority, -x.feasibility_score))
        
        logger.info(f"Generated {len(self.suggestions)} optimization suggestions")
        return self.suggestions
    
    def _generate_clash_suggestions(self, clash) -> List[OptimizationSuggestion]:
        """Generate suggestions for a single clash"""
        suggestions = []
        
        elem1 = self.element_dict.get(clash.element1_guid)
        elem2 = self.element_dict.get(clash.element2_guid)
        
        if not elem1 or not elem2:
            return suggestions
        
        # Generate different types of suggestions
        
        # 1. Relocation suggestions
        relocation_suggs = self._suggest_relocation(clash, elem1, elem2)
        suggestions.extend(relocation_suggs)
        
        # 2. Resize suggestions
        resize_suggs = self._suggest_resize(clash, elem1, elem2)
        suggestions.extend(resize_suggs)
        
        # 3. Route change suggestions (for MEP elements)
        route_suggs = self._suggest_route_change(clash, elem1, elem2)
        suggestions.extend(route_suggs)
        
        # 4. Design alternative suggestions
        alternative_suggs = self._suggest_design_alternative(clash, elem1, elem2)
        suggestions.extend(alternative_suggs)
        
        # Limit to max suggestions per clash
        suggestions = suggestions[:self.max_suggestions]
        
        return suggestions
    
    def _suggest_relocation(self, clash, elem1: Dict, elem2: Dict) -> List[OptimizationSuggestion]:
        """Suggest relocating one of the elements"""
        suggestions = []
        
        # Determine which element is easier to relocate
        movability_score1 = self._calculate_movability(elem1)
        movability_score2 = self._calculate_movability(elem2)
        
        # Suggest moving the more movable element
        if movability_score1 > movability_score2:
            element_to_move = elem1
            other_element = elem2
            movability = movability_score1
        else:
            element_to_move = elem2
            other_element = elem1
            movability = movability_score2
        
        # Calculate suggested movement direction
        bbox1 = np.array(elem1['bounding_box']['center'])
        bbox2 = np.array(elem2['bounding_box']['center'])
        
        direction = bbox1 - bbox2
        direction = direction / np.linalg.norm(direction)
        
        # Calculate required movement distance
        required_distance = clash.distance + 0.15  # Add 15cm clearance
        
        movement = direction * required_distance
        
        suggestion = OptimizationSuggestion(
            clash_id=f"{clash.element1_guid}_{clash.element2_guid}",
            suggestion_type="relocation",
            priority=self._calculate_priority(clash, "relocation"),
            description=f"Relocate {element_to_move['type']} by {required_distance:.2f}m to resolve clash",
            element_to_modify=element_to_move['guid'],
            modification={
                'type': 'translate',
                'direction': movement.tolist(),
                'distance': required_distance
            },
            estimated_cost=self._estimate_cost(element_to_move, 'relocation', required_distance),
            feasibility_score=movability * 0.8,
            detailed_steps=[
                f"1. Identify {element_to_move['type']} '{element_to_move['name']}'",
                f"2. Move element {required_distance:.2f}m in direction [{movement[0]:.2f}, {movement[1]:.2f}, {movement[2]:.2f}]",
                "3. Verify no new clashes are created",
                "4. Update connected elements and relationships"
            ]
        )
        
        suggestions.append(suggestion)
        return suggestions
    
    def _suggest_resize(self, clash, elem1: Dict, elem2: Dict) -> List[OptimizationSuggestion]:
        """Suggest resizing one of the elements"""
        suggestions = []
        
        # Check if elements are resizable
        resizable_types = ['IfcWall', 'IfcSlab', 'IfcBeam', 'IfcColumn']
        
        for elem in [elem1, elem2]:
            if elem['type'] in resizable_types:
                # Suggest reducing size
                bbox = elem['bounding_box']
                dims = np.array(bbox['max']) - np.array(bbox['min'])
                
                # Suggest reducing the smallest dimension
                min_dim_idx = np.argmin(dims)
                reduction = clash.overlap_volume / (dims[(min_dim_idx + 1) % 3] * dims[(min_dim_idx + 2) % 3])
                reduction = min(reduction + 0.05, dims[min_dim_idx] * 0.3)  # Max 30% reduction
                
                suggestion = OptimizationSuggestion(
                    clash_id=f"{clash.element1_guid}_{clash.element2_guid}",
                    suggestion_type="resize",
                    priority=self._calculate_priority(clash, "resize"),
                    description=f"Reduce {elem['type']} dimension by {reduction:.3f}m",
                    element_to_modify=elem['guid'],
                    modification={
                        'type': 'resize',
                        'axis': ['x', 'y', 'z'][min_dim_idx],
                        'amount': -reduction
                    },
                    estimated_cost=self._estimate_cost(elem, 'resize', reduction),
                    feasibility_score=0.6,
                    detailed_steps=[
                        f"1. Access {elem['type']} '{elem['name']}' properties",
                        f"2. Reduce {['width', 'depth', 'height'][min_dim_idx]} by {reduction:.3f}m",
                        "3. Verify structural integrity is maintained",
                        "4. Update material quantities"
                    ]
                )
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_route_change(self, clash, elem1: Dict, elem2: Dict) -> List[OptimizationSuggestion]:
        """Suggest changing route for MEP elements"""
        suggestions = []
        
        mep_types = ['IfcPipeSegment', 'IfcDuctSegment', 'IfcCableCarrierSegment']
        
        for elem in [elem1, elem2]:
            if elem['type'] in mep_types:
                suggestion = OptimizationSuggestion(
                    clash_id=f"{clash.element1_guid}_{clash.element2_guid}",
                    suggestion_type="route_change",
                    priority=self._calculate_priority(clash, "route_change"),
                    description=f"Re-route {elem['type']} to avoid clash",
                    element_to_modify=elem['guid'],
                    modification={
                        'type': 'reroute',
                        'strategy': 'offset',
                        'offset_distance': 0.3
                    },
                    estimated_cost=self._estimate_cost(elem, 'reroute', 1.0),
                    feasibility_score=0.75,
                    detailed_steps=[
                        f"1. Identify current route of {elem['type']}",
                        "2. Calculate alternative route with 30cm offset",
                        "3. Verify sufficient clearance on new route",
                        "4. Update connected fittings and supports",
                        "5. Verify system functionality is maintained"
                    ]
                )
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_design_alternative(self, clash, elem1: Dict, elem2: Dict) -> List[OptimizationSuggestion]:
        """Suggest alternative design solutions"""
        suggestions = []
        
        # Suggest using different component sizes or types
        if elem1['type'] in ['IfcPipeSegment', 'IfcDuctSegment']:
            suggestion = OptimizationSuggestion(
                clash_id=f"{clash.element1_guid}_{clash.element2_guid}",
                suggestion_type="design_alternative",
                priority=self._calculate_priority(clash, "design_alternative"),
                description=f"Use smaller diameter {elem1['type']} if flow requirements allow",
                element_to_modify=elem1['guid'],
                modification={
                    'type': 'component_change',
                    'change': 'reduce_diameter'
                },
                estimated_cost=self._estimate_cost(elem1, 'component_change', 0.5),
                feasibility_score=0.5,
                detailed_steps=[
                    "1. Review flow requirements and pressure drop calculations",
                    "2. Identify smaller diameter option that meets requirements",
                    "3. Update component specification",
                    "4. Verify system performance is acceptable"
                ]
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_movability(self, element: Dict) -> float:
        """Calculate how easy it is to move an element (0-1)"""
        # MEP elements are generally more movable
        if element['type'] in ['IfcPipeSegment', 'IfcDuctSegment', 'IfcCableCarrierSegment']:
            return 0.9
        
        # Doors and windows are moderately movable
        if element['type'] in ['IfcDoor', 'IfcWindow']:
            return 0.6
        
        # Structural elements are less movable
        if element['type'] in ['IfcBeam', 'IfcColumn']:
            return 0.3
        
        # Walls and slabs are least movable
        if element['type'] in ['IfcWall', 'IfcSlab']:
            return 0.2
        
        return 0.5
    
    def _calculate_priority(self, clash, suggestion_type: str) -> int:
        """Calculate priority of suggestion (1-5, 1 being highest)"""
        severity_priority = {
            'critical': 1,
            'high': 2,
            'medium': 3,
            'low': 4
        }
        
        base_priority = severity_priority.get(clash.severity, 5)
        
        # Adjust based on suggestion type
        if suggestion_type == 'route_change':
            base_priority = max(1, base_priority - 1)  # Prefer rerouting MEP
        elif suggestion_type == 'resize':
            base_priority = min(5, base_priority + 1)  # Deprioritize resizing
        
        return base_priority
    
    def _estimate_cost(self, element: Dict, modification_type: str, magnitude: float) -> float:
        """Estimate cost of modification (relative scale)"""
        base_costs = {
            'relocation': 100,
            'resize': 150,
            'reroute': 80,
            'component_change': 120
        }
        
        base_cost = base_costs.get(modification_type, 100)
        
        # Scale by element type
        type_multipliers = {
            'IfcWall': 2.0,
            'IfcSlab': 2.5,
            'IfcBeam': 1.5,
            'IfcColumn': 1.8,
            'IfcPipeSegment': 0.8,
            'IfcDuctSegment': 0.9,
            'IfcCableCarrierSegment': 0.7
        }
        
        multiplier = type_multipliers.get(element['type'], 1.0)
        
        # Scale by magnitude
        cost = base_cost * multiplier * (1 + magnitude)
        
        return round(cost, 2)
    
    def get_suggestions_by_clash(self, clash_id: str) -> List[OptimizationSuggestion]:
        """Get all suggestions for a specific clash"""
        return [s for s in self.suggestions if s.clash_id == clash_id]
    
    def get_suggestions_by_priority(self, priority: int) -> List[OptimizationSuggestion]:
        """Get all suggestions with specific priority"""
        return [s for s in self.suggestions if s.priority == priority]
    
    def export_suggestions_to_dict(self) -> List[Dict]:
        """Export suggestions as list of dictionaries"""
        return [
            {
                'clash_id': s.clash_id,
                'suggestion_type': s.suggestion_type,
                'priority': s.priority,
                'description': s.description,
                'element_to_modify': s.element_to_modify,
                'modification': s.modification,
                'estimated_cost': s.estimated_cost,
                'feasibility_score': s.feasibility_score,
                'detailed_steps': s.detailed_steps
            }
            for s in self.suggestions
        ]
