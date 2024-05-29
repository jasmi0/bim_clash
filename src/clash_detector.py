from __future__ import annotations
"""
Clash Detection Module
Implements geometric-based clash detection algorithms
"""
import numpy as np
from typing import List, Dict, Tuple, Any
import logging
from scipy.spatial.distance import cdist
from shapely.geometry import box, Polygon
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class Clash:
    """Data class for representing a clash"""
    element1_id: str
    element2_id: str
    element1_guid: str
    element2_guid: str
    element1_type: str
    element2_type: str
    clash_type: str
    severity: str
    distance: float
    overlap_volume: float
    center_point: Tuple[float, float, float]
    description: str


class ClashDetector:
    """Geometric clash detection between BIM elements"""
    
    def __init__(self, elements: List[Dict], config: Dict[str, Any]):
        """
        Initialize Clash Detector
        
        Args:
            elements: List of BIM elements
            config: Configuration dictionary with thresholds
        """
        self.elements = elements
        self.config = config
        self.clashes = []
        
        # Get thresholds from config
        self.distance_threshold = config.get('clash_detection', {}).get('distance_threshold', 0.05)
        self.severity_levels = config.get('clash_detection', {}).get('severity_levels', {
            'critical': 0.0,
            'high': 0.02,
            'medium': 0.05,
            'low': 0.10
        })
    
    def detect_clashes(self) -> List[Clash]:
        """
        Detect clashes between all elements
        
        Returns:
            List of detected clashes
        """
        logger.info("Starting clash detection...")
        self.clashes = []
        
        n_elements = len(self.elements)
        
        # Compare each pair of elements
        for i in range(n_elements):
            for j in range(i + 1, n_elements):
                clash = self._check_element_pair(self.elements[i], self.elements[j])
                if clash:
                    self.clashes.append(clash)
        
        logger.info(f"Detected {len(self.clashes)} clashes")
        return self.clashes
    
    def _check_element_pair(self, elem1: Dict, elem2: Dict) -> Clash | None:
        """
        Check if two elements clash
        
        Args:
            elem1: First element
            elem2: Second element
            
        Returns:
            Clash object if clash detected, None otherwise
        """
        # Skip if either element has no geometry
        if not elem1.get('bounding_box') or not elem2.get('bounding_box'):
            return None
        
        bbox1 = elem1['bounding_box']
        bbox2 = elem2['bounding_box']
        
        # Check bounding box intersection
        if not self._bounding_boxes_intersect(bbox1, bbox2):
            return None
        
        # Calculate minimum distance between bounding boxes
        distance = self._calculate_bbox_distance(bbox1, bbox2)
        
        # Check if distance is below threshold
        if distance > self.distance_threshold:
            return None
        
        # Determine clash severity
        severity = self._determine_severity(distance)
        
        # Calculate overlap volume (approximate)
        overlap_volume = self._calculate_overlap_volume(bbox1, bbox2)
        
        # Determine clash type
        clash_type = self._determine_clash_type(elem1, elem2, distance)
        
        # Calculate center point of clash (intersection center, not element centers)
        clash_center = self._calculate_intersection_center(bbox1, bbox2)
        
        # Create description
        description = f"{elem1['type']} '{elem1['name']}' clashes with {elem2['type']} '{elem2['name']}' - {severity} severity"
        
        clash = Clash(
            element1_id=str(elem1['id']),
            element2_id=str(elem2['id']),
            element1_guid=elem1['guid'],
            element2_guid=elem2['guid'],
            element1_type=elem1['type'],
            element2_type=elem2['type'],
            clash_type=clash_type,
            severity=severity,
            distance=distance,
            overlap_volume=overlap_volume,
            center_point=tuple(clash_center),
            description=description
        )
        
        return clash
    
    def _bounding_boxes_intersect(self, bbox1: Dict, bbox2: Dict) -> bool:
        """Check if two bounding boxes intersect"""
        min1 = np.array(bbox1['min'])
        max1 = np.array(bbox1['max'])
        min2 = np.array(bbox2['min'])
        max2 = np.array(bbox2['max'])
        
        # Check for separation on each axis
        for i in range(3):
            if max1[i] < min2[i] or max2[i] < min1[i]:
                return False
        
        return True
    
    def _calculate_bbox_distance(self, bbox1: Dict, bbox2: Dict) -> float:
        """
        Calculate minimum distance between two bounding boxes
        Returns 0 if they intersect
        """
        min1 = np.array(bbox1['min'])
        max1 = np.array(bbox1['max'])
        min2 = np.array(bbox2['min'])
        max2 = np.array(bbox2['max'])
        
        # Calculate distance on each axis
        dx = max(0, max(min1[0] - max2[0], min2[0] - max1[0]))
        dy = max(0, max(min1[1] - max2[1], min2[1] - max1[1]))
        dz = max(0, max(min1[2] - max2[2], min2[2] - max1[2]))
        
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        return distance
    
    def _calculate_overlap_volume(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate approximate overlap volume between bounding boxes"""
        min1 = np.array(bbox1['min'])
        max1 = np.array(bbox1['max'])
        min2 = np.array(bbox2['min'])
        max2 = np.array(bbox2['max'])
        
        # Calculate intersection box
        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        
        # Check if there's overlap
        if np.any(inter_min >= inter_max):
            return 0.0
        
        # Calculate volume
        dims = inter_max - inter_min
        volume = np.prod(dims)
        
        return volume
    
    def _calculate_intersection_center(self, bbox1: Dict, bbox2: Dict) -> Tuple[float, float, float]:
        """
        Calculate the center point of the intersection between two bounding boxes
        If boxes don't intersect, return the midpoint on the line connecting their centers
        """
        min1 = np.array(bbox1['min'])
        max1 = np.array(bbox1['max'])
        min2 = np.array(bbox2['min'])
        max2 = np.array(bbox2['max'])
        
        # Calculate intersection box
        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        
        # Check if there's actual overlap
        if np.all(inter_min < inter_max):
            # Boxes intersect - return center of intersection volume
            intersection_center = (inter_min + inter_max) / 2.0
            return tuple(intersection_center.tolist())
        else:
            # Boxes don't overlap - find closest points between boxes
            # For clearance clashes, return point on line between centers
            center1 = np.array(bbox1['center'])
            center2 = np.array(bbox2['center'])
            
            # Find the closest point on each box to the other box's center
            # Clamp the other center to this box's bounds
            closest_point1 = np.clip(center2, min1, max1)
            closest_point2 = np.clip(center1, min2, max2)
            
            # Return midpoint between closest points
            clash_point = (closest_point1 + closest_point2) / 2.0
            return tuple(clash_point.tolist())
    
    def _determine_severity(self, distance: float) -> str:
        """Determine clash severity based on distance"""
        for severity, threshold in sorted(self.severity_levels.items(), key=lambda x: x[1]):
            if distance <= threshold:
                return severity
        return 'low'
    
    def _determine_clash_type(self, elem1: Dict, elem2: Dict, distance: float) -> str:
        """Determine the type of clash"""
        if distance == 0.0:
            return 'hard_clash'  # Complete overlap
        elif distance < 0.01:
            return 'soft_clash'  # Very close proximity
        else:
            return 'clearance_clash'  # Insufficient clearance
    
    def get_clashes_by_severity(self, severity: str) -> List[Clash]:
        """Get all clashes of a specific severity"""
        return [clash for clash in self.clashes if clash.severity == severity]
    
    def get_clashes_by_element(self, element_guid: str) -> List[Clash]:
        """Get all clashes involving a specific element"""
        return [
            clash for clash in self.clashes
            if clash.element1_guid == element_guid or clash.element2_guid == element_guid
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get clash detection statistics"""
        severity_counts = {}
        type_counts = {}
        
        for clash in self.clashes:
            severity_counts[clash.severity] = severity_counts.get(clash.severity, 0) + 1
            type_counts[clash.clash_type] = type_counts.get(clash.clash_type, 0) + 1
        
        return {
            'total_clashes': len(self.clashes),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'critical_count': severity_counts.get('critical', 0),
            'elements_affected': len(set(
                [c.element1_guid for c in self.clashes] + 
                [c.element2_guid for c in self.clashes]
            ))
        }
    
    def export_clashes_to_dict(self) -> List[Dict]:
        """Export clashes as list of dictionaries"""
        return [
            {
                'element1_id': clash.element1_id,
                'element2_id': clash.element2_id,
                'element1_guid': clash.element1_guid,
                'element2_guid': clash.element2_guid,
                'element1_type': clash.element1_type,
                'element2_type': clash.element2_type,
                'clash_type': clash.clash_type,
                'severity': clash.severity,
                'distance': clash.distance,
                'overlap_volume': clash.overlap_volume,
                'center_point': clash.center_point,
                'description': clash.description
            }
            for clash in self.clashes
        ]
