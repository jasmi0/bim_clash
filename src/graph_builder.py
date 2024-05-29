"""
Graph Builder Module
Converts BIM elements into graph structure for GNN processing
"""
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple
import logging
from scipy.spatial.distance import cdist


logger = logging.getLogger(__name__)


class BIMGraphBuilder:
    """Build graph representation of BIM model for GNN processing"""
    
    def __init__(self, elements: List[Dict], distance_threshold: float = 2.0):
        """
        Initialize Graph Builder
        
        Args:
            elements: List of BIM elements from parser
            distance_threshold: Maximum distance for creating edges between elements
        """
        self.elements = elements
        self.distance_threshold = distance_threshold
        self.node_features = []
        self.edge_index = None
        self.edge_features = []
        
    def build_graph(self) -> Data:
        """
        Build PyTorch Geometric graph from BIM elements
        
        Returns:
            PyTorch Geometric Data object
        """
        logger.info("Building graph from BIM elements...")
        
        # Extract node features
        self.node_features = self._extract_node_features()
        
        # Build edges based on spatial proximity and relationships
        self.edge_index, self.edge_features = self._build_edges()
        
        # Create PyTorch Geometric Data object
        graph_data = Data(
            x=torch.FloatTensor(self.node_features),
            edge_index=torch.LongTensor(self.edge_index),
            edge_attr=torch.FloatTensor(self.edge_features),
            num_nodes=len(self.elements)
        )
        
        logger.info(f"Graph built: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        return graph_data
    
    def _extract_node_features(self) -> np.ndarray:
        """
        Extract node features from elements
        
        Features include:
        - Element type (one-hot encoded with fixed categories)
        - Bounding box dimensions
        - Center coordinates
        - Volume
        - Material properties (encoded)
        
        Returns:
            Feature matrix of shape (n_elements, n_features)
        """
        features = []
        
        # Define FIXED element type mapping for consistency across all graphs
        FIXED_ELEMENT_TYPES = [
            'IfcWall', 'IfcBeam', 'IfcColumn', 'IfcSlab',
            'IfcPipeSegment', 'IfcDuctSegment', 'IfcCableCarrierSegment',
            'IfcDoor', 'IfcWindow', 'IfcRoof', 'IfcStair', 'IfcRailing',
            'IfcPlate', 'IfcMember', 'IfcFooting', 'IfcPile',
            'IfcCurtainWall', 'IfcBearing', 'IfcOther'
        ]
        n_types = len(FIXED_ELEMENT_TYPES)
        type_to_idx = {etype: idx for idx, etype in enumerate(FIXED_ELEMENT_TYPES)}
        
        for element in self.elements:
            feature_vector = []
            
            # One-hot encode element type with fixed vocabulary
            type_encoding = [0] * n_types
            elem_type = element['type']
            if elem_type in type_to_idx:
                type_encoding[type_to_idx[elem_type]] = 1
            else:
                # Unknown types map to 'IfcOther'
                type_encoding[type_to_idx['IfcOther']] = 1
            feature_vector.extend(type_encoding)
            
            # Geometric features
            if element['bounding_box']:
                bbox = element['bounding_box']
                min_coords = np.array(bbox['min'])
                max_coords = np.array(bbox['max'])
                center = np.array(bbox['center'])
                
                # Dimensions
                dimensions = max_coords - min_coords
                feature_vector.extend(dimensions.tolist())
                
                # Center coordinates (normalized)
                feature_vector.extend(center.tolist())
                
                # Volume
                volume = np.prod(dimensions)
                feature_vector.append(volume)
                
            else:
                # Use zeros if no geometry available
                feature_vector.extend([0] * 7)  # 3 dims + 3 center + 1 volume
            
            # Material encoding (simple hash for now)
            material = element['properties'].get('Material', 'Unknown')
            material_hash = hash(material) % 100 / 100.0  # Normalize to [0, 1]
            feature_vector.append(material_hash)
            
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def _build_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build edges based on spatial proximity and relationships
        
        Returns:
            Tuple of (edge_index, edge_features)
            edge_index: shape (2, n_edges)
            edge_features: shape (n_edges, n_edge_features)
        """
        edges = []
        edge_attrs = []
        
        # Get centers for distance calculation
        centers = []
        for element in self.elements:
            if element['bounding_box']:
                centers.append(element['bounding_box']['center'])
            else:
                centers.append([0, 0, 0])
        
        centers = np.array(centers)
        
        # Handle edge case: no elements or single element
        if centers.shape[0] < 2:
            # Return empty edges
            return torch.tensor([[], []], dtype=torch.long), torch.tensor([], dtype=torch.float)
        
        # Ensure centers is 2D
        if centers.ndim == 1:
            centers = centers.reshape(-1, 1)
        
        # Calculate pairwise distances
        distances = cdist(centers, centers)
        
        # Create edges for nearby elements
        n_elements = len(self.elements)
        for i in range(n_elements):
            for j in range(i + 1, n_elements):
                distance = distances[i, j]
                
                # Create edge if within threshold
                if distance < self.distance_threshold:
                    # Add bidirectional edges
                    edges.append([i, j])
                    edges.append([j, i])
                    
                    # Edge features: distance, relative position
                    rel_pos = centers[j] - centers[i]
                    rel_distance = distance
                    
                    edge_feature = [rel_distance] + rel_pos.tolist()
                    edge_attrs.append(edge_feature)
                    edge_attrs.append([-rel_distance] + (-rel_pos).tolist())
        
        # Add edges based on relationships
        for idx, element in enumerate(self.elements):
            relationships = element.get('relationships', {})
            
            # Find connected elements
            for rel_type in ['contains', 'connects']:
                for related_guid in relationships.get(rel_type, []):
                    # Find index of related element
                    for j, elem in enumerate(self.elements):
                        if elem['guid'] == related_guid:
                            # Add relationship edge
                            edges.append([idx, j])
                            
                            # Edge feature with special marker for relationship
                            rel_pos = np.array(centers[j]) - np.array(centers[idx])
                            distance = np.linalg.norm(rel_pos)
                            edge_feature = [distance] + rel_pos.tolist()
                            edge_attrs.append(edge_feature)
                            break
        
        if not edges:
            # If no edges, create self-loops
            logger.warning("No edges found, creating self-loops")
            edges = [[i, i] for i in range(n_elements)]
            edge_attrs = [[0, 0, 0, 0] for _ in range(n_elements)]
        
        edge_index = np.array(edges, dtype=np.int64).T
        edge_features = np.array(edge_attrs, dtype=np.float32)
        
        return edge_index, edge_features
    
    def get_element_mapping(self) -> Dict[int, Dict]:
        """Get mapping from node index to element data"""
        return {idx: elem for idx, elem in enumerate(self.elements)}
