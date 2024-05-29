"""
BIM Data Parser Module
Handles parsing and extraction of building elements from IFC files
"""
import ifcopenshell
import ifcopenshell.geom
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class BIMParser:
    """Parse IFC files and extract building elements and their properties"""
    
    def __init__(self, ifc_file_path: str):
        """
        Initialize BIM Parser
        
        Args:
            ifc_file_path: Path to the IFC file
        """
        self.ifc_file_path = ifc_file_path
        self.ifc_file = None
        self.elements = []
        self.element_dict = {}
        
    def load_ifc_file(self) -> bool:
        """Load IFC file"""
        try:
            self.ifc_file = ifcopenshell.open(self.ifc_file_path)
            logger.info(f"Successfully loaded IFC file: {self.ifc_file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading IFC file: {str(e)}")
            return False
    
    def extract_elements(self, element_types: List[str] = None) -> List[Dict]:
        """
        Extract building elements from IFC file
        
        Args:
            element_types: List of IFC element types to extract
            
        Returns:
            List of element dictionaries
        """
        if not self.ifc_file:
            logger.error("IFC file not loaded")
            return []
        
        if element_types is None:
            element_types = [
                'IfcWall', 'IfcBeam', 'IfcColumn', 'IfcSlab',
                'IfcPipeSegment', 'IfcDuctSegment', 'IfcCableCarrierSegment',
                'IfcDoor', 'IfcWindow'
            ]
        
        self.elements = []
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        
        for element_type in element_types:
            try:
                elements = self.ifc_file.by_type(element_type)
                logger.info(f"Found {len(elements)} elements of type {element_type}")
                
                for element in elements:
                    element_data = self._extract_element_data(element, settings)
                    if element_data:
                        self.elements.append(element_data)
                        self.element_dict[element_data['guid']] = element_data
                        
            except Exception as e:
                logger.warning(f"Error extracting {element_type}: {str(e)}")
                continue
        
        logger.info(f"Total elements extracted: {len(self.elements)}")
        return self.elements
    
    def _extract_element_data(self, element, settings) -> Dict[str, Any]:
        """
        Extract data from a single IFC element
        
        Args:
            element: IFC element
            settings: Geometry settings
            
        Returns:
            Dictionary with element data
        """
        try:
            element_data = {
                'guid': element.GlobalId,
                'type': element.is_a(),
                'name': element.Name if hasattr(element, 'Name') else '',
                'id': element.id(),
            }
            
            # Extract geometry
            try:
                shape = ifcopenshell.geom.create_shape(settings, element)
                
                # Get bounding box
                verts = shape.geometry.verts
                vertices = np.array(verts).reshape(-1, 3)
                
                element_data['bounding_box'] = {
                    'min': vertices.min(axis=0).tolist(),
                    'max': vertices.max(axis=0).tolist(),
                    'center': vertices.mean(axis=0).tolist()
                }
                
                element_data['vertices'] = vertices.tolist()
                element_data['geometry'] = {
                    'verts': verts,
                    'faces': shape.geometry.faces,
                    'edges': shape.geometry.edges
                }
                
            except Exception as e:
                logger.warning(f"Could not extract geometry for {element.GlobalId}: {str(e)}")
                element_data['bounding_box'] = None
                element_data['geometry'] = None
            
            # Extract properties
            element_data['properties'] = self._extract_properties(element)
            
            # Extract relationships
            element_data['relationships'] = self._extract_relationships(element)
            
            return element_data
            
        except Exception as e:
            logger.error(f"Error extracting element data: {str(e)}")
            return None
    
    def _extract_properties(self, element) -> Dict[str, Any]:
        """Extract properties from an IFC element"""
        properties = {}
        
        try:
            # Get property sets
            if hasattr(element, 'IsDefinedBy'):
                for definition in element.IsDefinedBy:
                    if definition.is_a('IfcRelDefinesByProperties'):
                        property_set = definition.RelatingPropertyDefinition
                        if property_set.is_a('IfcPropertySet'):
                            for prop in property_set.HasProperties:
                                if prop.is_a('IfcPropertySingleValue'):
                                    properties[prop.Name] = str(prop.NominalValue.wrappedValue) if prop.NominalValue else None
            
            # Get material
            if hasattr(element, 'HasAssociations'):
                for association in element.HasAssociations:
                    if association.is_a('IfcRelAssociatesMaterial'):
                        material = association.RelatingMaterial
                        if material.is_a('IfcMaterial'):
                            properties['Material'] = material.Name
                        elif material.is_a('IfcMaterialLayerSetUsage'):
                            properties['Material'] = material.ForLayerSet.LayerSetName
                            
        except Exception as e:
            logger.warning(f"Error extracting properties: {str(e)}")
        
        return properties
    
    def _extract_relationships(self, element) -> Dict[str, List[str]]:
        """Extract relationships from an IFC element"""
        relationships = {
            'contains': [],
            'contained_by': [],
            'connects': []
        }
        
        try:
            # Spatial containment
            if hasattr(element, 'ContainedInStructure'):
                for rel in element.ContainedInStructure:
                    if rel.RelatingStructure:
                        relationships['contained_by'].append(rel.RelatingStructure.GlobalId)
            
            # Contains elements
            if hasattr(element, 'ContainsElements'):
                for rel in element.ContainsElements:
                    for elem in rel.RelatedElements:
                        relationships['contains'].append(elem.GlobalId)
            
            # Connections
            if hasattr(element, 'ConnectedTo'):
                for rel in element.ConnectedTo:
                    if rel.RelatedElement:
                        relationships['connects'].append(rel.RelatedElement.GlobalId)
                        
        except Exception as e:
            logger.warning(f"Error extracting relationships: {str(e)}")
        
        return relationships
    
    def get_element_by_guid(self, guid: str) -> Dict[str, Any]:
        """Get element data by GUID"""
        return self.element_dict.get(guid)
    
    def get_elements_by_type(self, element_type: str) -> List[Dict]:
        """Get all elements of a specific type"""
        return [elem for elem in self.elements if elem['type'] == element_type]
    
    def get_bounding_boxes(self) -> np.ndarray:
        """
        Get all bounding boxes as numpy array
        
        Returns:
            Array of shape (n_elements, 6) with [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        bboxes = []
        for element in self.elements:
            if element['bounding_box']:
                bbox = element['bounding_box']
                bboxes.append(bbox['min'] + bbox['max'])
        
        return np.array(bboxes)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the parsed BIM model"""
        element_types = {}
        for element in self.elements:
            elem_type = element['type']
            element_types[elem_type] = element_types.get(elem_type, 0) + 1
        
        return {
            'total_elements': len(self.elements),
            'element_types': element_types,
            'has_geometry': sum(1 for e in self.elements if e['bounding_box'] is not None)
        }
