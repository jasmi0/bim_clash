"""
3D Visualization Module for BIM Models
Uses Plotly for interactive 3D visualization
"""
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any
import logging


logger = logging.getLogger(__name__)


class BIMVisualizer:
    """3D visualization of BIM models with clash highlighting"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BIM Visualizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        
        # Colors
        self.default_color = self.viz_config.get('default_color', '#808080')
        self.clash_colors = self.viz_config.get('clash_colors', {
            'critical': '#FF0000',
            'high': '#FF6600',
            'medium': '#FFAA00',
            'low': '#FFFF00'
        })
        self.default_opacity = self.viz_config.get('default_opacity', 0.5)
        self.clash_opacity = self.viz_config.get('highlight_opacity', 0.7)
    
    def create_3d_model_view(
        self,
        elements: List[Dict],
        clashes: List = None,
        highlight_elements: List[str] = None,
        show_clash_connections: bool = False
    ) -> go.Figure:
        """
        Create interactive 3D view of BIM model
        
        Args:
            elements: List of BIM elements
            clashes: List of detected clashes
            highlight_elements: List of element GUIDs to highlight
            show_clash_connections: Whether to show lines connecting clashes to elements
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Create element lookup for clashes
        clashing_elements = set()
        element_lookup = {elem['guid']: elem for elem in elements}
        
        if clashes:
            for clash in clashes:
                clashing_elements.add(clash.element1_guid)
                clashing_elements.add(clash.element2_guid)
        
        # Add each element as a mesh or box
        for element in elements:
            if not element.get('bounding_box'):
                continue
            
            bbox = element['bounding_box']
            guid = element['guid']
            
            # Determine color
            if highlight_elements and guid in highlight_elements:
                color = '#00FF00'  # Green for highlighted
                opacity = 0.9
            elif guid in clashing_elements:
                color = '#FF0000'  # Red for clashing
                opacity = self.clash_opacity
            else:
                color = self._get_element_color(element['type'])
                opacity = self.default_opacity
            
            # Create bounding box mesh
            mesh = self._create_box_mesh(bbox, color, opacity, element)
            fig.add_trace(mesh)
        
        # Add clash markers
        if clashes:
            clash_points = self._create_clash_markers(clashes)
            fig.add_trace(clash_points)
            
            # Add connection lines from clash points to element centers
            if show_clash_connections:
                for clash in clashes:
                    elem1 = element_lookup.get(clash.element1_guid)
                    elem2 = element_lookup.get(clash.element2_guid)
                    
                    if elem1 and elem1.get('bounding_box') and elem2 and elem2.get('bounding_box'):
                        clash_center = clash.center_point
                        elem1_center = elem1['bounding_box']['center']
                        elem2_center = elem2['bounding_box']['center']
                        
                        # Create line from clash to element 1
                        fig.add_trace(go.Scatter3d(
                            x=[clash_center[0], elem1_center[0]],
                            y=[clash_center[1], elem1_center[1]],
                            z=[clash_center[2], elem1_center[2]],
                            mode='lines',
                            line=dict(color=self.clash_colors.get(clash.severity, '#FFFF00'), width=2, dash='dot'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Create line from clash to element 2
                        fig.add_trace(go.Scatter3d(
                            x=[clash_center[0], elem2_center[0]],
                            y=[clash_center[1], elem2_center[1]],
                            z=[clash_center[2], elem2_center[2]],
                            mode='lines',
                            line=dict(color=self.clash_colors.get(clash.severity, '#FFFF00'), width=2, dash='dot'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (m)', backgroundcolor="rgb(230, 230,230)"),
                yaxis=dict(title='Y (m)', backgroundcolor="rgb(230, 230,230)"),
                zaxis=dict(title='Z (m)', backgroundcolor="rgb(230, 230,230)"),
                aspectmode='data'
            ),
            title='BIM Model - 3D View',
            showlegend=True,
            hovermode='closest',
            height=700
        )
        
        return fig
    
    def _create_box_mesh(
        self,
        bbox: Dict,
        color: str,
        opacity: float,
        element: Dict
    ) -> go.Mesh3d:
        """Create a 3D box mesh from bounding box"""
        min_pt = bbox['min']
        max_pt = bbox['max']
        
        # Define vertices of the box
        x = [min_pt[0], max_pt[0], max_pt[0], min_pt[0],
             min_pt[0], max_pt[0], max_pt[0], min_pt[0]]
        y = [min_pt[1], min_pt[1], max_pt[1], max_pt[1],
             min_pt[1], min_pt[1], max_pt[1], max_pt[1]]
        z = [min_pt[2], min_pt[2], min_pt[2], min_pt[2],
             max_pt[2], max_pt[2], max_pt[2], max_pt[2]]
        
        # Define faces (triangles)
        i = [0, 0, 0, 0, 4, 4, 6, 6, 2, 2, 1, 1]
        j = [1, 2, 3, 7, 5, 6, 5, 2, 3, 6, 5, 2]
        k = [2, 3, 7, 4, 6, 7, 1, 1, 7, 7, 4, 6]
        
        hover_text = (
            f"<b>{element['type']}</b><br>"
            f"Name: {element.get('name', 'N/A')}<br>"
            f"GUID: {element['guid']}<br>"
            f"Material: {element['properties'].get('Material', 'N/A')}"
        )
        
        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=opacity,
            name=element['type'],
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=True,
            legendgroup=element['type']
        )
        
        return mesh
    
    def _create_clash_markers(self, clashes: List) -> go.Scatter3d:
        """Create markers for clash locations"""
        x_coords = []
        y_coords = []
        z_coords = []
        colors = []
        sizes = []
        hover_texts = []
        
        for clash in clashes:
            center = clash.center_point
            x_coords.append(center[0])
            y_coords.append(center[1])
            z_coords.append(center[2])
            
            # Color by severity
            colors.append(self.clash_colors.get(clash.severity, '#FFFF00'))
            
            # Size by severity (critical clashes are larger)
            severity_sizes = {
                'critical': 15,
                'high': 12,
                'medium': 10,
                'low': 8
            }
            sizes.append(severity_sizes.get(clash.severity, 10))
            
            hover_text = (
                f"<b>CLASH - {clash.severity.upper()}</b><br>"
                f"{clash.element1_type} vs {clash.element2_type}<br>"
                f"Distance: {clash.distance:.3f}m<br>"
                f"Overlap: {clash.overlap_volume:.4f}m³<br>"
                f"Type: {clash.clash_type}<br>"
                f"Location: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})<br>"
                f"{clash.description}"
            )
            hover_texts.append(hover_text)
        
        markers = go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                symbol='diamond',
                line=dict(color='white', width=2),
                opacity=0.9
            ),
            name='Clashes',
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=True
        )
        
        return markers
    
    def _get_element_color(self, element_type: str) -> str:
        """Get color for element type"""
        color_map = {
            'IfcWall': '#A9A9A9',
            'IfcSlab': '#808080',
            'IfcBeam': '#654321',
            'IfcColumn': '#8B4513',
            'IfcPipeSegment': '#4169E1',
            'IfcDuctSegment': '#87CEEB',
            'IfcCableCarrierSegment': '#FFD700',
            'IfcDoor': '#8B4513',
            'IfcWindow': '#87CEEB'
        }
        return color_map.get(element_type, self.default_color)
    
    def create_clash_distribution_chart(self, clashes: List) -> go.Figure:
        """Create chart showing clash distribution by severity"""
        severity_counts = {}
        for clash in clashes:
            severity_counts[clash.severity] = severity_counts.get(clash.severity, 0) + 1
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(severity_counts.keys()),
                y=list(severity_counts.values()),
                marker_color=[self.clash_colors.get(s, '#808080') for s in severity_counts.keys()],
                text=list(severity_counts.values()),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Clash Distribution by Severity',
            xaxis_title='Severity',
            yaxis_title='Number of Clashes',
            height=400
        )
        
        return fig
    
    def create_element_type_chart(self, elements: List[Dict]) -> go.Figure:
        """Create chart showing element type distribution"""
        type_counts = {}
        for elem in elements:
            elem_type = elem['type']
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                hole=0.3
            )
        ])
        
        fig.update_layout(
            title='BIM Elements Distribution',
            height=400
        )
        
        return fig
