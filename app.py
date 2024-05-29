"""
Main Streamlit Application for BIM Clash Detection
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.bim_parser import BIMParser
from src.clash_detector import ClashDetector
from src.optimization_engine import OptimizationEngine
from src.visualizer import BIMVisualizer
from src.graph_builder import BIMGraphBuilder
from src.gnn_model import ClashDetectionModel
from utils.helpers import load_config, setup_logging, ensure_directories
import os
import torch


# Page configuration
st.set_page_config(
    page_title="BIM Clash Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
logger = setup_logging()
ensure_directories()

# Load configuration
config = load_config()


def initialize_session_state():
    """Initialize session state variables"""
    if 'ifc_file_uploaded' not in st.session_state:
        st.session_state.ifc_file_uploaded = False
    if 'elements' not in st.session_state:
        st.session_state.elements = []
    if 'clashes' not in st.session_state:
        st.session_state.clashes = []
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []
    if 'parser' not in st.session_state:
        st.session_state.parser = None
    if 'gnn_model' not in st.session_state:
        st.session_state.gnn_model = None
    if 'use_gnn' not in st.session_state:
        st.session_state.use_gnn = False


def main():
    """Main application"""
    initialize_session_state()
    
    # Sidebar
    st.sidebar.title("BIM Clash Detection")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Upload & Parse", "Clash Detection", "Optimization", "Visualization", "Reports"]
    )
    
    # Display selected page
    if page == "Upload & Parse":
        upload_and_parse_page()
    elif page == "Clash Detection":
        clash_detection_page()
    elif page == "Optimization":
        optimization_page()
    elif page == "Visualization":
        visualization_page()
    elif page == "Reports":
        reports_page()
    
    # Sidebar statistics
    if st.session_state.elements:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Statistics")
        st.sidebar.metric("Total Elements", len(st.session_state.elements))
        st.sidebar.metric("Detected Clashes", len(st.session_state.clashes))
        st.sidebar.metric("Optimization Suggestions", len(st.session_state.suggestions))


def upload_and_parse_page():
    """Upload and parse IFC file page"""
    st.title("Upload & Parse BIM Model")
    
    st.markdown("""
    Upload an IFC file to begin clash detection analysis. The system will parse the BIM model
    and extract all building elements along with their properties and relationships.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an IFC file",
        type=['ifc'],
        help="Upload a BIM model in IFC format"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        file_path = Path("data/uploaded_files") / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Parse button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Parse IFC File", type="primary"):
                with st.spinner("Parsing IFC file..."):
                    try:
                        # Initialize parser
                        parser = BIMParser(str(file_path))
                        
                        # Load IFC file
                        if parser.load_ifc_file():
                            st.success("IFC file loaded successfully")
                            
                            # Extract elements
                            element_types = config.get('clash_detection', {}).get('element_types', None)
                            elements = parser.extract_elements(element_types)
                            
                            # Save to session state
                            st.session_state.parser = parser
                            st.session_state.elements = elements
                            st.session_state.ifc_file_uploaded = True
                            
                            # Display statistics
                            stats = parser.get_statistics()
                            
                            st.success(f"Extracted {stats['total_elements']} elements")
                            
                            # Show element type distribution
                            st.subheader("Element Distribution")
                            df_types = pd.DataFrame(
                                list(stats['element_types'].items()),
                                columns=['Element Type', 'Count']
                            ).sort_values('Count', ascending=False)
                            
                            st.dataframe(df_types, width="stretch")
                            
                            logger.info(f"Successfully parsed {uploaded_file.name}")
                            
                        else:
                            st.error("Failed to load IFC file")
                            
                    except Exception as e:
                        st.error(f"Error parsing IFC file: {str(e)}")
                        logger.error(f"Error parsing IFC: {str(e)}")
    
    # Display parsed elements
    if st.session_state.elements:
        st.markdown("---")
        st.subheader("Parsed Elements")
        
        # Create DataFrame
        elements_data = []
        for elem in st.session_state.elements:
            elements_data.append({
                'Type': elem['type'],
                'Name': elem.get('name', 'N/A'),
                'GUID': elem['guid'],
                'Has Geometry': 'Yes' if elem.get('bounding_box') else 'No',
                'Material': elem['properties'].get('Material', 'N/A')
            })
        
        df = pd.DataFrame(elements_data)
        
        # Filter by type
        selected_types = st.multiselect(
            "Filter by Element Type",
            options=df['Type'].unique(),
            default=df['Type'].unique()
        )
        
        df_filtered = df[df['Type'].isin(selected_types)]
        st.dataframe(df_filtered, width="stretch", height=400)


def clash_detection_page():
    """Clash detection page"""
    st.title("Clash Detection")
    
    if not st.session_state.elements:
        st.warning("Please upload and parse an IFC file first!")
        return
    
    st.markdown("""
    Detect clashes between building elements. The system uses geometric analysis
    to identify overlaps and insufficient clearances.
    """)
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        distance_threshold = st.slider(
            "Distance Threshold (m)",
            min_value=0.0,
            max_value=0.5,
            value=config.get('clash_detection', {}).get('distance_threshold', 0.05),
            step=0.01,
            help="Maximum distance for clash detection"
        )
    
    with col2:
        # Check if GNN model exists
        gnn_model_path = 'models/saved_models/clash_gnn.pth'
        gnn_available = os.path.exists(gnn_model_path)
        
        use_gnn = st.checkbox(
            "Use GNN Enhancement",
            value=gnn_available,
            disabled=not gnn_available,
            help="Use trained GNN model for faster clash screening" if gnn_available else "No trained GNN model found"
        )
        st.session_state.use_gnn = use_gnn and gnn_available
    
    with col3:
        st.info(f"**Analyzing {len(st.session_state.elements)} elements**")
        if st.session_state.use_gnn:
            st.success("GNN enabled")
    
    # Detect clashes button
    if st.button("Detect Clashes", type="primary"):
        with st.spinner("Detecting clashes..."):
            try:
                # Update config
                config['clash_detection']['distance_threshold'] = distance_threshold
                
                # GNN-enhanced detection if enabled
                if st.session_state.use_gnn:
                    st.info("Using GNN for fast clash screening...")
                    
                    # Build graph
                    graph_builder = BIMGraphBuilder(st.session_state.elements)
                    graph = graph_builder.build_graph()
                    
                    # Load GNN model if not already loaded
                    if st.session_state.gnn_model is None:
                        st.session_state.gnn_model = ClashDetectionModel(
                            in_channels=graph.num_node_features,
                            hidden_channels=128,
                            num_layers=3
                        )
                        st.session_state.gnn_model.load_model(gnn_model_path)
                    
                    # Get GNN predictions
                    with st.spinner("Running GNN prediction..."):
                        predictions = st.session_state.gnn_model.predict(graph)
                        high_risk_count = (predictions > 0.5).sum().item()
                        st.info(f"GNN identified {high_risk_count} high-risk element pairs")
                
                # Initialize detector
                detector = ClashDetector(st.session_state.elements, config)
                
                # Detect clashes
                clashes = detector.detect_clashes()
                
                # If GNN was used, add GNN scores to clashes
                if st.session_state.use_gnn:
                    for clash in clashes:
                        # Find corresponding edge in graph and add GNN score
                        elem1_idx = next((i for i, e in enumerate(st.session_state.elements) 
                                        if e.get('guid') == clash.element1_guid), None)
                        elem2_idx = next((i for i, e in enumerate(st.session_state.elements) 
                                        if e.get('guid') == clash.element2_guid), None)
                        
                        if elem1_idx is not None and elem2_idx is not None:
                            edge_mask = (
                                ((graph.edge_index[0] == elem1_idx) & (graph.edge_index[1] == elem2_idx)) |
                                ((graph.edge_index[0] == elem2_idx) & (graph.edge_index[1] == elem1_idx))
                            )
                            
                            if edge_mask.any():
                                edge_idx = edge_mask.nonzero(as_tuple=True)[0][0]
                                clash.gnn_score = predictions[edge_idx].item()
                            else:
                                clash.gnn_score = 0.0
                        else:
                            clash.gnn_score = 0.0
                    
                    # Sort clashes by GNN score
                    clashes.sort(key=lambda c: getattr(c, 'gnn_score', 0), reverse=True)
                    st.success(f"Clashes prioritized by GNN confidence")
                
                # Save to session state
                st.session_state.clashes = clashes
                
                # Get statistics
                stats = detector.get_statistics()
                
                st.success(f"Detected {stats['total_clashes']} clashes")
                
                # Display statistics
                st.subheader("Clash Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Clashes", stats['total_clashes'])
                with col2:
                    st.metric("Critical", stats['by_severity'].get('critical', 0))
                with col3:
                    st.metric("High", stats['by_severity'].get('high', 0))
                with col4:
                    st.metric("Elements Affected", stats['elements_affected'])
                
                logger.info(f"Detected {stats['total_clashes']} clashes")
                
            except Exception as e:
                st.error(f"Error detecting clashes: {str(e)}")
                logger.error(f"Error detecting clashes: {str(e)}")
    
    # Display detected clashes
    if st.session_state.clashes:
        st.markdown("---")
        st.subheader("Detected Clashes")
        
        # Create DataFrame
        clashes_data = []
        for clash in st.session_state.clashes:
            row = {
                'Severity': clash.severity,
                'Type': clash.clash_type,
                'Element 1': f"{clash.element1_type} ({clash.element1_id})",
                'Element 2': f"{clash.element2_type} ({clash.element2_id})",
                'Distance (m)': f"{clash.distance:.3f}",
                'Overlap (m³)': f"{clash.overlap_volume:.4f}",
                'Description': clash.description
            }
            
            # Add GNN score if available
            if hasattr(clash, 'gnn_score'):
                row['GNN Confidence'] = f"{clash.gnn_score*100:.1f}%"
            
            clashes_data.append(row)
        
        df_clashes = pd.DataFrame(clashes_data)
        
        # Filter by severity
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=['critical', 'high', 'medium', 'low'],
            default=['critical', 'high', 'medium', 'low']
        )
        
        df_filtered = df_clashes[df_clashes['Severity'].isin(severity_filter)]
        
        # Color code severity
        def highlight_severity(row):
            colors = {
                'critical': 'background-color: #ffcccc',
                'high': 'background-color: #ffe6cc',
                'medium': 'background-color: #ffffcc',
                'low': 'background-color: #e6f7ff'
            }
            return [colors.get(row['Severity'], '')] * len(row)
        
        st.dataframe(
            df_filtered.style.apply(highlight_severity, axis=1),
            width="stretch",
            height=400
        )


def optimization_page():
    """Optimization suggestions page"""
    st.title("Optimization Suggestions")
    
    if not st.session_state.clashes:
        st.warning("Please run clash detection first!")
        return
    
    st.markdown("""
    Get AI-powered suggestions to resolve detected clashes. Each suggestion includes
    estimated cost, feasibility score, and detailed implementation steps.
    """)
    
    # Generate suggestions button
    if st.button("Generate Suggestions", type="primary"):
        with st.spinner("Generating optimization suggestions..."):
            try:
                # Initialize optimization engine
                optimizer = OptimizationEngine(
                    st.session_state.elements,
                    st.session_state.clashes,
                    config
                )
                
                # Generate suggestions
                suggestions = optimizer.generate_suggestions()
                
                # Save to session state
                st.session_state.suggestions = suggestions
                
                st.success(f"Generated {len(suggestions)} optimization suggestions")
                
                logger.info(f"Generated {len(suggestions)} suggestions")
                
            except Exception as e:
                st.error(f"Error generating suggestions: {str(e)}")
                logger.error(f"Error generating suggestions: {str(e)}")
    
    # Display suggestions
    if st.session_state.suggestions:
        st.markdown("---")
        st.subheader("Optimization Suggestions")
        
        # Filter by priority
        priority_filter = st.multiselect(
            "Filter by Priority",
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3],
            format_func=lambda x: f"Priority {x}" + (" (Highest)" if x == 1 else " (Lowest)" if x == 5 else "")
        )
        
        filtered_suggestions = [
            s for s in st.session_state.suggestions
            if s.priority in priority_filter
        ]
        
        # Display each suggestion
        for i, suggestion in enumerate(filtered_suggestions):
            with st.expander(
                f"{suggestion.suggestion_type.upper()} - Priority {suggestion.priority}: {suggestion.description}",
                expanded=(i < 3)  # Expand first 3
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Estimated Cost", f"${suggestion.estimated_cost:.2f}")
                with col2:
                    st.metric("Feasibility", f"{suggestion.feasibility_score*100:.0f}%")
                with col3:
                    st.metric("Type", suggestion.suggestion_type)
                
                st.markdown("**Implementation Steps:**")
                for step in suggestion.detailed_steps:
                    st.markdown(f"- {step}")
                
                st.markdown("**Modification Details:**")
                st.json(suggestion.modification)


def visualization_page():
    """3D visualization page"""
    st.title("3D Visualization")
    
    if not st.session_state.elements:
        st.warning("Please upload and parse an IFC file first!")
        return
    
    st.markdown("""
    Interactive 3D visualization of the BIM model with clash highlighting.
    """)
    
    try:
        # Initialize visualizer
        visualizer = BIMVisualizer(config)
        
        # Display options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_clashes = st.checkbox("Show Clashes", value=True)
        with col2:
            show_all_elements = st.checkbox("Show All Elements", value=True)
        with col3:
            show_connections = st.checkbox("Show Clash Connections", value=False, 
                                          help="Draw lines from clash points to element centers")
        
        # Create 3D model view
        with st.spinner("Rendering 3D model..."):
            fig = visualizer.create_3d_model_view(
                elements=st.session_state.elements,
                clashes=st.session_state.clashes if show_clashes else None,
                show_clash_connections=show_connections
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional charts
        if st.session_state.clashes:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Clash Distribution")
                fig_clashes = visualizer.create_clash_distribution_chart(st.session_state.clashes)
                st.plotly_chart(fig_clashes, use_container_width=True)
            
            with col2:
                st.subheader("Element Types")
                fig_elements = visualizer.create_element_type_chart(st.session_state.elements)
                st.plotly_chart(fig_elements, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        logger.error(f"Error in visualization: {str(e)}")


def reports_page():
    """Reports and export page"""
    st.title("Reports & Export")
    
    if not st.session_state.elements:
        st.warning("Please upload and parse an IFC file first!")
        return
    
    st.markdown("""
    Generate comprehensive reports and export data in various formats.
    """)
    
    # Summary report
    st.subheader("Summary Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Elements", len(st.session_state.elements))
    with col2:
        st.metric("Total Clashes", len(st.session_state.clashes))
    with col3:
        st.metric("Optimization Suggestions", len(st.session_state.suggestions))
    
    # Export options
    st.markdown("---")
    st.subheader("Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.clashes:
            # Export clashes to CSV
            clashes_data = []
            for clash in st.session_state.clashes:
                clashes_data.append({
                    'Severity': clash.severity,
                    'Type': clash.clash_type,
                    'Element_1_GUID': clash.element1_guid,
                    'Element_2_GUID': clash.element2_guid,
                    'Element_1_Type': clash.element1_type,
                    'Element_2_Type': clash.element2_type,
                    'Distance_m': clash.distance,
                    'Overlap_Volume_m3': clash.overlap_volume,
                    'Description': clash.description
                })
            
            df_export = pd.DataFrame(clashes_data)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="Download Clashes (CSV)",
                data=csv,
                file_name="bim_clashes.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.session_state.suggestions:
            # Export suggestions to CSV
            suggestions_data = []
            for sugg in st.session_state.suggestions:
                suggestions_data.append({
                    'Priority': sugg.priority,
                    'Type': sugg.suggestion_type,
                    'Description': sugg.description,
                    'Element_to_Modify': sugg.element_to_modify,
                    'Estimated_Cost': sugg.estimated_cost,
                    'Feasibility_Score': sugg.feasibility_score
                })
            
            df_export = pd.DataFrame(suggestions_data)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="Download Suggestions (CSV)",
                data=csv,
                file_name="optimization_suggestions.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
