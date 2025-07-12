"""
UI Components and Utilities
===========================

Reusable UI components and utilities for the Streamlit application.

Author: AI Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union
import base64
import io
from PIL import Image


class UIComponents:
    """Collection of reusable UI components for the Streamlit app."""
    
    def __init__(self):
        """Initialize UI components."""
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#ff9800',
            'error': '#d62728',
            'info': '#17becf'
        }
    
    def create_metric_card(self, title: str, value: Union[str, float], 
                          delta: Optional[Union[str, float]] = None,
                          color: str = 'primary') -> None:
        """
        Create a styled metric card.
        
        Args:
            title: Metric title
            value: Metric value
            delta: Change from previous value
            color: Color theme
        """
        color_hex = self.color_palette.get(color, self.color_palette['primary'])
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color_hex}22, {color_hex}11);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid {color_hex};
            margin: 0.5rem 0;
        ">
            <h3 style="color: {color_hex}; margin: 0; font-size: 1.2rem;">{title}</h3>
            <h2 style="margin: 0.25rem 0; color: #333;">{value}</h2>
            {f'<p style="margin: 0; color: #666; font-size: 0.9rem;">{delta}</p>' if delta else ''}
        </div>
        """, unsafe_allow_html=True)
    
    def create_progress_ring(self, value: float, max_value: float = 100, 
                            color: str = 'primary', size: int = 120) -> None:
        """
        Create a circular progress indicator.
        
        Args:
            value: Current value
            max_value: Maximum value
            color: Color theme
            size: Size of the progress ring
        """
        percentage = (value / max_value) * 100
        color_hex = self.color_palette.get(color, self.color_palette['primary'])
        
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 1rem 0;">
            <div style="
                width: {size}px;
                height: {size}px;
                border-radius: 50%;
                background: conic-gradient(
                    {color_hex} {percentage * 3.6}deg,
                    #e0e0e0 {percentage * 3.6}deg
                );
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <div style="
                    width: {size - 20}px;
                    height: {size - 20}px;
                    border-radius: 50%;
                    background: white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                ">
                    <span style="font-size: 1.5rem; font-weight: bold; color: {color_hex};">
                        {percentage:.1f}%
                    </span>
                    <span style="font-size: 0.8rem; color: #666;">
                        {value:.1f}/{max_value}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def create_status_badge(self, status: str, color: Optional[str] = None) -> str:
        """
        Create a status badge HTML.
        
        Args:
            status: Status text
            color: Badge color (auto-determined if None)
            
        Returns:
            HTML string for status badge
        """
        if color is None:
            if status.lower() in ['success', 'passed', 'ok', 'good']:
                color = 'success'
            elif status.lower() in ['warning', 'caution', 'medium']:
                color = 'warning'
            elif status.lower() in ['error', 'failed', 'bad', 'critical']:
                color = 'error'
            else:
                color = 'info'
        
        color_hex = self.color_palette.get(color, self.color_palette['info'])
        
        return f"""
        <span style="
            background: {color_hex};
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        ">
            {status}
        </span>
        """
    
    def create_comparison_chart(self, data: Dict[str, Dict[str, float]], 
                               title: str = "Model Comparison") -> go.Figure:
        """
        Create a radar chart for model comparison.
        
        Args:
            data: Dictionary of model data
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Get all metrics
        all_metrics = set()
        for model_data in data.values():
            all_metrics.update(model_data.keys())
        
        metrics = sorted(all_metrics)
        
        # Add trace for each model
        for model_name, model_data in data.items():
            values = [model_data.get(metric, 0) for metric in metrics]
            values += [values[0]]  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title,
            font=dict(size=12)
        )
        
        return fig
    
    def create_performance_timeline(self, performance_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create a timeline chart showing model performance over time.
        
        Args:
            performance_data: List of performance data points
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score'),
            vertical_spacing=0.1
        )
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, (row, col) in zip(metrics, positions):
            x_values = [d.get('timestamp', i) for i, d in enumerate(performance_data)]
            y_values = [d.get(metric, 0) for d in performance_data]
            
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines+markers',
                    name=metric.capitalize(),
                    line=dict(width=2)
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Model Performance Timeline",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_confusion_matrix_heatmap(self, confusion_matrix: np.ndarray, 
                                       labels: List[str] = None) -> go.Figure:
        """
        Create a confusion matrix heatmap.
        
        Args:
            confusion_matrix: Confusion matrix as numpy array
            labels: Class labels
            
        Returns:
            Plotly figure
        """
        if labels is None:
            labels = ['No Hotspot', 'Hotspot']
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 20},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=400,
            height=400
        )
        
        return fig
    
    def create_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, 
                        auc: float, model_name: str = "Model") -> go.Figure:
        """
        Create ROC curve visualization.
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            auc: Area under curve
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc:.3f})',
            line=dict(color=self.color_palette['primary'], width=3)
        ))
        
        # Diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=500,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_image_gallery(self, images: List[Image.Image], 
                           titles: List[str] = None,
                           captions: List[str] = None,
                           cols: int = 3) -> None:
        """
        Create an image gallery with titles and captions.
        
        Args:
            images: List of PIL Images
            titles: List of image titles
            captions: List of image captions
            cols: Number of columns in the gallery
        """
        if titles is None:
            titles = [f"Image {i+1}" for i in range(len(images))]
        
        if captions is None:
            captions = [""] * len(images)
        
        # Create columns
        columns = st.columns(cols)
        
        for i, (image, title, caption) in enumerate(zip(images, titles, captions)):
            col_idx = i % cols
            
            with columns[col_idx]:
                st.markdown(f"**{title}**")
                st.image(image, use_container_width=True)
                if caption:
                    st.caption(caption)
    
    def create_download_button(self, data: Union[str, bytes], 
                              filename: str, mime_type: str,
                              button_text: str = "Download") -> None:
        """
        Create a styled download button.
        
        Args:
            data: Data to download
            filename: Name of the downloaded file
            mime_type: MIME type of the file
            button_text: Text to display on button
        """
        st.download_button(
            label=f"📥 {button_text}",
            data=data,
            file_name=filename,
            mime=mime_type,
            use_container_width=True
        )
    
    def create_info_panel(self, title: str, content: Dict[str, Any],
                         panel_type: str = 'info') -> None:
        """
        Create an information panel with key-value pairs.
        
        Args:
            title: Panel title
            content: Dictionary of key-value pairs
            panel_type: Type of panel ('info', 'success', 'warning', 'error')
        """
        color = self.color_palette.get(panel_type, self.color_palette['info'])
        
        st.markdown(f"""
        <div style="
            border: 1px solid {color};
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            background: {color}11;
        ">
            <h4 style="color: {color}; margin-top: 0;">{title}</h4>
        """, unsafe_allow_html=True)
        
        for key, value in content.items():
            st.markdown(f"**{key}:** {value}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def create_loading_spinner(self, text: str = "Processing...") -> None:
        """
        Create a custom loading spinner.
        
        Args:
            text: Loading text to display
        """
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            flex-direction: column;
        ">
            <div style="
                border: 4px solid #f3f3f3;
                border-top: 4px solid {self.color_palette['primary']};
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            "></div>
            <p style="margin-top: 1rem; color: #666;">{text}</p>
        </div>
        
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def create_alert_box(self, message: str, alert_type: str = 'info',
                        dismissible: bool = False) -> None:
        """
        Create a styled alert box.
        
        Args:
            message: Alert message
            alert_type: Type of alert ('info', 'success', 'warning', 'error')
            dismissible: Whether the alert can be dismissed
        """
        icons = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }
        
        color = self.color_palette.get(alert_type, self.color_palette['info'])
        icon = icons.get(alert_type, 'ℹ️')
        
        st.markdown(f"""
        <div style="
            background: {color}22;
            border: 1px solid {color};
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            display: flex;
            align-items: center;
        ">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
            <span style="color: #333;">{message}</span>
        </div>
        """, unsafe_allow_html=True)
    
    def create_feature_highlight(self, features: List[Dict[str, str]]) -> None:
        """
        Create a feature highlight section.
        
        Args:
            features: List of feature dictionaries with 'icon', 'title', 'description'
        """
        cols = st.columns(len(features))
        
        for i, feature in enumerate(features):
            with cols[i]:
                st.markdown(f"""
                <div style="
                    text-align: center;
                    padding: 1rem;
                    border-radius: 10px;
                    background: linear-gradient(135deg, #667eea22, #764ba222);
                    margin: 0.5rem 0;
                ">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">
                        {feature.get('icon', '🔧')}
                    </div>
                    <h4 style="color: #333; margin: 0.5rem 0;">
                        {feature.get('title', 'Feature')}
                    </h4>
                    <p style="color: #666; font-size: 0.9rem; margin: 0;">
                        {feature.get('description', 'Feature description')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
