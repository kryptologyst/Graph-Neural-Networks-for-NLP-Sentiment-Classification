"""Streamlit demo for Graph Neural Networks for NLP sentiment classification."""

import streamlit as st
import torch
import spacy
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.device import get_device
from src.data.processing import DependencyGraphBuilder
from src.models.gnn_models import create_model, get_model_info


# Page configuration
st.set_page_config(
    page_title="GNN for NLP - Sentiment Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-positive {
        color: #28a745;
        font-weight: bold;
    }
    .prediction-negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_spacy_model():
    """Load spaCy model with caching."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
        return None


@st.cache_resource
def load_model():
    """Load pre-trained model with caching."""
    try:
        device = get_device("cpu")  # Use CPU for demo
        model = create_model(
            model_name="gcn",
            input_dim=300,
            hidden_dim=64,
            output_dim=2,
            num_layers=2,
            dropout=0.5,
            activation="relu",
            pooling="mean",
            use_residual=True,
            use_batch_norm=True
        )
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def create_dependency_graph(sentence: str, nlp_model) -> Tuple[nx.DiGraph, List[str], List[Tuple[int, int, str]]]:
    """Create a NetworkX graph from sentence dependencies."""
    doc = nlp_model(sentence)
    
    G = nx.DiGraph()
    tokens = []
    dependencies = []
    
    # Add nodes (tokens)
    for i, token in enumerate(doc):
        if not token.is_space:
            G.add_node(i, text=token.text, pos=token.pos_, dep=token.dep_)
            tokens.append(token.text)
    
    # Add edges (dependencies)
    for token in doc:
        if not token.is_space and token.i != token.head.i:
            src = token.i
            dst = token.head.i
            dep_type = token.dep_
            
            if src in G.nodes and dst in G.nodes:
                G.add_edge(src, dst, label=dep_type)
                dependencies.append((src, dst, dep_type))
    
    return G, tokens, dependencies


def visualize_dependency_graph(G: nx.DiGraph, tokens: List[str], title: str = "Dependency Graph"):
    """Create an interactive visualization of the dependency graph."""
    # Get layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Extract edge information
    edge_x = []
    edge_y = []
    edge_labels = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Get edge label
        label = G[edge[0]][edge[1]].get('label', '')
        edge_labels.append(label)
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Extract node information
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(tokens[node])
        
        # Create hover text
        hover_text = f"Token: {tokens[node]}<br>POS: {G.nodes[node].get('pos', '')}<br>Dep: {G.nodes[node].get('dep', '')}"
        node_hover.append(hover_text)
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        hovertext=node_hover,
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            reversescale=True,
            color=[],
            size=30,
            colorbar=dict(
                thickness=15,
                xanchor="left",
                len=0.5
            ),
            line=dict(width=2, color='black')
        )
    )
    
    # Color nodes by degree
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    
    node_trace.marker.color = node_adjacencies
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Node size and color indicate degree centrality",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    
    return fig


def predict_sentiment(sentence: str, model, device, graph_builder) -> Tuple[int, float, torch.Tensor]:
    """Predict sentiment for a sentence."""
    try:
        # Create graph
        graph_data, _, _ = graph_builder.sentence_to_graph(sentence)
        graph_data = graph_data.to(device)
        
        # Make prediction
        with torch.no_grad():
            logits = model(graph_data.x, graph_data.edge_index)
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][prediction].item()
        
        return prediction, confidence, probabilities[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return 0, 0.0, torch.tensor([0.5, 0.5])


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">Graph Neural Networks for NLP</h1>', unsafe_allow_html=True)
    st.markdown("### Sentiment Classification using Dependency Parsing")
    
    # Load models
    nlp_model = load_spacy_model()
    model, device = load_model()
    
    if nlp_model is None or model is None:
        st.error("Failed to load required models. Please check the installation.")
        return
    
    # Create graph builder
    graph_builder = DependencyGraphBuilder()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["GCN", "GraphSAGE", "GAT", "GIN"],
        index=0
    )
    
    # Example sentences
    st.sidebar.header("Example Sentences")
    example_sentences = [
        "I love this amazing product!",
        "This is absolutely terrible.",
        "What a wonderful experience.",
        "I hate this completely.",
        "Outstanding quality and service.",
        "This is disappointing.",
        "Perfect solution for my needs.",
        "Worst purchase ever."
    ]
    
    selected_example = st.sidebar.selectbox("Choose an example:", example_sentences)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Input Sentence")
        
        # Text input
        sentence = st.text_area(
            "Enter a sentence for sentiment analysis:",
            value=selected_example,
            height=100,
            help="Enter any sentence to analyze its sentiment using dependency parsing and GNNs."
        )
        
        if st.button("Analyze Sentiment", type="primary"):
            if sentence.strip():
                # Make prediction
                prediction, confidence, probabilities = predict_sentiment(
                    sentence, model, device, graph_builder
                )
                
                # Display prediction
                st.subheader("Prediction Results")
                
                col_pred, col_conf = st.columns(2)
                
                with col_pred:
                    if prediction == 1:
                        st.markdown('<p class="prediction-positive">Sentiment: Positive</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="prediction-negative">Sentiment: Negative</p>', unsafe_allow_html=True)
                
                with col_conf:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Probability distribution
                prob_df = pd.DataFrame({
                    'Sentiment': ['Negative', 'Positive'],
                    'Probability': probabilities.numpy()
                })
                
                fig_prob = px.bar(
                    prob_df, 
                    x='Sentiment', 
                    y='Probability',
                    color='Sentiment',
                    color_discrete_map={'Negative': '#dc3545', 'Positive': '#28a745'},
                    title="Sentiment Probability Distribution"
                )
                fig_prob.update_layout(showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Dependency graph visualization
                st.subheader("Dependency Graph")
                
                try:
                    G, tokens, dependencies = create_dependency_graph(sentence, nlp_model)
                    
                    if len(G.nodes()) > 0:
                        fig_graph = visualize_dependency_graph(G, tokens, f"Dependency Graph: '{sentence}'")
                        st.plotly_chart(fig_graph, use_container_width=True)
                        
                        # Show dependency details
                        with st.expander("Dependency Details"):
                            dep_df = pd.DataFrame(dependencies, columns=['Source', 'Target', 'Dependency Type'])
                            st.dataframe(dep_df, use_container_width=True)
                    else:
                        st.warning("Could not parse the sentence into a dependency graph.")
                        
                except Exception as e:
                    st.error(f"Error creating dependency graph: {e}")
            else:
                st.warning("Please enter a sentence to analyze.")
    
    with col2:
        st.header("Model Information")
        
        # Model details
        model_info = get_model_info(model)
        
        st.markdown("""
        <div class="metric-card">
            <h4>Model Architecture</h4>
            <p><strong>Type:</strong> Graph Convolutional Network (GCN)</p>
            <p><strong>Layers:</strong> 2</p>
            <p><strong>Hidden Dim:</strong> 64</p>
            <p><strong>Parameters:</strong> {:,}</p>
        </div>
        """.format(model_info['total_parameters']), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>How It Works</h4>
            <ol>
                <li>Parse sentence into dependency tree</li>
                <li>Create graph with words as nodes</li>
                <li>Apply GCN layers for message passing</li>
                <li>Pool node features globally</li>
                <li>Classify sentiment</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset statistics
        st.subheader("Dataset Statistics")
        st.markdown("""
        - **Total Samples:** 1,000
        - **Classes:** 2 (Positive/Negative)
        - **Avg Nodes per Graph:** ~8
        - **Avg Edges per Graph:** ~12
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>Graph Neural Networks for NLP - Sentiment Classification Demo</p>
        <p>Built with PyTorch Geometric, spaCy, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
