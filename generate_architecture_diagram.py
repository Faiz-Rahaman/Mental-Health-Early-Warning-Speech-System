#!/usr/bin/env python3
"""
MH-EWSS Architecture Diagram Generator
Creates a professional architecture diagram for the Mental Health Early Warning Speech System
"""

from graphviz import Digraph
import os

def create_architecture_diagram():
    """Generate the MH-EWSS architecture diagram using Graphviz"""

    # Create a new directed graph
    dot = Digraph('mh_ewss_architecture', comment='MH-EWSS System Architecture')
    dot.attr(rankdir='LR', size='14,8', dpi='300')  # PPT-friendly size with high DPI for crisp text
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='14')
    dot.attr('edge', fontname='Arial', fontsize='12')
    dot.attr(bgcolor='white', pad='0.5', ranksep='1.0', nodesep='0.8')  # Add padding and spacing for PPT

    # Define colors for different component types
    colors = {
        'input': '#E3F2FD',      # Light blue
        'processing': '#FFF3E0', # Light orange
        'storage': '#F3E5F5',    # Light purple
        'ai': '#E8F5E8',         # Light green
        'ml': '#FFF9C4',         # Light yellow
        'output': '#FFEBEE'      # Light red
    }

    # Outer container for PPT slide format
    with dot.subgraph(name='cluster_architecture') as arch_container:
        arch_container.attr(label='MH-EWSS System Architecture (Hybrid Neurosymbolic AI)', style='bold', fontsize='16', fontcolor='#1976D2')
        arch_container.attr(bgcolor='#F8F9FA', color='#1976D2', penwidth='2')

        # Input Layer
        with arch_container.subgraph(name='cluster_input') as input_cluster:
            input_cluster.attr(label='Input Layer', style='filled', color='#BBDEFB')
            input_cluster.node('crema_dataset', 'CREMA-D Dataset\n(7,441 AudioWAV files)', fillcolor=colors['input'])
            input_cluster.node('user_audio', 'User Audio Input\n(.wav upload)', fillcolor=colors['input'])
            input_cluster.node('audio_processing', 'Audio Preprocessing\n• Load & Normalize\n• Noise Reduction\n• Voice Detection\n• Silence Trimming', fillcolor=colors['processing'])

        # Feature Extraction Layer
        with arch_container.subgraph(name='cluster_features') as feature_cluster:
            feature_cluster.attr(label='Feature Extraction Layer (LangGraph Nodes)', style='filled', color='#FFCC80')
            feature_cluster.node('prosody_agent', 'Prosody Agent\n(How you speak)\n• Pitch Mean & Std\n• Energy Mean & Std\n• Speech Rate\n• Pause Ratio\n• HNR\n• Zero Crossing Rate\n• Spectral Centroid\n• Spectral Rolloff', fillcolor=colors['processing'])
            feature_cluster.node('language_agent', 'Language Agent\n(What you say)\n• Speech-to-Text (Google API)\n• Sentiment Polarity\n• Sentiment Subjectivity\n• Vocabulary Richness\n• Word Count\n• Sentence Complexity', fillcolor=colors['processing'])
            feature_cluster.node('temporal_agent', 'Temporal Agent\n(Changes over time)\n• Store Features in DB\n• Calculate Personal Baseline\n• Track Trends', fillcolor=colors['processing'])

        # Storage Layer
        with arch_container.subgraph(name='cluster_storage') as storage_cluster:
            storage_cluster.attr(label='Storage Layer', style='filled', color='#CE93D8')
            storage_cluster.node('database', 'SQLite Database\n• Feature Records (16 cols)\n• Personal Baselines\n• Historical Data', fillcolor=colors['storage'])

        # ML Model Layer
        with arch_container.subgraph(name='cluster_ml') as ml_cluster:
            ml_cluster.attr(label='ML Model Layer', style='filled', color='#FFF176')
            ml_cluster.node('ml_specialist', 'ML Specialist Agent\n(Random Forest Classifier)\n• Trained on 7,441 CREMA-D samples\n• Acoustic Risk Probability\n• Binary: Low Risk / High Risk', fillcolor=colors['ml'])
            ml_cluster.node('train_script', 'train_ml_model.py\n• scikit-learn\n• 70% Accuracy\n• Saved as .pkl', fillcolor=colors['ml'], style='rounded,filled,dashed')

        # AI Decision Layer
        with arch_container.subgraph(name='cluster_ai') as ai_cluster:
            ai_cluster.attr(label='AI Decision Layer (LLM)', style='filled', color='#81C784')
            ai_cluster.node('llm_agent', 'LLM Decision Agent\n(Google Gemini API)\n• Receives ML Risk Score\n• Receives Baseline Deviation\n• Risk Assessment\n• Natural Language Explanation\n• Recommendations', fillcolor=colors['ai'])

        # Output Layer
        with arch_container.subgraph(name='cluster_output') as output_cluster:
            output_cluster.attr(label='Output Layer', style='filled', color='#EF5350')
            output_cluster.node('dashboard', 'Streamlit Dashboard\n• Risk Level Display\n• Plotly Radar Chart\n• Voice Diagnostics Metrics\n• AI Recommendations\n• Feature List', fillcolor=colors['output'])

        # Workflow Orchestration
        arch_container.node('langgraph', 'LangGraph Workflow\nOrchestrator', shape='diamond', fillcolor='#FFF9C4', color='#F57C00')

        # Define edges (data flow)
        # Input flow
        arch_container.edge('crema_dataset', 'audio_processing', label='Training Data')
        arch_container.edge('crema_dataset', 'train_script', label='Train ML Model', style='dashed', color='#F57C00')
        arch_container.edge('user_audio', 'audio_processing', label='Live Audio')

        # Processing flow
        arch_container.edge('audio_processing', 'prosody_agent', label='10 Features')
        arch_container.edge('audio_processing', 'language_agent', label='6 Features')

        # Feature flow to temporal agent
        arch_container.edge('prosody_agent', 'temporal_agent', label='Prosody Features')
        arch_container.edge('language_agent', 'temporal_agent', label='Language Features')

        # Temporal to database
        arch_container.edge('temporal_agent', 'database', label='Store & Retrieve\nBaseline')

        # Database to ML specialist
        arch_container.edge('temporal_agent', 'ml_specialist', label='Current Features')
        arch_container.edge('train_script', 'ml_specialist', label='Trained Model (.pkl)', style='dashed', color='#F57C00')

        # ML Specialist to LLM
        arch_container.edge('ml_specialist', 'llm_agent', label='Acoustic Risk\nProbability %')
        arch_container.edge('database', 'llm_agent', label='Baseline Data')

        # All agents connect to LangGraph
        arch_container.edge('prosody_agent', 'langgraph', style='dashed', color='#666666')
        arch_container.edge('language_agent', 'langgraph', style='dashed', color='#666666')
        arch_container.edge('temporal_agent', 'langgraph', style='dashed', color='#666666')
        arch_container.edge('ml_specialist', 'langgraph', style='dashed', color='#666666')
        arch_container.edge('llm_agent', 'langgraph', style='dashed', color='#666666')

        # LangGraph to final output
        arch_container.edge('langgraph', 'dashboard', label='Analysis Results')

        # Direct flow from LLM to dashboard
        arch_container.edge('llm_agent', 'dashboard', label='Risk Assessment\n+ Explanation')

    return dot

def main():
    """Generate and save the architecture diagram"""
    print("Generating MH-EWSS Architecture Diagram...")

    # Create the diagram
    diagram = create_architecture_diagram()

    # Ensure output directory exists
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Save in multiple formats
    output_path = os.path.join(output_dir, 'mh_ewss_architecture')

    # Generate PNG
    diagram.format = 'png'
    diagram.render(output_path, view=False, cleanup=True)

    # Generate PDF
    diagram.format = 'pdf'
    diagram.render(output_path, view=False, cleanup=True)

    # Generate SVG
    diagram.format = 'svg'
    diagram.render(output_path, view=False, cleanup=True)

    print("✅ Architecture diagram generated successfully!")
    print(f"📁 Files saved in: {output_dir}/")
    print("   - mh_ewss_architecture.png")
    print("   - mh_ewss_architecture.pdf")
    print("   - mh_ewss_architecture.svg")
    print("   - mh_ewss_architecture.gv (source)")

    # Display the source code for reference
    print("\n📄 Graphviz source code:")
    print(diagram.source)

if __name__ == "__main__":
    main()