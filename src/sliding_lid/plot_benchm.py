import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Enable interactive plotting
plt.ion()

# Try to use plotly for interactive HTML exports
try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

PLOT_DIR = Path("plots/benchmarks")
PLOT_DIR.mkdir(exist_ok=True)

# Benchmark data
grid_sizes = np.array([1000, 3000, 5000, 8000, 10000, 15000, 18000])

# H100 benchmark data
h100_mlups = np.array([786.34, 1279.32, 1341.63, 1356.63, 1361.91, 1339.69, 1320.82])
h100_blups = h100_mlups / 1000  # Convert MLUPS to BLUPS

# A100 benchmark data
a100_mlups = np.array([468.70, 913.97, 941.73, 949.62, 954.22, 931.93, 920.45])
a100_blups = a100_mlups / 1000  # Convert MLUPS to BLUPS

# Create BLUPS plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(grid_sizes, h100_blups, 'o-', color='tab:blue', label='H100', linewidth=2, markersize=6)
ax.plot(grid_sizes, a100_blups, 'o-', color='tab:red', label='A100', linewidth=2, markersize=6)
ax.set_title('BLUPS (Billion Lattice Updates Per Second)', fontsize=14)
ax.set_xlabel('Grid Size (N×N)')
ax.set_ylabel('BLUPS')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

plt.tight_layout()

# Save both static and interactive versions
plt.savefig(f"{PLOT_DIR}/blups_plot.png", dpi=300, bbox_inches='tight')

# Create interactive HTML version if plotly is available
if PLOTLY_AVAILABLE:
    fig_interactive = go.Figure()
    fig_interactive.add_trace(go.Scatter(
        x=grid_sizes, y=h100_blups, mode='lines+markers',
        name='H100', line=dict(color='blue', width=2),
        marker=dict(size=8)))
    fig_interactive.add_trace(go.Scatter(
        x=grid_sizes, y=a100_blups, mode='lines+markers',
        name='A100', line=dict(color='red', width=2),
        marker=dict(size=8)))
    fig_interactive.update_layout(
        title='BLUPS (Billion Lattice Updates Per Second)',
        xaxis_title='Grid Size (N×N)',
        yaxis_title='BLUPS',
        width=1000, height=600,
        showlegend=True,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
    pyo.plot(fig_interactive, filename=f"{PLOT_DIR}/blups_interactive.html", auto_open=False)
    
    # Also save as a standalone HTML that can be embedded in presentations
    fig_interactive.write_html(f"{PLOT_DIR}/blups_standalone.html", 
                              include_plotlyjs='inline',
                              config={'displayModeBar': True, 'displaylogo': False})
    
    print(f"Interactive BLUPS plot saved as: {PLOT_DIR}/blups_interactive.html")
    print(f"Standalone BLUPS plot saved as: {PLOT_DIR}/blups_standalone.html")
    print("For PowerPoint: Use the standalone HTML file or embed the interactive plot")

plt.show()

# Create MLUPS plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(grid_sizes, h100_mlups, 'o-', color='tab:blue', label='H100', linewidth=2, markersize=6)
ax.plot(grid_sizes, a100_mlups, 'o-', color='tab:red', label='A100', linewidth=2, markersize=6)
ax.set_title('MLUPS (Million Lattice Updates Per Second)', fontsize=14)
ax.set_xlabel('Grid Size (N×N)')
ax.set_ylabel('MLUPS')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

plt.tight_layout()

# Save both static and interactive versions
plt.savefig(f"{PLOT_DIR}/mlups_plot.png", dpi=300, bbox_inches='tight')

# Create interactive HTML version if plotly is available
if PLOTLY_AVAILABLE:
    fig_interactive = go.Figure()
    fig_interactive.add_trace(go.Scatter(
        x=grid_sizes, y=h100_mlups, mode='lines+markers',
        name='H100', line=dict(color='blue', width=2),
        marker=dict(size=8)))
    fig_interactive.add_trace(go.Scatter(
        x=grid_sizes, y=a100_mlups, mode='lines+markers',
        name='A100', line=dict(color='red', width=2),
        marker=dict(size=8)))
    fig_interactive.update_layout(
        title='MLUPS (Million Lattice Updates Per Second)',
        xaxis_title='Grid Size (N×N)',
        yaxis_title='MLUPS',
        width=1000, height=600,
        showlegend=True,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
    pyo.plot(fig_interactive, filename=f"{PLOT_DIR}/mlups_interactive.html", auto_open=False)
    
    # Also save as a standalone HTML that can be embedded in presentations
    fig_interactive.write_html(f"{PLOT_DIR}/mlups_standalone.html", 
                              include_plotlyjs='inline',
                              config={'displayModeBar': True, 'displaylogo': False})
    
    print(f"Interactive MLUPS plot saved as: {PLOT_DIR}/mlups_interactive.html")
    print(f"Standalone MLUPS plot saved as: {PLOT_DIR}/mlups_standalone.html")
    print("For PowerPoint: Use the standalone HTML file or embed the interactive plot")

plt.show()