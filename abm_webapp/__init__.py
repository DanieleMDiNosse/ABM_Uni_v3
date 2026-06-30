"""
Single-user web application for running ABM_Uni_v3 simulations with live updates.

This package is intentionally lightweight and keeps dependencies minimal:
- The UI uses Dash (Plotly-native).
- Live metrics are stored in a local SQLite database (stdlib `sqlite3`).
- The simulation runs in a separate process (stdlib `multiprocessing`) to keep the UI responsive.
"""

__version__ = "0.2.0"
