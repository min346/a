import streamlit as st
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==========================================
# 1. SETUP & CLASSES
# ==========================================

# --- 🚨 IMPORTANT: PASTE YOUR ParkingGrid CLASS BELOW 🚨 ---
# (Paste the entire 'class ParkingGrid' code from your notebook here)
# For this example to run, I will assume you have imported it or pasted it.
# If you have it in a separate file named 'parking_env.py', you can do:
# from parking_env import ParkingGrid 

# Placeholder for the user to fill:
# class ParkingGrid:
#     ... (Your code) ...


# Define the Env Builders (Same as in your notebook)
# Make sure to uncomment and fill in your actual grid parameters
env_builders = {
    'easy':   lambda: ParkingGrid(10, 10, obstacles_num=5),
    'medium': lambda: ParkingGrid(15, 15, obstacles_num=15),
    'hard':   lambda: ParkingGrid(30, 30, obstacles_num=50) # Example params
}

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_resource
def load_models():
    filename = "parking_models.pkl"
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data["q_tables"], data["dq_tables"]
    except FileNotFoundError:
        return None, None

q_tables, dq_tables = load_models()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_greedy_action(env, table, state):
    # If state not in table, random action
    if state not in table:
        return env.action_space.sample()
    return max(table[state], key=table[state].get)

def render_grid(env, path):
    """
    Draws the grid using Matplotlib to display in the web app
    """
    rows, cols = env.rows, env.cols
    grid = env.grid
    
    # Create a custom colormap: White (0), Gray (1), Green (Goal)
    # We map generic grid values to colors
    # 0 = Road, 1 = Obstacle
    cmap = mcolors.ListedColormap(['white', '#424242'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, norm=norm)

    # Draw Grid Lines
    ax.grid(which='major', axis='both', linestyle='-', color='#E0E0E0', linewidth=1)
    ax.set_xticks(np.arange(-.5, cols, 1));
    ax.set_yticks(np.arange(-.5, rows, 1));
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Draw Goal
    gy, gx = env.goal_pos
    ax.add_patch(plt.Rectangle((gx-0.5, gy-0.5), 1, 1, color='#66BB6A')) # Green Goal

    # Draw Path
    if len(path) > 1:
        path_y, path_x = zip(*path)
        ax.plot(path_x, path_y, color='#FFCC80', linewidth=2, alpha=0.7)

    # Draw Agent (Car)
    ay, ax_pos = env.agent_pos
    ax.add_patch(plt.Circle((ax_pos, ay), 0.3, color='#2196F3')) # Blue Dot

    return fig

# ==========================================
# 4. STREAMLIT APP LAYOUT
# ==========================================
st.set_page_config(page_title="RL Parking Simulator", page_icon="🚗", layout="wide")

st.title("🚗 Autonomous Parking Simulator")
st.markdown("Comparing **Q-Learning** vs **Double Q-Learning** on the Web.")

# Sidebar Controls
st.sidebar.header("⚙️ Settings")

if q_tables is None:
    st.error("❌ 'parking_models.pkl' not found! Please run your training notebook to save the models first.")
    st.stop()

selected_level = st.sidebar.selectbox("Select Level", ["easy", "medium", "hard"])
selected_agent = st.sidebar.selectbox("Select Agent", ["Q-Learning", "Double-Q"])
speed = st.sidebar.slider("Simulation Speed (Delay)", 0.01, 1.0, 0.1)

# Initialize Session State for the Environment
if 'env' not in st.session_state:
    st.session_state.env = None

# Buttons
col1, col2 = st.sidebar.columns(2)
if col1.button("🔄 New Episode"):
    st.session_state.env = env_builders[selected_level]()
    st.session_state.env.reset() # Random reset
    st.session_state.path = [st.session_state.env.agent_pos]
    st.session_state.done = False
    st.session_state.msg = ""

run_simulation = st.sidebar.button("▶ Run Drive")

# ==========================================
# 5. MAIN VISUALIZATION
# ==========================================

# Placeholder for the Map
map_placeholder = st.empty()
info_placeholder = st.empty()

# Use existing env or create new one if none exists
if st.session_state.env is None:
    st.session_state.env = env_builders[selected_level]()
    st.session_state.env.reset()
    st.session_state.path = [st.session_state.env.agent_pos]
    st.session_state.done = False

env = st.session_state.env

# Select the correct brain
if selected_agent == "Q-Learning":
    current_table = q_tables[selected_level]
else:
    current_table = dq_tables[selected_level]

# Logic to run the loop
if run_simulation and not st.session_state.done:
    
    # Run loop until done
    for _ in range(500): # Max steps safety
        # 1. Action
        action = get_greedy_action(env, current_table, env.agent_pos)
        
        # 2. Step
        state, reward, done, info = env.step(action)
        st.session_state.path.append(state)
        
        # 3. Render
        fig = render_grid(env, st.session_state.path)
        map_placeholder.pyplot(fig)
        plt.close(fig) # Clean up memory
        
        # 4. Info Update
        status = "CRASHED! 💥" if info.get('is_collision') else "DRIVING..."
        if info.get('is_success'): status = "PARKED! 🅿️"
        
        info_placeholder.info(f"**Status:** {status} | **Steps:** {len(st.session_state.path)} | **Pos:** {state}")
        
        if done:
            st.session_state.done = True
            if info.get('is_success'):
                st.success(f"Successfully Parked in {len(st.session_state.path)} steps!")
            else:
                st.error("Failed: Collision or Out of Bounds.")
            break
            
        time.sleep(speed)

else:
    # Just show static current state if not running
    fig = render_grid(env, st.session_state.path)
    map_placeholder.pyplot(fig)
    plt.close(fig)
