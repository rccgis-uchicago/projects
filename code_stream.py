import streamlit as st
from pysheds.grid import Grid
import matplotlib.pyplot as plt
from matplotlib import colors
import rasterio
import numpy as np
from matplotlib import colors

#dem_path = "./data/wadi_84.tif"

# Define the custom HTML and CSS for the fixed sticky header
header = st.container()
header.title("Welcome to the Flow Visualizer")

header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

### Custom CSS for the sticky header
st.markdown(
    """
<style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 2.875rem;
        background-color: white;
        z-index: 999;
    }
    .fixed-header {
        border-bottom: 1px solid black;
    }
</style>
    """,
    unsafe_allow_html=True
)


with st.expander("About this application"):
    st.write("""
    This application provides an in-depth analysis and visualization tool tailored to explore and understand the hydrological characteristics of  in Petra, Jordan. By leveraging advanced digital elevation models and hydrological processing techniques, this tool offers insights into water flow dynamics within this culturally and historically significant area.
    """)

with st.expander("Threshold Functionality Explained:"):
    st.write("""
    The accumulation threshold sets the minimum amount of upstream water accumulation required for a channel to be visualized as part of the stream network. Adjusting this threshold allows users to filter out smaller stream channels and focus on more significant water flow paths, or vice versa. This feature is particularly useful for understanding the impact of varying hydrological conditions and can help in planning flood control measures, water resource management, or archaeological site protection.
    """)
def setup_dem_selector():
# Dropdown menu for DEM file selection
    dem_files = {
        "Wadi Musa": "./data/wadi_84.tif",
        "Al_nayra": "./data/al_nayraDEM_WGS84.tif",
        "Silaysil": "./data/silaysilDEM_WGS84.tif"
    }
    selected_name = st.selectbox('Select a DEM file:', options=list(dem_files.keys()))
    return dem_files[selected_name]
    
def process_dem_and_extract_stream_network(dem_path, acc_threshold):
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    # Fill pits
    pit_filled_dem = grid.fill_pits(dem)
    # Fill depressions
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    # Resolve flats
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Derive flow direction
    fdir = grid.flowdir(dem=inflated_dem, dirmap=(64, 128, 1, 2, 4, 8, 16, 32))

    # Compute accumulation
    acc = grid.accumulation(fdir=fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32), routing='d8')

    # Extract stream network
    branches = grid.extract_river_network(fdir, acc > acc_threshold, dirmap=(64, 128, 1, 2, 4, 8, 16, 32))

    return dem, acc, branches, grid

def plot_dem_and_acc(dem, acc, grid):
    # Plotting DEM
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    dem_im = axs[0].imshow(dem, extent=grid.extent, cmap='gist_earth',
                           interpolation='bilinear', vmin=900, vmax=1100)
    axs[0].set_title('Digital Elevation Model')
    plt.colorbar(dem_im, ax=axs[0], orientation='vertical', label='Elevation')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    # Plotting Flow Accumulation
    acc_im = axs[1].imshow(acc, extent=grid.extent, cmap='cubehelix',
                           norm=colors.LogNorm(1, acc.max()), interpolation='bilinear')
    axs[1].set_title('Flow Accumulation')
    plt.colorbar(acc_im, ax=axs[1], orientation='vertical', label='Upstream Cells')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    
    plt.tight_layout()
    return fig

def main(dem_path):
    #st.title("Welcome to the Flow Network Visualizer for Wadi, Petra, Jordan")


    # Display static plots outside the button control
    #dem_path = "/home/pnsinha/wadiDEM/wadiwgs_wgs84.tif"
    dem, acc, branches, grid = process_dem_and_extract_stream_network(dem_path, 1000)  # Example threshold
    fig_static = plot_dem_and_acc(dem, acc, grid)

    # Slider for accumulation threshold
    acc_threshold = st.slider("Select accumulation threshold for stream network extraction:", min_value=100, max_value=10000, step=100, value=1000)

    if st.checkbox("Process DEM and Extract Stream Network", value=True):
        with st.spinner('Processing DEM and computing stream network...'):
            dem, acc, branches, grid = process_dem_and_extract_stream_network(dem_path, acc_threshold)
            fig_network = plot_stream_network(acc, branches, grid, acc_threshold)
            st.pyplot(fig_network)  # Show network plot in the middle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

def plot_stream_network(acc, branches, grid, acc_threshold):
    fig, ax = plt.subplots(figsize=(6, 8))  # Adjust the figure size as needed
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)

    # Extract grid details
    min_x, min_y, max_x, max_y = grid.extent
    num_cols, num_rows = acc.shape
    cellsize_x = (max_x - min_x) / num_cols
    cellsize_y = (max_y - min_y) / num_rows

    acc_img = ax.imshow(acc, extent=grid.extent, zorder=2,
                        cmap='cubehelix',
                        norm=colors.LogNorm(1, acc.max()),
                        interpolation='bilinear')
    # The colorbar call has been removed here

    # Prepare a color map for lines
    cmap = plt.cm.viridis

    # Define line width scaling
    max_log_acc = np.log(np.max(acc))
    min_log_acc = np.log(acc_threshold)
    log_range = max_log_acc - min_log_acc

    # Process each branch
    for branch in branches['features']:
        coords = branch['geometry']['coordinates']
        # Precompute index values for efficiency
        indices = [(int((y - min_y) / cellsize_y), int((x - min_x) / cellsize_x)) 
                   for x, y in coords if min_x <= x <= max_x and min_y <= y <= max_y]

        # Filter valid indices and accumulate values
        valid_indices = [index for index in indices if 0 <= index[0] < num_rows and 0 <= index[1] < num_cols]
        acc_values = [acc[index] for index in valid_indices]
        
        if not acc_values:
            continue
        
        max_acc = np.max(acc_values)
        norm_acc = max_acc / np.max(acc)  # Normalize accumulation for color mapping

        # Apply geometric scaling to line width
        if max_acc > acc_threshold:
            scaled_log_acc = np.log(max_acc)
            linewidth = ((scaled_log_acc - min_log_acc) / log_range) * 9 + 1  # Scale from 1 to 10
        else:
            linewidth = 1

        color = cmap(norm_acc)  # Map normalized accumulation to color

        # Plot using precomputed indices for coordinates
        ax.plot(*zip(*[(min_x + idx[1] * cellsize_x, min_y + idx[0] * cellsize_y) for idx in valid_indices]), 
                color=color, linewidth=linewidth, alpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.title('Flow Accumulation and Stream Network', size=14)
    plt.tight_layout()

    return fig




if __name__ == "__main__":
    dem_path = setup_dem_selector()
    main(dem_path)

st.sidebar.image("https://gis.rcc.uchicago.edu/wp-content/uploads/2024/02/LOGO_GIS_New_1.png", use_column_width=True)

with st.sidebar:
    expander = st.expander("Key Features of the Application:")
    st.write("""
    This application visualizes the flow accumulation and stream network extracted from a Digital Elevation Model (DEM).
    **Key processes include:**
    - **DEM Processing**: Fill pits, depressions, and resolve flats to prepare the DEM for hydrological analysis.
    - **Flow Direction**: Calculates the steepest descent direction from each cell.
    - **Flow Accumulation**: Determines how many cells contribute water to each cell.
    - **Stream Network Extraction**: Identifies rivers from areas where flow accumulation exceeds a threshold.
    """)

st.sidebar.title("Wadi, Petra, Jordan")
st.sidebar.info(
    """
    PI: Dr. Sarah Newman, \n 
    University of Chicago
    
    RCC GIS: <https://gis.rcc.uchicago.edu>
    """
)


dem, acc, branches, grid = process_dem_and_extract_stream_network(dem_path, 1000)  # Example threshold
fig_static = plot_dem_and_acc(dem, acc, grid)
st.pyplot(fig_static)

st.markdown("""
- **Dynamic Visualization:** Interactively explore the flow accumulation and stream networks derived from high-resolution elevation data. Adjust parameters to see how different conditions affect water flow paths and network formations.
- **Geographical Focus:** Specifically focused on the Wadi regions in Petra, Jordan, this tool brings valuable geographical insights that are crucial for environmental planning, archaeological research, and conservation efforts.
- **Advanced Hydrological Analysis:**
    - **DEM Processing:** The tool begins with raw digital elevation models (DEM) which undergo several processing steps:
        - **Fill Pits:** Automatically detects and fills isolated depressions in the elevation data which may inaccurately represent surface water collection.
        - **Fill Depressions and Resolve Flats:** Further processes the DEM to prepare it for accurate flow direction analysis by ensuring the elevation data is optimally conditioned.
        - **Flow Direction Computation:** Calculates the precise direction that water is expected to flow across the landscape, which is critical for understanding watershed dynamics.
        - **Flow Accumulation Calculation:** Determines how much water is likely to flow through each point on the map, a key metric for identifying potential river and stream channels.
        - **Stream Network Extraction:** Identifies actual water channels based on flow accumulation data, highlighting the main watercourses and their tributaries.
""")

with st.expander("Purpose and Utility:"):
    st.write("""
    This tool is designed to aid in the effective management and study of water resources in arid regions such as Petra. This visualizer serves as an aid in understanding and preserving the delicate balance of Petra's natural and historical landscapes.

    Feel free to adjust the accumulation threshold using the slider and press the button to visualize the stream network based on current settings. This will provide you with a real-time interpretation of how water interacts with the landscape under various scenarios.
    """)
