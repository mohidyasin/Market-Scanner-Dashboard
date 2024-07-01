import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime
from lightweight_charts.widgets import StreamlitChart

# Set page to wide mode
st.set_page_config(layout="wide")

# Load your data from URLs
@st.cache_data
def load_data():
    result_df = pd.read_pickle('https://github.com/mohidyasin/Market-Scanner-Dashboard/raw/main/screener.pkl')
    df = pd.read_pickle('https://github.com/mohidyasin/Market-Scanner-Dashboard/raw/main/df.pkl')
    return result_df, df

result_df, df = load_data()

# Get unique dates and define min_date and max_date
dates = pd.to_datetime(df.index.get_level_values('Date').unique())
min_date = dates.min()
max_date = dates.max()

st.title('Stock Screener Dashboard')

# Sidebar for date selection and additional info
st.sidebar.title("Stock Screener Controls")
selected_date = st.sidebar.date_input('Select Date:', min_value=min_date, max_value=max_date, value=max_date)

# Convert selected_date to datetime for compatibility with your data
selected_date = pd.Timestamp(selected_date)

# Create columns for filters
col1, col2, col3 = st.columns(3)

# Filter 1
with col1:
    filter_column1 = st.selectbox('Filter Column 1:', options=result_df.columns)
with col2:
    filter_condition1 = st.selectbox('Condition 1:', options=['>=', '<='])
with col3:
    filter_threshold1 = st.number_input('Threshold 1:', value=0.2, format="%.2f")

# Enable second filter
enable_second_filter = st.checkbox('Enable 2nd Filter')

if enable_second_filter:
    col4, col5, col6 = st.columns(3)
    with col4:
        filter_column2 = st.selectbox('Filter Column 2:', options=result_df.columns)
    with col5:
        filter_condition2 = st.selectbox('Condition 2:', options=['>=', '<='])
    with col6:
        filter_threshold2 = st.number_input('Threshold 2:', value=0.2, format="%.2f")

# Sorting options
col7, col8 = st.columns(2)
with col7:
    sort_column = st.selectbox('Sort Column:', options=result_df.columns)
with col8:
    sort_order = st.selectbox('Sort Order:', options=['Ascending', 'Descending'])

# Filtering and sorting function
def apply_filter_and_sort(date, filters, sort_column, sort_order):
    # Get data for the selected date
    data = result_df.loc[date]

    # Remove rows with more than 5 NAs
    data = data.dropna(thresh=len(data.columns)-5)

    # Apply filters
    for column, condition, threshold, enabled in filters:
        if enabled:
            if condition == '>=':
                data = data[data[column] >= threshold]
            else:
                data = data[data[column] <= threshold]

    # Sort data
    ascending = sort_order == 'Ascending'
    sorted_data = data.sort_values(by=sort_column, ascending=ascending)

    return sorted_data.head(20).reset_index()

def prepare_chart_data(df, symbol, end_date):
    filtered_data = df.loc[(slice(None), symbol), :]
    filtered_data = filtered_data[filtered_data.index.get_level_values('Date') <= end_date]
    
    # Reset index to make Date a column
    filtered_data = filtered_data.reset_index()
    
    # Rename columns to match what lightweight_charts expects
    filtered_data = filtered_data.rename(columns={
        'Date': 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Convert time to string format that lightweight_charts expects
    filtered_data['date'] = filtered_data['date'].dt.strftime('%Y-%m-%d')
    
    return filtered_data



def add_regression_channel(chart, data, days, color_base, width=1):
    # Use the last 'days' worth of data
    data = data.tail(days)
    x = np.arange(len(data))
    y = data['close'].values
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept
    
    # Calculate standard deviation
    std_dev = np.std(y - line)
    
    # Create main regression line
    regression_line = chart.create_line(name=f'regression_{days}', color=color_base, style='solid', width=width, price_line=False, price_label=False)
    regression_line.set(pd.DataFrame({'time': data['date'], f'regression_{days}': line}))
    
    # Create upper and lower bands for +/-1 and +/-2 standard deviations
    for i in range(1, 3):
        style = 'dashed' if i == 1 else 'solid'
        upper_band = chart.create_line(name=f'upper_band_{i}_{days}', color=color_base, style=style, width=width, price_line=False, price_label=False)
        lower_band = chart.create_line(name=f'lower_band_{i}_{days}', color=color_base, style=style, width=width, price_line=False, price_label=False)
        
        upper_band.set(pd.DataFrame({'time': data['date'], f'upper_band_{i}_{days}': line + i * std_dev}))
        lower_band.set(pd.DataFrame({'time': data['date'], f'lower_band_{i}_{days}': line - i * std_dev}))

def add_ema(chart, data, period, width=1):
    # Calculate the EMA
    ema = data['close'].ewm(span=period, adjust=False).mean()
    
    # Prepare the EMA data with slopes
    ema_data = pd.DataFrame({'time': data['date'], 'EMA': ema})
    ema_data['slope'] = ema_data['EMA'].diff()
    
    # Create segments for dynamic coloring
    ema_segments = []
    for i in range(1, len(ema_data)):
        color = 'rgba(0, 255, 0, 0.8)' if ema_data['slope'].iloc[i] > 0 else 'rgba(255, 0, 0, 0.8)'
        segment = {
            'x': [ema_data['time'].iloc[i-1], ema_data['time'].iloc[i]],
            'y': [ema_data['EMA'].iloc[i-1], ema_data['EMA'].iloc[i]],
            'color': color
        }
        ema_segments.append(segment)
    
    # Add segments to the chart
    for segment in ema_segments:
        ema_line = chart.create_line(name=f'EMA_{period}', color=segment['color'], style='solid', width=width, price_line=False, price_label=False)
        ema_line.set(pd.DataFrame({'time': segment['x'], f'EMA_{period}': segment['y']}))




# Function to get next or previous symbol
def get_adjacent_symbol(current_symbol, symbols, direction):
    if current_symbol not in symbols:
        return symbols[0]  # Return the first symbol if the current one is not in the list
    current_index = symbols.index(current_symbol)
    if direction == 'next':
        return symbols[(current_index + 1) % len(symbols)]
    else:  # previous
        return symbols[(current_index - 1) % len(symbols)]

# Initialize session state
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = None

# Apply filters and display results
if st.button('Apply Filters and Sort'):
    filters = [
        (filter_column1, filter_condition1, filter_threshold1, True),
        (filter_column2, filter_condition2, filter_threshold2, enable_second_filter) if enable_second_filter else (None, None, None, False)
    ]

    st.session_state.filtered_df = apply_filter_and_sort(selected_date, filters, sort_column, sort_order)

    # Display filter and sort information
    st.subheader("Filter and Sort Settings")
    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.info(f"**Filter 1:** {filter_column1} {filter_condition1} {filter_threshold1}")
    with col_info2:
        if enable_second_filter:
            st.info(f"**Filter 2:** {filter_column2} {filter_condition2} {filter_threshold2}")
        else:
            st.info("**Filter 2:** Not enabled")
    with col_info3:
        st.info(f"**Sorted by:** {sort_column} ({sort_order})")

# Display filtered data if available
if st.session_state.filtered_df is not None:
    st.subheader("Filtered and Sorted Data")
    st.dataframe(st.session_state.filtered_df, use_container_width=True)

    # Display charts below the DataFrame
    if 'Symbol' in st.session_state.filtered_df.columns:
        selected_symbols = st.session_state.filtered_df['Symbol'].unique().tolist()
        
        # Reset selected_symbol if it's not in the new list
        if st.session_state.selected_symbol not in selected_symbols:
            st.session_state.selected_symbol = selected_symbols[0] if selected_symbols else None

        # Create three columns for the navigation
        col1, col2, col3 = st.columns([1,3,1])
        
        with col1:
            if st.button('⬅️ Previous'):
                if selected_symbols:
                    st.session_state.selected_symbol = get_adjacent_symbol(st.session_state.selected_symbol, selected_symbols, 'previous')

        with col2:
            if selected_symbols:
                st.session_state.selected_symbol = st.selectbox("Select a symbol", 
                                                                options=selected_symbols, 
                                                                key="symbol_select", 
                                                                index=selected_symbols.index(st.session_state.selected_symbol) if st.session_state.selected_symbol in selected_symbols else 0)
            else:
                st.write("No symbols available based on current filters.")

        with col3:
            if st.button('Next ➡️'):
                if selected_symbols:
                    st.session_state.selected_symbol = get_adjacent_symbol(st.session_state.selected_symbol, selected_symbols, 'next')

# Plot chart if symbol is selected
if st.session_state.selected_symbol:
    st.subheader(f"Chart for {st.session_state.selected_symbol}")
    
    chart_data = prepare_chart_data(df, st.session_state.selected_symbol, selected_date)
    
    # Center the chart
    col1, col2, col3 = st.columns([1,24,1])
    with col2:
        chart = StreamlitChart(width=1600, height=800)
        chart.set(chart_data)
        
        # Customize chart
        chart.layout(
            background_color='rgb(25, 25, 25)',  # Dark background
            text_color='rgb(255, 255, 255)'  # White text
        )
        # Add a watermark
        chart.watermark(
            text=f'{st.session_state.selected_symbol}',
            font_size=24,
            color='rgba(255, 255, 255, 0.5)'  # Semi-transparent white
        )
        # Configure legend
        chart.legend(
            visible=False,
            ohlc=False,  # Show OHLC values
            percent=True,  # Show percentage change
            lines=True,  # Show line for price
            color='rgb(255, 215, 0)',  # Gold color
            font_size=12
        )
        # Configure price line (disabled for regression lines)
        chart.price_line(
            label_visible=True,
            line_visible=True,
            title=""
        )
        # Add grid lines
        chart.grid(
            vert_enabled=True,
            horz_enabled=True,
            color='rgba(255, 255, 255, 0.1)'  # Semi-transparent white
        )

        # Add regression channel for the last 252 days (blue)
        add_regression_channel(chart, chart_data, days=252, color_base='rgba(0, 128, 255, 0.5)', width=2)
        # Add regression channel for the last 80 days (pink)
        add_regression_channel(chart, chart_data, days=80, color_base='rgba(255, 0, 255, 0.5)', width=2)
        # Add EMA with color based on the slope
        add_ema(chart, chart_data, period=21, width=2)
        # Fit the chart to the data
        chart.fit()
        chart.load()

# Add some space
st.write("")

# Add a note about the data
st.info("Note: This dashboard uses pre-loaded data from 'screener.pkl' and 'df.pkl'. Make sure these files are in the same directory as your Streamlit script.")

# Add the about section to the bottom of the sidebar
st.sidebar.title("About")
st.sidebar.info("This is a stock screener dashboard created with Streamlit. It allows you to filter and sort stock data based on various criteria.")