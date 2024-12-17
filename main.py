import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data():
    """Load the CSV file into a DataFrame."""
    try:
        data = pd.read_csv("data/original_Values.csv", header=None, index_col=None)
        labels = data.iloc[:, -3:]  # Last 3 columns for labels
        data = data.iloc[:, 1:-3]  # Exclude the first column and last 3 columns
        return data, labels
    except FileNotFoundError:
        st.error("File not found. Please ensure it is in the same directory.")
        return None, None


def visualize(pattern, title, ylabel, end):
    """Visualize the given pattern."""
    ran = np.linspace(0, end, len(pattern))
    fig, ax = plt.subplots(figsize=(18, 3))
    ax.plot(ran, pattern, linewidth=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig)  # Streamlit rendering


def overlapMean(pattern, title, ylabel, end):
    results = np.zeros(len(pattern) - 100, dtype="int32")
    for i in range(pattern.shape[0] - 100):
        results[i] = np.round(np.mean(pattern[i : i + 100])).astype("int32")

    visualize(results, title=title, ylabel=ylabel, end=end)


def nonOverlapMean(pattern, title, ylabel, end):
    results = np.zeros(len(pattern) - 100, dtype="int32")
    result2 = []
    for i in range(pattern.shape[0] - 100):
        results[i] = np.round(np.mean(pattern[i : i + 100])).astype("int32")

    for j in range(0, results.shape[0] - 100, 100):
        result2.append(np.round(results[j : j + 100].mean()).astype("int32"))
    visualize(result2, title=title, ylabel=ylabel, end=end)


def normalizeAndVisualize(pattern, title, ylabel, end, range):
    scaler = MinMaxScaler(feature_range=(0, range))
    normalized_data = scaler.fit_transform(pattern.reshape(-1, 1)).flatten()
    ylabel = f"0-{range} Normalized Values"
    visualize(visualize(normalized_data, title=title, ylabel=ylabel, end=end))


def main():
    # Title of the app
    st.title("XRD Pattern Analysis App")

    # Load the CSV data
    data, labels = load_data()
    if data is None or labels is None:
        return

    # Display the data preview
    st.write("Preview of the data:")

    st.dataframe(data.head())

    # Input field for location
    location = st.text_input("Enter the location (index) you want to filter:")

    if location:
        try:
            location_index = int(location)
            filtered_data = data.iloc[location_index, :].values
            st.write("Preview of the selected example:")
            st.dataframe(filtered_data.reshape(1, -1))
            filtered_label = labels.iloc[location_index, :].values
        except (ValueError, IndexError):
            st.error("Invalid location. Please enter a valid numeric index.")
            return

        # Sidebar for buttons
        with st.sidebar:
            st.write("Select a visualization:")

            # Title to be reused for all visualizations
            title = (
                f"Example at Index: {location}, Crystal System: {int(filtered_label[0])}, "
                f"Extinction Group: {int(filtered_label[1])}, Space Group: {int(filtered_label[2])}"
            )

            # Normal Visualization Checkboxes
            original = st.checkbox("Original Visualization", key="original_checkbox")
            norm_0_1000 = st.checkbox(
                "Normalized (0-1000) Visualization", key="norm_0_1000_checkbox"
            )
            norm_0_10000 = st.checkbox(
                "Normalized (0-10000) Visualization", key="norm_0_10000_checkbox"
            )

            # Exclusive checkboxes for Overlap and Non-Overlap Mean
            overlap_mean = st.checkbox("Overlap Mean", key="overlap_mean_checkbox")
            non_overlap_mean = st.checkbox(
                "Non-Overlap Mean", key="non_overlap_mean_checkbox"
            )
            # Overlap Mean Options
            normalized_option = None
            if overlap_mean:
                normalized_option = st.radio(
                    "Overlap Mean Options:",
                    options=[
                        "None",
                        "0-1000 Normalized Overlap Mean",
                        "0-10000 Normalized Overlap Mean",
                    ],
                    key="overlap_mean_options_radio",
                )

            # Non-Overlap Mean Options
            non_overlapped_normalized_option = None
            if non_overlap_mean:
                non_overlapped_normalized_option = st.radio(
                    "Non-Overlap Mean Options:",
                    options=[
                        "None",
                        "0-1000 Normalized Non-Overlap Mean",
                        "0-10000 Normalized Non-Overlap Mean",
                    ],
                    key="non_overlap_mean_options_radio",
                )

        # Render plots
        if original:
            # title = (
            #     f"Example at Location: {location}, Crystal System: {int(filtered_label[0])}, "
            #     f"Extinction Group: {int(filtered_label[1])}, Space Group: {int(filtered_label[2])}"
            # )
            st.write("Original Visualization:")
            visualize(
                filtered_data.flatten(), title, "Original Intensity Values", 10000
            )
        if norm_0_1000:
            scaler = MinMaxScaler(feature_range=(0, 1000))
            normalized_data = scaler.fit_transform(
                filtered_data.reshape(-1, 1)
            ).flatten()
            st.write("Normalized (0-1000) Visualization:")
            visualize(
                normalized_data,
                title=title,
                ylabel="Intensity (0-1000)",
                end=10000,
            )

        if norm_0_10000:
            scaler = MinMaxScaler(feature_range=(0, 10000))
            normalized_data = scaler.fit_transform(
                filtered_data.reshape(-1, 1)
            ).flatten()
            st.write("Normalized (0-10000) Visualization:")
            visualize(
                normalized_data,
                title=title,
                ylabel="Intensity (0-10000)",
                end=10000,
            )

        if overlap_mean:
            visualize(
                filtered_data.flatten(),
                title=title,
                ylabel="Original Intensity Values",
                end=10000,
            )
            if normalized_option == "0-1000 Normalized Overlap Mean":

                scaler = MinMaxScaler(feature_range=(0, 1000))
                normalized_overlap = scaler.fit_transform(
                    filtered_data.reshape(-1, 1)
                ).flatten()
                visualize(
                    normalized_overlap.flatten(),
                    title=title,
                    ylabel="0-1000 Normalized Values",
                    end=10000,
                )
                overlapMean(
                    normalized_overlap,
                    title=title,
                    ylabel="0-1000 Overlapped Mean",
                    end=9900,
                )

            elif normalized_option == "0-10000 Normalized Overlap Mean":

                scaler = MinMaxScaler(feature_range=(0, 10000))
                normalized_overlap = scaler.fit_transform(
                    filtered_data.reshape(-1, 1)
                ).flatten()
                visualize(
                    normalized_overlap.flatten(),
                    title=title,
                    ylabel="0-10000 Normalized Values",
                    end=10000,
                )
                overlapMean(
                    normalized_overlap,
                    title=title,
                    ylabel="0-10000 Overlapped Mean",
                    end=9900,
                )
        if non_overlap_mean:
            visualize(
                filtered_data.flatten(),
                title=title,
                ylabel="Original Intensity Values",
                end=10000,
            )
            if non_overlapped_normalized_option == "0-1000 Normalized Non-Overlap Mean":
                scaler = MinMaxScaler(feature_range=(0, 1000))
                normalized_Nonoverlap = scaler.fit_transform(
                    filtered_data.reshape(-1, 1)
                ).flatten()
                visualize(
                    normalized_Nonoverlap.flatten(),
                    title=title,
                    ylabel="0-1000 Normalized Values",
                    end=10000,
                )
                nonOverlapMean(
                    normalized_Nonoverlap,
                    title=title,
                    ylabel="0-1000 Non-Overlapped Mean",
                    end=99,
                )

            elif (
                non_overlapped_normalized_option
                == "0-10000 Normalized Non-Overlap Mean"
            ):
                scaler = MinMaxScaler(feature_range=(0, 10000))
                normalized_Nonoverlap = scaler.fit_transform(
                    filtered_data.reshape(-1, 1)
                ).flatten()
                visualize(
                    normalized_Nonoverlap.flatten(),
                    title=title,
                    ylabel="0-10000 Normalized Values",
                    end=10000,
                )
                nonOverlapMean(
                    normalized_Nonoverlap,
                    title=title,
                    ylabel="0-10000 Non-Overlapped Mean",
                    end=99,
                )
if __name__ == "__main__":
    main()
