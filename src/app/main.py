import os
from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash.dash_table.Format import Format, Scheme, Trim

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pickle
import SimpleITK as sitk

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

mount_point = '/mnt/raid/C1_ML_Analysis/'

csv_path = '/mnt/raid/C1_ML_Analysis/test_output/classification/c2_instance_table_clarius_resampled_file_path_extract_frames_blind_sweeps_merged_balanced_tsne.parquet'

df = pd.read_parquet(csv_path)
column_pred_class = 'pred_class'

# Create the scatter plot and include 'img_path' in custom_data so that it is available on click.
scatter_fig = px.scatter(
    df,
    x='tsne_0',
    y='tsne_1',
    color=column_pred_class,
    custom_data=['file_path_x']  # The image path will be returned in clickData
)

# Initialize the Dash app
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=scatter_fig),
    dcc.Graph(id='image-plot')  # This graph will display the image
])

def load_image(image_path):
    """
    Loads an image from the given file path and returns a numpy array
    for display in a Plotly figure.
    """
    try:
        img = sitk.ReadImage(os.path.join(mount_point, image_path))
        img_np = sitk.GetArrayFromImage(img)
        return img_np
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

@app.callback(
    Output('image-plot', 'figure'),
    Input('scatter-plot', 'clickData')
)
def display_image(clickData):
    if clickData is None:
        # If no data point has been clicked, return an empty figure with instructions.
        fig = go.Figure()
        fig.update_layout(
            title="Click on a data point to display the image",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    
    # Extract the image path from the custom_data of the clicked point.
    image_path = clickData['points'][0]['customdata'][0]
    img_array = load_image(image_path)
    
    if img_array is None:
        fig = go.Figure()
        fig.update_layout(
            title="Error loading image.",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig
    
    # Create a Plotly figure using the image array.
    fig = go.Figure(go.Image(z=img_array))
    fig.update_layout(title="US sample")
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8787)
