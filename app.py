import pandas as pd
import dash
from PIL import Image
from dash import dcc, html, Input, Output, dash_table, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import FixaTons
from FixaTons import COLLECTION_PATH
import plotly.graph_objects as go
import flask
import glob
import os
import numpy as np
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
import skimage.io as sio
from plotly.subplots import make_subplots
from skimage import data, draw
from scipy import ndimage

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# print(FixaTons.stats.statistics('SIENA12'))

datasets_list = FixaTons.info.datasets()

AOI = []
AOI_type = "rect"

# ------------------ App layout ----------------------
VIZ_STYLE = {
    'display': 'flex',
    'flex-direction': 'row'
}

VIZ_ROW_STYLE = {
    'padding': 10,
    'flex': 1
}

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
}

TITLE_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970',
    'font-size': 'large'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}

TABLE_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970',
    'font-size': 'small'
}

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

content_first_row = dbc.Row([
    dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.H4('Number of Fixations', className='card-title',
                        style=CARD_TEXT_STYLE),
                html.P(children=[22], style=CARD_TEXT_STYLE),
            ]),
        ]),
        md=4
    ),
    dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.H4(id='card_title_1', children=['Number of Words'], className='card-title',
                        style=CARD_TEXT_STYLE),
                html.P(id='card_text_1', children=[44],
                       style=CARD_TEXT_STYLE),
            ])
        ]),
        md=4
    ),
    dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.H4('Number of Filtered Words', className='card-title',
                        style=CARD_TEXT_STYLE),
                html.P(children=[66], style=CARD_TEXT_STYLE),
            ]),
        ]),
        md=4
    )
])

database_select = html.Div([
    dcc.Dropdown(id='ddDB', options=[{'label': DB_NAME, 'value': DB_NAME} for DB_NAME in datasets_list],
                 value=datasets_list[0])
])

image_select = html.Div([
    # dcc.Dropdown(
    #     id='image-dropdown',
    #     options=[{'label': i, 'value': i} for i in list_of_images],
    #     value=list_of_images[0]
    # ),
    dcc.Dropdown(id='image-dropdown')
])

participant_select = html.Div([
    dcc.Dropdown(id='ddParticipants', multi=True)
])

image_display = html.Div(id='image_display', children=[
    dcc.Graph(id='image', config={
        "modeBarButtonsToAdd": [
            "drawclosedpath",
            "drawrect"
        ]
    })
])

sidebar = html.Div([
    html.H2('EyeExplore'),
    html.Hr(),
    html.P('Select Dataset'),
    database_select,
    html.Hr(),
    html.P('Select Image Stimulus'),
    image_select,
    html.Hr(),
    html.P('Select Participants'),
    participant_select,
    html.Hr()
], style=SIDEBAR_STYLE)

content_tabs = html.Div([
    dcc.Tabs(id="content_tabs", value='tab-1', children=[
        dcc.Tab(label='Single User Statistics', value='single', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Multiple User Statistics', value='multiple', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Summary Statistics', value='summary', style=tab_style, selected_style=tab_selected_style)
    ], style=tabs_styles),
    html.Div(id='tabs-content-inline')
])

content = html.Div([
    sidebar,
    # content_first_row,
    image_display,
    html.Hr(),
    content_tabs
], style=CONTENT_STYLE)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
# app.layout = html.Div([sidebar, content])
app.layout = html.Div([content])


@app.callback(
    dash.dependencies.Output('image-dropdown', 'options'),
    [dash.dependencies.Input('ddDB', 'value')])
def update_image_dd(value):
    img_list = FixaTons.info.stimuli(value)
    return img_list


@app.callback(
    dash.dependencies.Output('ddParticipants', 'options'),
    [dash.dependencies.Input('image-dropdown', 'value'), dash.dependencies.Input('ddDB', 'value')])
def update_image_dd(stimulus, db):
    participants_list = FixaTons.info.subjects(db, stimulus)
    return participants_list


@app.callback(
    dash.dependencies.Output('image', 'figure'),
    [dash.dependencies.Input('ddDB', 'value'), dash.dependencies.Input('image-dropdown', 'value')])
def update_image_src(db_name, stimulus):
    img = np.array(FixaTons.get.stimulus(db_name, stimulus))
    fig = px.imshow(img)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(dragmode="drawclosedpath", newshape=dict(fillcolor="cyan", opacity=0.3, line=dict(color="darkblue", width=8)))
    return fig


def path_to_indices(path):
    """From SVG path to numpy array of coordinates, each row being a (row, col) point
    """
    indices_str = [el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")]
    return np.rint(np.array(indices_str, dtype=float)).astype(np.int32)

def path_to_mask(path, shape):
    """From SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)
    mask = np.zeros(shape, dtype=np.bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask


@app.callback(dash.dependencies.Output('tabs-content-inline', 'children'),
              [dash.dependencies.Input('ddParticipants', 'value'),
               Input('image', 'relayoutData'),
               dash.dependencies.Input('ddDB', 'value'),
               dash.dependencies.Input('image-dropdown', 'value'),
               Input('content_tabs', 'value')],
              prevent_initial_call=True)
def on_new_annotation(participants, relayout_data, db_name, stimulus, tab):
    # Source: https://dash.plotly.com/annotations
    trigger_id = ctx.triggered_id

    global AOI, AOI_type

    image_width, image_height = FixaTons.get.stimulus_size(db_name, stimulus)
    encoded_string = FixaTons.get.stimulus_base64_encoding(db_name, stimulus)
    image = f'data:image/png;base64,{encoded_string}'
    stimulus_image = FixaTons.get.stimulus(db_name, stimulus)
    eight_bit_img = Image.fromarray(stimulus_image).convert('P', palette='WEB', dither=None)

    if trigger_id == 'image-dropdown':
        AOI = []

    if trigger_id == 'image':
        if "shapes" in relayout_data:
            last_shape = relayout_data["shapes"][-1]
            if last_shape["type"] == "rect":
                x0, y0, x1, y1 = int(last_shape["x0"]), int(last_shape["y0"]), int(last_shape["x1"]), int(last_shape["y1"])
                AOI.append([x0, y0, x1, y1])
                AOI_type = "rect"
            else:
                indices = path_to_indices(last_shape["path"])
                AOI.append(indices)
                AOI_type = "free"
            # print([p.x, p.y] for p in indices)
            # mask = path_to_mask(last_shape["path"], stimulus_image.shape)
            # print(indices)
            # shape coordinates are floats, we need to convert to ints for slicing

            # print(indices)
            # print(indices[0][0], indices[0][1])


    if tab == 'single':
        if len(participants) > 0:
            df_scanpath = FixaTons.get.scanpath_aoi(db_name, stimulus, participants[0], AOI)
            df_scanpath["ELAPSED_TIME"] = df_scanpath["TIME_TO"] - df_scanpath["TIME_FROM"]
            print(df_scanpath)

            # Visualization: Animation of Single User Scanpath
            animation = px.scatter(df_scanpath, x="X", y="Y", animation_frame="TIME_FROM",
                                animation_group="SUBJECT",
                                size="ELAPSED_TIME", color="SUBJECT", hover_name="SUBJECT",
                                # log_x=True, size_max=55,
                                range_x=[0, image_width], range_y=[0, image_height])
            animation.add_layout_image(source=image, xref="x", yref="y", x=0, y=image_height,
                                       sizex=image_width, sizey=image_height,
                                       sizing="stretch", opacity=1, layer="below")
            animation.update_xaxes(visible=False)
            animation.update_yaxes(visible=False)
            return html.Div([dcc.Loading(dcc.Graph(figure=animation), type = "cube")])
        else: return html.Div(["Please select one participant!"])

    if tab == 'multiple':
        df = FixaTons.get.scanpath_aoi(db_name, stimulus, participants, AOI, AOI_type)
        df["ELAPSED_TIME"] = df["TIME_TO"] - df["TIME_FROM"]
        # df["AOI"] = df["AOI"].astype(str)

        # Visualization: Line Plot - Temporal Evolution of Scanpaths of Multiple Participants
        line_plot = px.line(df, x='TIME_FROM', y='AOI', color='SUBJECT')

        # Visualization: 3D Line Plot
        line_plot_3d = px.line_3d(df, x="X", y="Y", z="TIME_FROM", color='SUBJECT')
        line_plot_3d.update_xaxes(visible=False)
        line_plot_3d.update_xaxes(visible=False)
        z = np.zeros(stimulus_image.shape[:2])
        line_plot_3d.add_surface(z=z, surfacecolor=np.flipud(eight_bit_img), showscale=False)
        camera_params = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=-1.25, y=-1.25, z=0.5))
        line_plot_3d.update_layout(scene_camera=camera_params)

        # line_plot_3d.update_layout(scene = dict(
        #                             zaxis = dict(backgroundcolor="rgba(0, 0, 0, 0)",
        #                                  gridcolor="white",
        #                                  showbackground=True,
        #                                  zerolinecolor="white"),
        #                             yaxis = dict(backgroundcolor="rgba(0, 0, 0, 0)",
        #                                  gridcolor="white",
        #                                  showbackground=True,
        #                                  zerolinecolor="white"),
        #                             xaxis=dict(backgroundcolor="rgba(0, 0, 0, 0)",
        #                                        gridcolor="white",
        #                                        showbackground=True,
        #                                        zerolinecolor="white")))

        # Visualization: 3D scatter Plot
        scatter_3d = px.scatter_3d(df, x='X', y='Y', z='TIME_FROM',
                    color='SUBJECT', size='ELAPSED_TIME', size_max=18)
        scatter_3d.add_surface(z=z, surfacecolor=np.flipud(eight_bit_img), showscale=False)
        scatter_3d.update_xaxes(visible=False)
        scatter_3d.update_yaxes(visible=False)
        scatter_3d.update_layout(scene_camera=camera_params)

        # Visulization: Scarf Plot
        #TODO: Correct how x-axis is calculated.
        scarf_plot = px.bar(df, x="TIME_FROM", y="SUBJECT", color="AOI",
                            color_discrete_sequence=px.colors.qualitative.Vivid,
                            hover_data=["SUBJECT", "AOI"], orientation='h')
        scarf_plot.update_yaxes(categoryorder='category ascending')

        # animation = px.scatter(df, x="X", y="Y", animation_frame="TIME_FROM",
        #                        animation_group="TIME_FROM",
        #                        size="ELAPSED_TIME", color="SUBJECT", hover_name="SUBJECT",
        #                        # log_x=True, size_max=55,
        #                        range_x=[0, image_width], range_y=[0, image_height])
        # animation.add_layout_image(source=image, xref="x", yref="y", x=0, y=768,
        #                            sizex=image_width, sizey=image_height,
        #                            sizing="stretch", opacity=1, layer="below")

        return html.Div([html.Div([html.Div([dcc.Graph(figure=line_plot_3d)], style=VIZ_ROW_STYLE),
                                   html.Div([dcc.Graph(figure=scatter_3d)], style=VIZ_ROW_STYLE)], style=VIZ_STYLE),
                         html.Div([html.Div([dcc.Graph(figure=line_plot)], style=VIZ_ROW_STYLE),
                                   html.Div([dcc.Graph(figure=scarf_plot)], style=VIZ_ROW_STYLE)], style=VIZ_STYLE)
                         # html.Div([dcc.Loading(dcc.Graph(figure=animation), type="cube")])
                         ])
    if tab == 'summary':
        # Visualization: Attention Map
        fig = px.imshow(FixaTons.show.attention_map(db_name, stimulus))

        # Visualization: Transition Matrix
        df_transition_matrix = FixaTons.stats.AOI_transition_matrix(db_name, stimulus, AOI, AOI_type)
        z = np.array(df_transition_matrix)
        print(z)
        labels = df_transition_matrix.columns.values
        print(labels)
        transition_matrix = px.imshow(z, text_auto=True, x=labels, y=labels,
                                      labels=dict(x="AOI", y="AOI"), range_color=[0,1],
                                      color_continuous_scale='balance')
        transition_matrix.update_layout(xaxis=dict(tickmode='linear',tick0=0,dtick=1))
        transition_matrix.update_layout(yaxis=dict(tickmode='linear', tick0=0, dtick=1))

        return html.Div([html.Div([dcc.Graph(figure=fig)], style=VIZ_ROW_STYLE),
                         html.Div([dcc.Graph(figure=transition_matrix)], style=VIZ_ROW_STYLE)], style=VIZ_STYLE)


# @app.callback(Output('ddParticipants', 'value'),
#     [Input('content_tabs', 'value'), dash.dependencies.Input('ddParticipants', 'value')])
# def constraint_participants(tab, participants):
#     if tab == 'single':
#         if len(participants) > 0:
#             return participants[0]
#     else: return participants


if __name__ == '__main__':
    app.run_server(debug=True)
