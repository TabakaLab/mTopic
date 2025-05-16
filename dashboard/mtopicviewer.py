import dash
from dash import html, dcc, callback, Input, Output, State, ctx, dash_table
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly import graph_objects as go
import muon as mu
import base64
from base64 import b64decode
from io import BytesIO
import tempfile
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import warnings
import logging
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='MUTO')
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--file', type=str, default=None)
args = parser.parse_args()

SEQUENTIAL_SCALE = [
    'aggrnyl', 'agsunset', 'blackbody', 'bluered', 'blues',
    'blugrn', 'bluyl', 'brwnyl', 'bugn', 'bupu', 'burg',
    'burgyl', 'cividis', 'darkmint', 'electric', 'emrld',
    'gnbu', 'greens', 'greys', 'hot', 'inferno', 'jet',
    'magenta', 'magma', 'mint', 'orrd', 'oranges', 'oryel',
    'peach', 'pinkyl', 'plasma', 'pubu', 'pubugn', 'purd',
    'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdpu',
    'redor', 'reds', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
    'turbo', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
]

DIVERGING_SCALE = [
    'armyrose', 'balance', 'curl', 'delta', 'icefire',
    'matter', 'picnic', 'portland', 'rdgy', 'rdylbu',
    'rdylgn', 'spectral', 'temps', 'tealrose', 'tropic',
    'earth', 'fall', 'geyser', 'prgn', 'puor'
]

DISCRETE_SCALE = [
    'Alphabet', 'Antique', 'Bold', 'Dark2', 'Dark24',
    'Light24', 'Pastel', 'Pastel1', 'Pastel2', 'Prism',
    'Safe', 'Set1', 'Set2', 'Set3', 'Vivid', 'G10', 'D3'
]

hide_plot_attributes = {
    'showgrid': False,
    'showticklabels': False,
    'showline': False,
    'zeroline': False,
    'title': '',
    'showspikes': False,
    "ticklen": 0
}

global_store = {
    'mudata': None,         # main object
    'selected_cells': [],   # for barplot
    'modal_figures': None,  # preview all topics, static image
    'df': None,             # coords + coloring feature for main figure
    'pie_charts': None      # preview topic distributions
}

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.MINTY],
    suppress_callback_exceptions=True
)
app.title = "mTopic Viewer"
app.layout = dbc.Container([
    dmc.Paper([
        dbc.Row([
            dbc.Col([
                dmc.LoadingOverlay(
                    html.Div([
                        dmc.Paper([
                            dmc.Stack([
                                dcc.Upload(
                                    id='upload-data',
                                    children=dbc.Button(
                                        'Drag and drop or select h5mu file to upload',
                                        color='secondary',
                                        outline=True,
                                        className='w-100'
                                    ),
                                    multiple=False
                                ),
                                dmc.Text(
                                    id='file-info',
                                    size='xs',
                                    c='dimmed',
                                    ta='center'
                                ),
                                dmc.Select(
                                    id='select-coordinates',
                                    label='Coordinates',
                                    data=[],
                                ),
                                dmc.Select(
                                    id='select-obs',
                                    label='Metadata',
                                    searchable=True,
                                    data=[],
                                ),
                                dmc.Select(
                                    label='Topic',
                                    id='select-topic',
                                    searchable=True,
                                    data=[],
                                ),
                                dbc.Button(
                                    id='topics-modal-button',
                                    children=['Preview all topics'],
                                    color='secondary',
                                    disabled=True,
                                    className='w-100'
                                ),
                                dbc.Modal([
                                        dbc.ModalHeader(
                                            dbc.ModalTitle(['Topic scores'])
                                        ),
                                        dbc.ModalBody([
                                            html.Div([
                                                html.Img(id='topics-modal-plot')
                                            ],
                                            style={'textAlign': 'center'}
                                            )
                                        ]),
                                    ],
                                    id="topics-modal",
                                    scrollable=True,
                                    fullscreen=True,
                                ),
                                dbc.Button(
                                    id='pie-charts-modal-button',
                                    children=['Explore with pie charts'],
                                    color='secondary',
                                    disabled=True,
                                    className='w-100'
                                ),
                                dbc.Modal([
                                        dbc.ModalHeader(
                                            dbc.ModalTitle(['Topic distributions'])
                                        ),
                                        dbc.ModalBody([
                                            html.Div([
                                                dcc.Graph(
                                                    id='pie-charts-modal-plot',
                                                    style={'height': '80vh'},
                                                    config={
                                                        'displaylogo': False,
                                                        'scrollZoom': True
                                                    }
                                                )
                                            ],
                                            style={'textAlign': 'center'}
                                            )
                                        ]),
                                    ],
                                    id="pie-charts-modal",
                                    size="xl",
                                ),
                                dmc.Select(
                                    id='select-modality',
                                    label='Modality feature',
                                    data=[],
                                ),
                                dcc.Dropdown(
                                    id='select-modality-feature',
                                    placeholder='Search modality feature',
                                    searchable=True,
                                    clearable=True,
                                ),
                                dbc.Row([
                                    dbc.Col(
                                        dmc.NumberInput(
                                            id='point-size',
                                            label='Marker size',
                                            value=8,
                                            min=1,
                                            precision=0,
                                            step=1
                                        ),
                                    ),
                                    dbc.Col(
                                        dmc.Select(
                                            id='point-symbol',
                                            label='Marker shape',
                                            value='circle',
                                            data=['circle', 'square'],
                                        ),
                                    )
                                ], className='g-1'),
                                dmc.Divider(
                                    label='Color scale',
                                    labelPosition='center',
                                ),
                                dbc.Row([
                                    dbc.Col(
                                        dmc.Select(
                                            id='continuous-scale',
                                            label='Continuous',
                                            value='magma',
                                            data=SEQUENTIAL_SCALE + DIVERGING_SCALE,
                                            searchable=True,
                                        ),
                                    ),
                                    dbc.Col(
                                        dmc.Select(
                                            id='discrete-scale',
                                            label='Discrete',
                                            value='Set1',
                                            data=DISCRETE_SCALE,
                                            searchable=True,
                                        ),
                                    ),
                                ], className='g-1'),
                                dbc.Row([
                                    dbc.Col(
                                        dmc.NumberInput(
                                            id='min-color-value',
                                            label='Min value',
                                            precision=3,
                                        ),
                                    ),
                                    dbc.Col(
                                        dmc.NumberInput(
                                            id='max-color-value',
                                            label='Max value',
                                            precision=3,
                                        ),
                                    ),
                                ], className='g-1'),
                            ], spacing='md')
                        ], p='md')
                    ], style={
                        'height': '99vh',
                        'overflowY': 'auto',
                        'overflowX': 'hidden',
                        'paddingRight': '10px',
                    }),
                loaderProps={'type': 'bars', 'size': 'xs', 'color': 'red'}),
            ], width=2),
            dbc.Col([
                dmc.Paper([
                    dcc.Graph(
                        id='main-plot',
                        style={'height': '99vh'},
                        config={
                            'displaylogo': False,
                            'scrollZoom': True,
                            'toImageButtonOptions': {
                                'filename': "MTM_main_plot"
                            }
                        },
                        figure={
                            'layout': {
                                'xaxis': {'visible': False},
                                'yaxis': {'visible': False},
                                'plot_bgcolor': 'white',
                                'paper_bgcolor': 'white',
                                'annotations': [{
                                    'text': 'Upload MTM data first',
                                    'xref': 'paper',
                                    'yref': 'paper',
                                    'x': 0.5,
                                    'y': 0.5,
                                    'showarrow': False,
                                    'font': {'size': 15}
                                }]
                            }
                        }
                    ),
                ])
            ], width=8),
            
            dbc.Col([
                dmc.Paper([
                    dcc.Graph(
                        id='topic-plot',
                        style={'height': '55vh',
                               'overflowY':'scroll'},
                        config={
                            'displaylogo': False,
                            'modeBarButtons': [['toImage']],
                            'toImageButtonOptions': {
                                'filename': "MTM_topic_distribution"
                            }
                        },
                        figure={
                            'layout': {
                                'xaxis': {'visible': False},
                                'yaxis': {'visible': False},
                                'plot_bgcolor': 'white',
                                'paper_bgcolor': 'white',
                                'annotations': [{
                                    'text': 'Select cells or spots first',
                                    'xref': 'paper',
                                    'yref': 'paper',
                                    'x': 0.5,
                                    'y': 0.5,
                                    'showarrow': False,
                                    'font': {'size': 15}
                                }]
                            }
                        }
                    ),
                    dmc.Space(h=10),
                    dbc.Row([
                        dbc.Col(
                            dmc.Select(
                                label='Show',
                                id='select-topic-display-method',
                                data=['Bar plot', 'Table'],
                                value='Bar plot',
                            ),
                            width=5
                        ),
                        dbc.Col(
                            dmc.Select(
                                label='Modality',
                                id='select-topic-modality-table',
                                searchable=True,
                                data=[],
                            ),
                            width=4
                        ),
                        dbc.Col(
                            dmc.NumberInput(
                                id='select-topic-n',
                                label='N',
                                value=100,
                                min=1
                            ),
                            width=3
                        ),
                    ], className="g-1"),
                    dmc.Space(h=10),
                    html.Div([],
                        id='top-topics',
                        style={
                            'height': '35vh',
                            'overflowY': 'auto',
                            'overflowX': 'hidden',
                            'paddingRight': '10px',
                        },
                    )
                ])
            ], width=2)
        ], className='g-0')
    ]), 
], fluid=True)


@callback(
    Output('file-info', 'children'),
    Output('file-info', 'c'),
    Output('select-modality', 'data'),
    Output('select-modality', 'value'),
    Output('select-coordinates', 'data'),
    Output('select-coordinates', 'value'),
    Output('select-topic', 'data'),
    Output('select-topic', 'value'),
    Output('select-obs', 'data'),
    Output('select-obs', 'value'),
    Output('topics-modal-button', 'disabled'),
    Output('pie-charts-modal-button', 'disabled'),
    Output('select-topic-modality-table', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    running=[
        (Output('upload-data', 'disabled'), True, False),
    ]
)
def upload_file(contents, filename):
    if contents is None and args.file is None:
        return (
            'No file uploaded', 'dimmed', 
            [], None, 
            [], None, 
            [], None, 
            [], None,
            True, True,
            []
        )

    if args.file is not None:
        filename = args.file
        decoded = b''
    else:    
        _, content_string = contents.split(',')
        decoded = b64decode(content_string)

    global global_store
    with tempfile.NamedTemporaryFile(suffix='.h5mu') as temp_file:
        temp_file.write(decoded)
        temp_file.flush()
        try:
            mdata = mu.read(temp_file.name if args.file is None else args.file)
            global_store['mudata'] = mdata
            modalities = list(mdata.mod.keys())
            metadata = list(mdata.obs.keys())
            coords = list(set(mdata.obsm.keys()) - set(mdata.mod.keys()) - set(['topics']))
            if len(coords) < 1:
                return (
                    f'Error. Didn\'t find coordinates in the observations obsm', 'red', 
                    [], None, 
                    [], None, 
                    [], None, 
                    [], None,
                    True, True,
                    []
                )
            elif 'topics' not in mdata.obsm.keys(): 
                return (
                    f'Uploaded {filename} but no topics were found', 'orange', 
                    modalities, modalities[0], 
                    coords, coords[0], 
                    [], None, 
                    [], None,
                    True, True,
                    []
                )
            else:
                topics = list(mdata.obsm['topics'].keys())
                return (
                    f'Succesfully uploaded {filename}', 'green', 
                    modalities, modalities[0], 
                    coords, coords[0], 
                    topics, None, 
                    metadata, None,
                    False, False,
                    modalities
                )
        except Exception as e:
            return (
                f'Error. {e}', 'red', 
                [], None, 
                [], None, 
                [], None, 
                [], None,
                True, True,
                []
            )


@callback(
    Output('select-modality-feature', 'options'),
    Input('select-modality-feature', 'search_value'),
    Input('select-modality', 'value'),
    prevent_initial_call=True
)
def update_feature_search_options(search_value, modality):
    if not search_value:
        raise PreventUpdate
    global global_store
    options = list(global_store['mudata'][modality].var.index)
    options_final = [o for o in options if search_value.lower() in o.lower()]
    modality_features = options_final[:100] if \
                        len(options_final) > 100 else options_final
    return modality_features


@callback(
    Output('main-plot', 'figure'),
    Input('select-topic', 'value'),
    Input('select-obs', 'value'),
    Input('select-coordinates', 'value'),
    Input('point-size', 'value'),
    Input('point-symbol', 'value'),
    Input('select-modality-feature', 'value'),
    Input('continuous-scale', 'value'),
    Input('discrete-scale', 'value'),
    Input('min-color-value', 'value'),
    Input('max-color-value', 'value'),
    State('select-modality', 'value'),
    prevent_initial_call=True
)
def update_main_graph(topic, obs, coordinates, point_size, point_symbol, modality_feature, 
                      c_scale, d_scale, min_color, max_color, modality):
    global global_store
    if global_store['mudata'] is None or point_size == '':
        raise PreventUpdate
    
    mdata = global_store['mudata']
    color_values = None 
    df = mdata.obsm[coordinates] if global_store['df'] is None else global_store['df']

    if ctx.triggered_id == 'select-topic':
        color_values = mdata.obsm['topics'][topic]
    elif ctx.triggered_id == 'select-obs':
        color_values = mdata.obs[obs]
    elif ctx.triggered_id == 'select-modality-feature' and modality_feature is not None:
        color_values = pd.DataFrame(
            mdata[modality][:, modality_feature].X.toarray().tolist(), 
            index = mdata.obs.index, 
            columns = [modality_feature]
        )
    if color_values is not None:
        df = pd.concat([mdata.obsm[coordinates], color_values], axis=1)
        global_store['df'] = df
    
    color_range_list = None
    if len(df.columns) > 2 and df.dtypes[2] != 'category':
        min_is_correct = min_color is None or min_color == ''        
        color_min = df.iloc[:, 2].min() if min_is_correct else min_color
        
        max_is_correct = max_color is None or max_color == ''
        color_max = df.iloc[:, 2].max() if max_is_correct else max_color
        
        color_range_list = [color_min, color_max]

    fig = go.Figure(
        px.scatter(
            df,
            x=df.columns[0],
            y=df.columns[1],
            color=df.columns[2] if len(df.columns) > 2 else None,
            color_discrete_sequence=eval('px.colors.qualitative.' + d_scale),
            color_continuous_scale=c_scale,
            range_color=color_range_list,
            hover_data={
                df.columns[0]: False,
                df.columns[1]: False,
                "Barcode": df.index,
            }
        )
    )
    fig.update_traces(
        marker=dict(
            size=point_size,
            symbol=point_symbol
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=10),
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(255,255,255)',
        yaxis_scaleanchor="x",
        uirevision=True,
        legend=dict(
            font=dict(size=15),
            itemsizing='constant'
        ),
    )
    return fig


@app.callback(
    Output('topic-plot', 'figure'),
    Input('main-plot', 'clickData'),
    Input('main-plot', 'selectedData'),
    prevent_initial_call=True
)
def update_barplot(selected_cell, selected_area):
    global global_store

    if ctx.triggered_id == 'main-plot':
        selection_type = list(ctx.triggered_prop_ids.keys())[0].split('.')[1]
        selected_data = selected_cell if selection_type == 'clickData' else selected_area
        if selected_data is None or selected_data['points'] == []:
            cell_ids = global_store['selected_cells']
        else:
            cell_ids = [point['pointIndex'] for point in selected_data['points']]
            global_store['selected_cells'] = cell_ids
    else:
        cell_ids = global_store['selected_cells']

    n_cells = len(cell_ids)
    if n_cells < 1:
        raise PreventUpdate

    mdata = global_store['mudata']
    data_source = mdata.obsm['topics']
    column_source = mdata.obsm['topics'].columns

    data_source = data_source.iloc[cell_ids, :].mean(axis=0) if n_cells > 1 else \
                  data_source.iloc[cell_ids[0], :]
    df = pd.DataFrame({
        'Topic proportion': data_source, 
        'topic': column_source
    }).iloc[::-1]
    
    fig_title = 'Selected cell: ' + data_source.name if n_cells == 1 else \
                'Number of selected cells: ' + str(n_cells)
    
    fig = go.Figure(px.bar(
        df,
        x='Topic proportion',
        y='topic',
        orientation='h',
        title = fig_title
    ))
    fig.update_coloraxes(showscale=False)
    fig.update_traces(marker_color='#555555')
    fig.update_layout(
        uirevision=True,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(font=dict(size=15)),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255,255,255)',
        yaxis=dict(
            tickfont=dict(size=10),
            dtick=1,
            title=None
        ),
        xaxis=dict(range=[0, 1.1], dtick=0.25),
        dragmode=False,
    )
    return fig


@app.callback(
    Output("topics-modal", "is_open"),
    Output('topics-modal-plot', 'src'),
    Input("topics-modal-button", "n_clicks"),
    State('select-coordinates', 'value'),
    State("topics-modal", "is_open"),
    prevent_initial_call=True
)
def topics_modal(button, coordinates, is_open):
    global global_store
    fig_data = global_store['modal_figures']
    
    if not button:
        return is_open, None

    if fig_data is None:
        mdata = global_store['mudata']
        df = pd.concat([mdata.obsm[coordinates], mdata.obsm['topics']], axis=1)
        df_melted = pd.melt(
            frame=df,
            id_vars=df.columns[:2],
            value_vars=df.columns[2:],
            var_name='Topic'
        )
        num_topics = len(df_melted['Topic'].unique())
        cols = 5
        rows = (num_topics + 3) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        axes = axes.ravel()
        for i, topic in enumerate(df_melted['Topic'].unique()):
            topic_data = df_melted[df_melted['Topic'] == topic]
            axes[i].scatter(
                topic_data.iloc[:,0],
                topic_data.iloc[:,1],
                c=topic_data['value'],
                cmap='inferno',
                s=2
            ) 
            axes[i].set_aspect('equal')
            axes[i].set_title(topic)

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=110, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        fig_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        fig_data = f'data:image/png;base64,{fig_data}'
        global_store['modal_figures'] = fig_data

    return not is_open, fig_data

@app.callback(
    Output("pie-charts-modal", "is_open"),
    Output('pie-charts-modal-plot', 'figure'),
    Input("pie-charts-modal-button", "n_clicks"),
    State('select-coordinates', 'value'),
    State("pie-charts-modal", "is_open"),
    prevent_initial_call=True
)
def pie_charts_modal(button, coordinates, is_open):
    global global_store
    mdata = global_store['mudata']
    fig = global_store['pie_charts']
    
    if not button:
        return is_open, None

    if fig is None:
        df = pd.concat([mdata.obsm[coordinates], mdata.obsm['topics']], axis=1)
        if df.shape[0] > 2000:
            df = df.sample(2000)

        x_col, y_col = df.columns[0:2]
        topic_cols = df.columns[2:]
        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        
        padding = 0.01
        pie_size = 0.015
        fig = go.Figure()
        for idx, row in df.iterrows():
            x_norm = (row[x_col] - x_min) / (x_max - x_min) * (1 - 2*padding) + padding
            y_norm = (row[y_col] - y_min) / (y_max - y_min) * (1 - 2*padding) + padding
            values = [row[col] for col in topic_cols]
            fig.add_trace(go.Pie(
                values=values,
                labels=topic_cols,
                name=f'{idx}',
                domain={
                    'x': [x_norm - pie_size/2, x_norm + pie_size/2],
                    'y': [y_norm - pie_size/2, y_norm + pie_size/2]
                },
                showlegend=True if idx == 0 else False,
                textinfo='none',
            ))
        
        fig.update_layout(
            title="",
            xaxis=dict(
                title="",
                range=[0, 1],
                showgrid=True,
                showticklabels=False,
                fixedrange=False
            ),
            yaxis=dict(
                title="",
                range=[0, 1],
                showgrid=True,
                showticklabels=False,
                fixedrange=False
            ),
            showlegend=True,
        )
        
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgb(255,255,255)',
            plot_bgcolor='rgb(255,255,255)',
            yaxis_scaleanchor="x",
            uirevision=True,
            dragmode='zoom'
        )
        global_store['pie_charts'] = fig

    return not is_open, fig


def generate_top_features_barplot(df, n):
    fig = go.Figure(px.bar(
        df.iloc[::-1],
        x='Score',
        y='Feature',
        orientation='h',
    ))
    fig.layout.uirevision = True
    fig.update_coloraxes(showscale=False)
    fig.update_traces(marker_color='#555555')
    fig.update_layout(
        margin=dict(l=0, r=0, t=5, b=0),
        paper_bgcolor='rgb(255, 255, 255)',
        plot_bgcolor='rgb(255,255,255)',
        yaxis=dict(
            tickfont=dict(size=10),
            dtick=1,
            title=None
        ),
        xaxis=dict(side='top'),
        dragmode=False,
    )
    
    graph = dcc.Graph(
        figure=fig,
        style={
            'height': str(1.5*n) + 'vh',
            'overflowY':'scroll'
        },
        config={
            'displaylogo': False,
            'modeBarButtons': [['toImage']],
            'toImageButtonOptions': {
                'filename': "MTM_top_features"
            }
        }
    )   
    return graph


def generate_top_features_table(df):
    df.loc[:, 'Score'] = df.loc[:, 'Score'].round(2)
    df = df.to_dict('records')
    table = dash_table.DataTable(
        data=df,
        columns=([
            {'id': 'No', 'name': 'No'},
            {'id': 'Feature', 'name': 'Feature'},
            {'id': 'Score', 'name': 'Score'}
        ]),
        page_current=0,
        page_size=10,
        
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold',
            'border': '0px',
            'padding': '2px 4px',
        },
        style_cell={
            'border': '0px',
            'padding': '2px 4px',
            'fontSize': '15px',
            'font-family': 'sans-serif'
        },
        style_data={
            'height': '20px',
            'lineHeight': '15px'
        },
        style_table={'overflowX': 'auto'},
    )
    return table


@app.callback(
    Output('top-topics', 'children'),
    Input('select-topic-display-method', 'value'),
    Input('select-topic-modality-table', 'value'),
    Input('select-topic', 'value'),
    Input('select-topic-n', 'value'),
    prevent_initial_call=True
)
def show_top_features(display_method, modality, topic, n):
    global global_store
    if modality is None or topic is None or n == '':
        raise PreventUpdate
    mdata = global_store['mudata']
    df = mdata[modality].varm['signatures']
    df = df.sort_values(by=[topic], ascending=False).head(n)
    df = pd.DataFrame({
        "No": list(range(1, df.shape[0] + 1)),
        "Feature": df.index, 
        "Score": df.loc[:, topic],
    })

    if display_method == 'Bar plot':
        return generate_top_features_barplot(df, n)
    else:
        return generate_top_features_table(df)

if __name__ == '__main__':
    app.run_server(debug=args.debug, port=args.port)
