#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Central script for running Dash app.
date: 20200823
author: spencerbraun
"""

import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.express as px
from dash.dependencies import Input, Output

import pandas as pd

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {"background": "#E0E0E2", "text": "#050949"}

with open("tsne.pkl", 'rb') as f:
    tsne = pickle.load(f)
tsne_df = pd.DataFrame(tsne, columns=['x', 'y'])

tsne_fig = px.scatter(
    tsne_df, x="x", y="y",
    # size="population", color="continent", hover_name="country",
    size_max=60)

with open("recommendation_table.pkl", 'rb') as f:
    table = pickle.load(f)

states = table.PageName.unique().tolist()
table_cols = [
    'Package', 
    'Language', 
    'Section', 
    'PageName',   
    'Similarity Score',
    'Link'
]
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(
        children='Machine Learning Package Lookup',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    dcc.Markdown('''

    # This is an <h1> tag

    ## This is an <h2> tag

    ###### This is an <h6> tag
    ''',
    style={
            'textAlign': 'center',
            'color': colors['text']
        }),
    html.H4(
        children='t-SNE Plot',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }),
    dcc.Graph(
        id='tsne-plot',
        figure=tsne_fig
    ),
    html.H4(
        children='Related Documentation Lookup',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }),
    dcc.Dropdown(
            id='filter_dropdown',
            options=[{'label':st, 'value':st} for st in states],
            value = states[0]
            ),
    dt.DataTable(
        id='table-container', 
        columns=[{'id': c, 'name': c} for c in table_cols]
        )
])

@app.callback(
    Output('table-container', 'data'),
    [Input('filter_dropdown', 'value') ] )
def display_table(state):
    idx = table.loc[table.PageName == state].index[0]
    outTable = processTable(table, idx)
    return outTable.to_dict('records')

def processTable(table, idx):
    probs = ["{:.2f}".format(x * 100) for x in table.iloc[idx].Rec_Probs]
    locs = table.iloc[idx].Rec_Index

    outputTable = (
        table
        .loc[locs]
        .join(pd.DataFrame(probs, index=locs, columns=['Similarity Score']))
    )
    
    return outputTable

if __name__ == '__main__':
    app.run_server()