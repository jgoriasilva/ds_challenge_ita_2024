import dash
from dash import dcc, html, Input, Output
import pandas as pd
from dash.dependencies import Input, Output
import requests
import plotly.express as px

def get_predictions(data: pd.DataFrame, url='http://127.0.0.1:8000/predict'):
    payload = {'data': data.to_json()}
    r = requests.post(url, params=payload)
    if r.status_code == 200:
        return r.json()['proba']
    else:
        print('Error:', r.status_code)
        return None

df = pd.read_csv('test.csv').sort_values(by=['dt_dep', 'dt_arr'])
df['dt_dep'] = pd.to_datetime(df['dt_dep'],utc=True)
df['dt_arr'] = pd.to_datetime(df['dt_arr'],utc=True)

app = dash.Dash(__name__)

# Define the layout of the application
app.layout = html.Div([
    html.H1("Flight Delay Visualization"),
    dcc.Dropdown(
        df['origem'].unique(),
        id='origem',
    ),
    dcc.RadioItems(
        ['Continuous', 'Manual'], 
        'Continuous',
        id='continuous-manual',
        inline=True,
        ),
    dcc.Graph(id='flight-delay-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # in milliseconds
        n_intervals=1
    ),
    # dcc.Slider(
    #     df['dt_dep'].min(),
    #     df['dt_dep'].max(),
    #     step=None,
    #     value=df['dt_dep'].min(),
    #     # marks={str(date): str(date) for date in }
    # ),
])

@app.callback(Output('interval-component', 'disabled'),
              Input('continuous-manual', 'value'))
def disable_interval(mode):
    if mode == 'Manual':
        return True
    return False

@app.callback(
    Output('flight-delay-graph', 'figure'),
    Input('interval-component', 'n_intervals'),
    # Input('continuous-manual', 'value')
)
def update_graph(n):
    global df
    
    n_samples = 100
    data = df[(n-1)*n_samples:n*n_samples]
    
    # data = df.sample(n=, replace=True)
    predictions = get_predictions(data)
    data['predictions'] = predictions

    fig = px.scatter(
        data, 
        x='dt_dep',
        y='predictions',
        color='origem',
    )

    fig.update_layout(transition_duration=500)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
