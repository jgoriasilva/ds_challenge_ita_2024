import dash
from dash import dcc, html, Input, Output
import pandas as pd
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
df['dt_dep'] = pd.to_datetime(df['dt_dep'], utc=True)
df['dt_arr'] = pd.to_datetime(df['dt_arr'], utc=True)

app = dash.Dash(__name__)

graphs = []
for origem in df['origem'].unique():
    graphs.append(
        html.Div([
            html.H2(f"Origem: {origem}", style={'textAlign': 'center', 'fontSize': '18px'}),
            dcc.Graph(id=f'flight-delay-graph-{origem}', style={'height': '300px'})
        ], style={'display': 'inline-block', 'width': '45%', 'margin': '10px'})
    )

app.layout = html.Div([
    html.H1("Previsão de atrasos", style={'textAlign': 'center'}),
    dcc.RadioItems(
        ['Continuous', 'Manual'],
        'Continuous',
        id='continuous-manual',
        inline=True,
        style={'textAlign': 'center', 'marginBottom': '20px'}
    ),
    html.Div(graphs, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
    dcc.Interval(
        id='interval-component',
        interval=5000,  # in milliseconds
        n_intervals=1
    ),
])

@app.callback(Output('interval-component', 'disabled'),
              Input('continuous-manual', 'value'))
def disable_interval(mode):
    if mode == 'Manual':
        return True
    return False

for origem in df['origem'].unique():
    @app.callback(
        Output(f'flight-delay-graph-{origem}', 'figure'),
        Input('interval-component', 'n_intervals')
    )
    def update_graph(n, origem=origem):
        global df
        
        n_samples = 100
        data = df[(n-1)*n_samples:n*n_samples]
        data = data[data['origem'] == origem]
        
        predictions = get_predictions(data)
        data['delay_probability'] = predictions

        fig = px.scatter(
            data,
            x='dt_dep',
            y='delay_probability',
            # color='origem',
            hover_data=['origem', 'destino', 'flightid'],
        )

        fig.update_layout(
            transition_duration=500,
            xaxis_title='Time of departure',
            yaxis_title='Probability of delay (%)'
        )

        return fig

if __name__ == '__main__':
    app.run_server(debug=False)
