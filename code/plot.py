import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.offline import *
init_notebook_mode(connected=True)

def plot_3d(point_cloud):
    x,y,z,_ = point_cloud.nonzero()
    trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=1,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.1
        ),
        opacity=1
    )
    )
    trace2 = go.Mesh3d(x=x,y=y,z=z,color='#FFB6C1', alphahull=1, opacity=0.6)
    data = [trace1, trace2]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)