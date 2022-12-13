import numpy as np 
import plotly.graph_objects as go

def plot_words_extreme(negative_words, positive_words, num_words, 
    x_title='x_axis', y_title='y_axis', title='Title'):
    
    x_axis = np.linspace(-1,1,num_words)

    word_range_list = []
    words_list = []

    for e, w in negative_words:
        word_range_list.append(e)
        words_list.append(w)

    for e, w in positive_words:
        word_range_list.append(e)
        words_list.append(w)
    
    layout = go.Layout(
        width=1000, 
        height=1000, 
        xaxis=go.layout.XAxis(title=x_title), 
        yaxis=go.layout.YAxis(title=y_title),
        title=go.layout.Title(text=title,x=0.5) 
    )

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(
    x=x_axis,
    y=word_range_list,
    mode="markers+text",
    name="Markers and Text",
    text=words_list,
    textposition="bottom center"
    ))
    fig.show()
    