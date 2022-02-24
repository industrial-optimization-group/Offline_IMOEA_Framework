from os import error
from matplotlib.pyplot import axes
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.offline.offline import plot
import plotly.graph_objects as go
from plotly.graph_objs import *
from plotly.subplots import make_subplots

def plot_vals(objs, unc, preference, iteration, interaction_count, ideal, nadir, min, max, path):
    #max=1
    #min=0
    #objs = np.random.rand(10,3)*20
    #unc = np.random.rand(10,3)
    #preference = np.array([2,2,2])
    #ideal = np.array([-100,-100,-100])
    #nadir = np.array([-50,-50,-50])
    ideal = np.repeat(np.min(objs),3)
    nadir = np.repeat(np.max(objs),3)

    
    columns = ["f_"+str(i+1) for i in range(np.shape(objs)[1])]
    range_plot = np.vstack((ideal,nadir))
    range_plot = np.hstack((range_plot,[[3],[3]]))
    if np.shape(objs)[0] > 0:
        unc_avg = np.mean(unc, axis=1)
        min = np.min(unc_avg)
        max = np.max(unc_avg)
        #unc_avg = (unc_avg-np.min(unc_avg))/(np.max(unc_avg)-np.min(unc_avg))
        unc_avg = (unc_avg - min) / (max-min)
        objs_col = unc_avg.reshape(-1, 1)
        objs = np.hstack((objs, objs_col))
    objs1 = np.vstack((objs, range_plot))
    objs1 = np.vstack((objs1, np.hstack((preference.reshape(1,-1), [[2]]))))
    unc_avg = np.hstack((unc_avg.reshape(1,-1), [[3,3,2]]))
    unc_avg=unc_avg.flatten()
    unc2=np.vstack((unc, [[0,0,0],[0,0,0],[0,0,0]]))
    unc3 = 1.96*unc2
    color_scale_custom = [(0.0, 'rgb(69,2,86)'), (0.083, 'rgb(59,28,140)'), (0.167, 'rgb(33,144,141)'),
                            (0.25, 'rgb(90,200,101)'), (0.334, 'rgb(249,231,33)'),
                            (0.334, 'red'), (0.7, 'red'), (0.7, 'white'),
                            (1.0, 'white')]

    fig = go.Figure(data = [go.Scatter3d(x=objs1[:,0], y=objs1[:,1], z=objs1[:,2],
                        error_x=dict(array=unc2[:,0], color='purple'),
                        error_y=dict(array=unc2[:,1],color='purple'),
                        error_z=dict(array=unc2[:,2], color='purple'),
                mode='markers',
        marker=dict(
            size=12,
            color=unc_avg, 
            opacity=0.8,
            colorscale=color_scale_custom,
            cmin=0,
            cmax=3
            ),

        customdata  = unc3,
        hovertemplate =
        '<br><b>f_1</b>: %{x:.2f}<br>'+
        '<br><b>f_2</b>: %{y:.2f}<br>'+
        '<br><b>f_3</b>: %{z:.2f}<br>'+
        '<br><b>Unc_1</b>: %{customdata[0]}<br>'+
        '<br><b>Unc_2</b>: %{customdata[1]}<br>'+
        '<br><b>Unc_3</b>: %{customdata[2]}<br>'
        )],
        layout = Layout(
        showlegend=True,
        scene=Scene(
            xaxis=XAxis(title='f_1'),
            yaxis=YAxis(title='f_2'),
            zaxis=ZAxis(title='f_3')
        )
    ))
    plot(fig, filename= path + "/3Dscatter_" + str(iteration) + "_" + str(interaction_count) + ".html")

    ############### Plot scatter matrix with confidence intervals ######################
    cdata= np.hstack((unc3,objs1))
    fig1 = make_subplots(rows=1, cols=3)
    fig1.add_trace(go.Scatter(x=objs1[:,0], y=objs1[:,1],
                        error_x=dict(array=unc2[:,0], color='purple'),
                        error_y=dict(array=unc2[:,1],color='purple'),
                mode='markers',
        marker=dict(
            size=12,
            color=unc_avg, 
            opacity=0.8,
            colorscale=color_scale_custom,
            cmin=0,
            cmax=3
            ),

        customdata  = cdata,
        hovertemplate =
        '<br><b>f_1</b>: %{x:.2f}<br>'+
        '<br><b>f_2</b>: %{y:.2f}<br>'+
        '<br><b>f_3</b>: %{customdata[5]}}<br>'+
        '<br><b>Unc_1</b>: %{customdata[0]}<br>'+
        '<br><b>Unc_2</b>: %{customdata[1]}<br>'+
        '<br><b>Unc_3</b>: %{customdata[2]}<br>'
        ),row=1, col=1)
    fig1.update_xaxes(title_text="f_1", row=1, col=1)
    fig1.update_yaxes(title_text="f_2", row=1, col=1)

    fig1.add_trace(go.Scatter(x=objs1[:,1], y=objs1[:,2],
                        error_x=dict(array=unc2[:,1], color='purple'),
                        error_y=dict(array=unc2[:,2],color='purple'),
                mode='markers',
        marker=dict(
            size=12,
            color=unc_avg, 
            opacity=0.8,
            colorscale=color_scale_custom,
            cmin=0,
            cmax=3
            ),

        customdata  = cdata,
        hovertemplate =
        '<br><b>f_2</b>: %{x:.2f}<br>'+
        '<br><b>f_3</b>: %{y:.2f}<br>'+
        '<br><b>f_1</b>: %{customdata[3]}}<br>'+
        '<br><b>Unc_1</b>: %{customdata[0]}<br>'+
        '<br><b>Unc_2</b>: %{customdata[1]}<br>'+
        '<br><b>Unc_3</b>: %{customdata[2]}<br>'
        ),row=1, col=2)
    fig1.update_xaxes(title_text="f_2", row=1, col=2)
    fig1.update_yaxes(title_text="f_3", row=1, col=2)

    fig1.add_trace(go.Scatter(x=objs1[:,0], y=objs1[:,2],
                        error_x=dict(array=unc2[:,0], color='purple'),
                        error_y=dict(array=unc2[:,2],color='purple'),
                mode='markers',
        marker=dict(
            size=12,
            color=unc_avg, 
            opacity=0.8,
            colorscale=color_scale_custom,
            cmin=0,
            cmax=3
            ),

        customdata  = cdata,
        hovertemplate =
        '<br><b>f_1</b>: %{x:.2f}<br>'+
        '<br><b>f_3</b>: %{y:.2f}<br>'+
        '<br><b>f_2</b>: %{customdata[4]}}<br>'+
        '<br><b>Unc_1</b>: %{customdata[0]}<br>'+
        '<br><b>Unc_2</b>: %{customdata[1]}<br>'+
        '<br><b>Unc_3</b>: %{customdata[2]}<br>'
        ),row=1, col=3)
    fig1.update_xaxes(title_text="f_1", row=1, col=3)
    fig1.update_yaxes(title_text="f_3", row=1, col=3)

    plot(fig1, filename= path + "/3Dmatrix_" + str(iteration) + "_" + str(interaction_count) + ".html")
    print('Plotting done!!')

