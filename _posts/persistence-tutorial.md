# Examples of Persistence Homology

Persistent homology is a method for computing topological features of a space at different spatial resolutions. The main way we do this is by plotting persistence diagrams. Specifically, we record the duration of the presence of each betti number, where the $n$-th Betti number $\beta_n$ to be the dimension of the $n$-th homology group. $\beta_0$ is the number of connected components, $\beta_1$ is the number of holes, $\beta_2$ the number of voids, etc. 


```python
from numpy import sin, cos, pi
from numpy.random import randint, normal
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
```

## Generate a noisy circle with $N$ points with radius $r$ 


```python
def noisyCircle(N, r, noise=0.1):
    t  = np.linspace(0,2*pi-2*pi/N, N)
    R= (r*normal(r,noise))
    X=np.array([[(np.sqrt(r*normal(r,noise)))*cos(t[i]), np.sqrt(r*normal(r,noise))*sin(t[i])] for i in range(N)])
    return X.reshape((N,2))
```


```python
def twoCircles(N, M, r1, r2, noise1=None, noise2 = None):
    if noise1 == None:
        noise1 = 0.2000
    if noise2 == None:
        noise2 = 0.25999
    X = noisyCircle(N, r1, noise1)
    Y = noisyCircle(M, r2, noise2)
    Z = np.concatenate([X,Y])
    fig = go.Figure(go.Scatter(x=X[:,0],y=X[:,1],mode="markers",
                               marker=dict(size=6,color='blue'),showlegend=False))
    fig.add_trace(
        go.Scatter(x=Y[:,0],y=Y[:,1], mode="markers",
                    marker=dict(size=6, color="red"),showlegend=False))

    fig.update_layout(title='Noisy Circles')
    fig.show()
    return Z
```


```python
Z = twoCircles(20, 20, 1, 3)
```


<div>                            <div id="361eb8bc-4363-4cc7-91b6-a90777d19984" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("361eb8bc-4363-4cc7-91b6-a90777d19984")) {                    Plotly.newPlot(                        "361eb8bc-4363-4cc7-91b6-a90777d19984",                        [{"marker": {"color": "blue", "size": 6}, "mode": "markers", "showlegend": false, "type": "scatter", "x": [0.9815650469285727, 0.9098622612722064, 0.8574866862782349, 0.5494186871099616, 0.31058908936866636, 6.222444736984153e-17, -0.35272722460755535, -0.6502782716949866, -0.8524253996765656, -0.9345997294567053, -0.7562344764993598, -0.9523227035552699, -0.8620442154013895, -0.5893818979166816, -0.2831543589062002, -2.168832995187417e-16, 0.3074734288217424, 0.6594518431260521, 0.8415198314603668, 0.9622765083726483], "y": [0.0, 0.24048069688208523, 0.5165826166237981, 0.8371137585346965, 0.9224852150328272, 0.9393631366485955, 0.7980752462773482, 0.8149596640932248, 0.5719681573885033, 0.2652442697183979, 1.1831997586876955e-16, -0.30538649235049214, -0.4693540849556535, -0.7423348203764689, -0.9563368726214926, -1.0180891593460706, -1.0680745907476084, -0.8123364083697028, -0.7061971593211014, -0.2951076754861022]}, {"marker": {"color": "red", "size": 6}, "mode": "markers", "showlegend": false, "type": "scatter", "x": [3.020778257154433, 2.97103602316235, 2.197671852512935, 1.6940764485861202, 0.9829766321263285, 1.829889209423244e-16, -0.8733412947810331, -1.7209321346265813, -2.3989300600216326, -2.9219963099917123, -2.911795065425067, -2.8323748794156143, -2.319570482994068, -1.7791848386130549, -0.907550180079903, -5.556179626912416e-16, 0.9428002337605418, 1.7964043178980023, 2.3608300919106533, 2.81104899487515], "y": [0.0, 0.9281532388163732, 1.7955909789959479, 2.443625547657271, 2.5862964933855004, 3.091150742876211, 2.9703170676852144, 2.2379054173708646, 1.723370528448093, 0.8717337384395544, 3.7771364076498397e-16, -0.8615683796163495, -1.6109119376127372, -2.2589710289085967, -2.940393300764349, -3.405663439623444, -2.4814241162149813, -2.5044582955659997, -1.84465284811355, -0.9166539996277376]}],                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Noisy Circles"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('361eb8bc-4363-4cc7-91b6-a90777d19984');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


### Generate $VR_{\epsilon}(Z)$



```python
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram

def VRpersistence(Z):
    VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2]) 
    return VR.fit_transform(Z[None,:,:])
```

Plot Persistence Diagram for Z


```python
diagrams=VRpersistence(Z)
plot_diagram(diagrams[0])
```


<div>                            <div id="9b7308f8-1012-4470-bd3b-800ee62c81f7" class="plotly-graph-div" style="height:500px; width:500px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("9b7308f8-1012-4470-bd3b-800ee62c81f7")) {                    Plotly.newPlot(                        "9b7308f8-1012-4470-bd3b-800ee62c81f7",                        [{"hoverinfo": "none", "line": {"color": "black", "dash": "dash", "width": 1}, "mode": "lines", "showlegend": false, "type": "scatter", "x": [-0.04311760425567627, 2.1989978170394897], "y": [-0.04311760425567627, 2.1989978170394897]}, {"hoverinfo": "text", "hovertext": ["(0.0, 0.18717792630195618)", "(0.0, 0.21074698865413666)", "(0.0, 0.25094273686408997)", "(0.0, 0.2536293864250183)", "(0.0, 0.2810257375240326)", "(0.0, 0.28980982303619385)", "(0.0, 0.2957373559474945)", "(0.0, 0.29802972078323364)", "(0.0, 0.31104734539985657)", "(0.0, 0.31150996685028076)", "(0.0, 0.31608280539512634)", "(0.0, 0.3175407946109772)", "(0.0, 0.31963837146759033)", "(0.0, 0.3629207909107208)", "(0.0, 0.37359359860420227)", "(0.0, 0.37997204065322876)", "(0.0, 0.3858279585838318)", "(0.0, 0.428458571434021)", "(0.0, 0.43507564067840576)", "(0.0, 0.7252709269523621)", "(0.0, 0.8207052946090698)", "(0.0, 0.8437992930412292)", "(0.0, 0.8511329889297485)", "(0.0, 0.8539147973060608)", "(0.0, 0.8652211427688599)", "(0.0, 0.8682854771614075)", "(0.0, 0.8717934489250183)", "(0.0, 0.8816608190536499)", "(0.0, 0.9080110788345337)", "(0.0, 0.9294852018356323)", "(0.0, 0.9403408765792847)", "(0.0, 0.9994416236877441)", "(0.0, 1.019864559173584)", "(0.0, 1.031445026397705)", "(0.0, 1.1050434112548828)", "(0.0, 1.106382966041565)", "(0.0, 1.1201951503753662)", "(0.0, 1.1621274948120117)", "(0.0, 1.5495796203613281)"], "mode": "markers", "name": "H0", "type": "scatter", "x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "y": [0.18717792630195618, 0.21074698865413666, 0.25094273686408997, 0.2536293864250183, 0.2810257375240326, 0.28980982303619385, 0.2957373559474945, 0.29802972078323364, 0.31104734539985657, 0.31150996685028076, 0.31608280539512634, 0.3175407946109772, 0.31963837146759033, 0.3629207909107208, 0.37359359860420227, 0.37997204065322876, 0.3858279585838318, 0.428458571434021, 0.43507564067840576, 0.7252709269523621, 0.8207052946090698, 0.8437992930412292, 0.8511329889297485, 0.8539147973060608, 0.8652211427688599, 0.8682854771614075, 0.8717934489250183, 0.8816608190536499, 0.9080110788345337, 0.9294852018356323, 0.9403408765792847, 0.9994416236877441, 1.019864559173584, 1.031445026397705, 1.1050434112548828, 1.106382966041565, 1.1201951503753662, 1.1621274948120117, 1.5495796203613281]}, {"hoverinfo": "text", "hovertext": ["(2.039213180541992, 2.046281099319458)", "(1.9504562616348267, 1.9807416200637817)", "(1.9276453256607056, 1.9359697103500366)", "(1.8525546789169312, 1.906677007675171)", "(1.8513609170913696, 2.066493511199951)", "(1.7945398092269897, 2.124569892883301)", "(1.78075110912323, 2.0799880027770996)", "(1.3202615976333618, 2.1558802127838135)", "(0.4445740580558777, 1.653116226196289)"], "mode": "markers", "name": "H1", "type": "scatter", "x": [2.039213180541992, 1.9504562616348267, 1.9276453256607056, 1.8525546789169312, 1.8513609170913696, 1.7945398092269897, 1.78075110912323, 1.3202615976333618, 0.4445740580558777], "y": [2.046281099319458, 1.9807416200637817, 1.9359697103500366, 1.906677007675171, 2.066493511199951, 2.124569892883301, 2.0799880027770996, 2.1558802127838135, 1.653116226196289]}, {"hoverinfo": "text", "hovertext": ["(1.7100536823272705, 1.7377995252609253)"], "mode": "markers", "name": "H2", "type": "scatter", "x": [1.7100536823272705], "y": [1.7377995252609253]}],                        {"height": 500, "plot_bgcolor": "white", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "width": 500, "xaxis": {"autorange": false, "exponentformat": "e", "linecolor": "black", "linewidth": 1, "mirror": false, "range": [-0.04311760425567627, 2.1989978170394897], "showexponent": "all", "showline": true, "side": "bottom", "ticks": "outside", "title": {"text": "Birth"}, "type": "linear", "zeroline": true}, "yaxis": {"autorange": false, "exponentformat": "e", "linecolor": "black", "linewidth": 1, "mirror": false, "range": [-0.04311760425567627, 2.1989978170394897], "scaleanchor": "x", "scaleratio": 1, "showexponent": "all", "showline": true, "side": "left", "ticks": "outside", "title": {"text": "Death"}, "type": "linear", "zeroline": true}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('9b7308f8-1012-4470-bd3b-800ee62c81f7');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


What can we
