import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm, kstest, levy_stable
import itertools
import plotly_express as px
import matplotlib.pyplot as plt 
from scipy import stats

def plot_weight_hist_plotly(model):
    fig = go.Figure()
    colors = itertools.cycle(px.colors.qualitative.Plotly)
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            color = next(colors)  
            data = param.detach().cpu().numpy().flatten()
            
            # hist plot
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=50,
                name=name,
                opacity=0.6,
                marker_color=color
            ))
            
            # normal distrubution
            mu, std = np.mean(data), np.std(data)
            x = np.linspace(data.min(), data.max(), 200)
            y = norm.pdf(x, mu, std)
            y_scaled = y * len(data) * (x[1]-x[0])
            
            fig.add_trace(go.Scatter(
                x=x, y=y_scaled,
                mode='lines',
                name=f"{name} normal distribution",
                line=dict(color=color, width=2)
            ))

    fig.update_layout(
        title="Distribution of weights vs normal distribution for each layer",
        xaxis_title="values",
        yaxis_title="count",
        barmode='overlay'
    )
    fig.show()


def plot_weight_heatmap(layer: str, model=None, weights: np.ndarray = None):
    """
    plot heatmap of weights according to layer name or np.ndarray

    Parameter:
    ----------
    model : torch.nn.Module, optional
        (trained) pytorch model, only necessray if no weight array is provided
    layer : str
        layer name
    weights : np.ndarray, optional
        array storing the weiights. If provided/layer, model will be ignored
    """

    if weights is not None:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        fig = px.imshow(
            weights,
            color_continuous_scale='Viridis',
            title= f"Heatmap of weights for {layer}",
            labels={'x': 'Input Features', 'y': 'Output Features', 'color': 'Gewicht'}
        )
        fig.show()
        return
    
    if model is None:
        raise ValueError("No model or weight array provided")

    found = False
    for name, param in model.named_parameters():
        if layer in name:
            found = True
            w = param.detach().cpu().numpy()
            fig = px.imshow(
                w,
                color_continuous_scale='Viridis',
                title=f"Heatmap of weights for {layer}",
                labels={'x': 'Input Features', 'y': 'Output Features', 'color': 'Gewicht'}
            )
            fig.show()

    if not found:
        print(f"No layer named {layer}")


def log_log_plot(data=None, title:str="", bins=50, x_range=(1e-3, 1e1), grid=False, opacity=1):
    """
    Plots a log-log histogram of the absolute values of data or a theoretical probability density function (PDF).

    Parameters
    ----------
    data : array-like, optional
        The empirical data to plot. If None, a theoretical distribution is plotted instead.
    title : str
        Title of the plot.
    bins : int
        Number of histogram bins.
    x_range : tuple
        Logarithmic range (min, max) for the x-axis.
    """

    data = np.abs(np.asarray(data))
    plt.hist(data, bins=np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), bins), 
                density=True, alpha=opacity)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('|Value|')
    plt.ylabel('Density')
    plt.title(f'Log-Log Plot: {title}')
    plt.legend()
    if grid: 
        plt.grid(True, which="both", ls="--", lw=0.5)
    plt.show()

def compare_stable_vs_gaussian(weights: np.ndarray):
    """
    1. fits a gaussian distribution to the input data
    2. fits a levy stable distribution to the input data

    perform a KS Test for both fits with the empirical data

    devide the p value of the stable fit by the p value of the gaussian fit: 
        r > 1 -> stable is a better fit
        r < 1 -> gaussian is a better fit

    Args:
        weights (np.ndarray): weight matrix of a layer
    """

    alpha, beta, sigma, mu = levy_stable.fit(weights.flatten())
    muN, sigN = norm.fit(weights.flatten())

    p_stable = kstest(weights.flatten(), 'levy_stable', args=(alpha, beta, sigma, mu))[1]
    p_gauss = kstest(weights.flatten(), 'norm', args=(muN, sigN))[1]

    r = p_stable / p_gauss

    return r

