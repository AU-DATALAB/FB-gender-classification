     
# %%

# https://www.datatechnotes.com/2020/11/tsne-visualization-example-in-python.html
# Good page for general plot knowledge https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html
# https://stackoverflow.com/questions/59567330/scatter-plot-with-different-colors-and-labels
# https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
import seaborn as sns
import matplotlib.pyplot as plt
#import argparse
import pandas as pd
import numpy as np
from numpy.core.fromnumeric import argmax
from sklearn.manifold import TSNE

def plot_tsne(data, condition = ['center', 'sample'], n=2000):
    x = data.iloc[:,:-1].values
    y = pd.DataFrame(x).idxmax(axis=1)

    tsne = TSNE(n_components=2, verbose=1, random_state=2022)
    z = tsne.fit_transform(x) 

    # Topic number
    df = pd.DataFrame({'y' : y})

    # NB 'first' dataframe is only used to maually label topics in dict
    # Add size value
    df['size']=df.groupby('y')['y'].transform(len)
    first = df.groupby('y').first().reset_index()
    # Take first row from every group
    first = first.sort_values('size', ascending=False)
    first['new_topics'] = range(0,50)
    # Mapping new labels that are made according to size 
    df['y']=df['y'].map(first['new_topics'])
    
    # Get 50 values 
    import matplotlib.colors as colors
    cmap = plt.cm.get_cmap('Spectral', 50)
    hex = list()
    for number in range(cmap.N):
        rgba = cmap(number)
        # rgb2hex accepts rgb or rgba
        hex.append(colors.rgb2hex(rgba))

    color = list()
    for i in df.y:
        color.append(hex[i])
        
    # Adding hex values as column
    df['color'] = color
   # df['color'].sort_values('y') 
    # Add correct topic label
    df["y"] = df['y']+1

    # Adding topic labs
    labs = {6:"Municipality",49:"Trading; purchase",14:"Support",40:"Gardening", 19:"Tradi6ng; clothes",36:"Planning appointments", 4:"Trading; children related",42:"Fans",7:"Sport",28:"Job",29:"Photography and drawing",26:"Cooking",34:"Horseback riding",17:"National politics",27:"Movies/theater",8:"History",21:"Dating",9:"Trip",38:"Hunting", 23:"Parenting; school",10:"Trading; selling",44:"Pets",25:"Politics", 22:"Mental wellbeing",24:"Business",48:"Betting",46:"Security",16:"Socializing with family and friends",47:"Programming language",31:"Accommodation",18:"Interchange",11:"University",35:"Transportation",2:"Study support",37:"Food",12:"Socializing as couple",30:"News",13:"Workshop and courses",43:"Software",45:"Book trading",32:"Diseases and medical help",1:"Celebrations",39:"Parenting; arrangements",15:"School",20:"Music",50:"Gaming",5:"Group communication",41:"Medical help",33:"Dieting",3:"Programming"}

    df['lab'] = (df['y'].astype(str) + " : " + df['y'].map(labs))

    # Components from tsne
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    # Taking mean component coordinates
    mean_components = df.groupby("y", as_index=False).mean()

    # Figure
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(111)

    df = df.sort_values('y')

    for color in df.color.unique():
        df_color = df[df.color==color]
        plt.scatter(x=df_color["comp-1"],
                    y=df_color["comp-2"], 
                    edgecolor='darkgrey', 
                    alpha=0.7,
                    c=color, 
                    label=df_color.lab.iloc[0])
    

    bbox_props = dict(boxstyle="circle,pad=0.3", fc="white", ec="black", lw=2, alpha=0.9)
    # Using mean component
    if condition == 'center':
        for i,xy in enumerate(zip(mean_components["comp-1"], mean_components["comp-2"])):
            ax.annotate(mean_components['y'].values[i], xy=xy, xytext=xy,  ha='center', va='center', bbox=bbox_props)
    elif condition == 'sample':
        sample = df.sample(n)
        for i,xy in enumerate(zip(sample["comp-1"], sample["comp-2"])):
            ax.annotate(sample['y'].values[i], xy=xy, xytext=xy)


    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend(bbox_to_anchor=(1, 1), title = "Topic number content")

    plt.savefig(f"tsne_topics_{condition}_sizeordered_labs.jpg", dpi=400, bbox_inches="tight")
    return(print(f"Figure is saved!"))

data = pd.read_csv("doc_theta_50.csv")
plot_tsne(data=data, condition='center')
plot_tsne(data=data, condition='sample')


# %%
