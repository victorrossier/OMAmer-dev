import omamer.custom_seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


def prerec_plot(df, x, y, hue=None, row=None, col=None, style=None, height=13, aspect=1.2, font_scale=3.5, linewitdth=10, markersize=30, 
	hue_order=None, row_order=None, col_order=None, method2line=None, method2marker=None, colors=["#1b9e77", "#7570b3", "#d95f02"], 
	row_labels=None, col_labels=None, legend='brief', xticks=[], yticks=[], **kwargs):

	hue_nr = len(set(df[hue]))

	sns.set(style="whitegrid")                                                 
	sns.set_context("paper", rc = {'lines.linewidth': linewitdth, 'lines.markersize': markersize, 'grid.linewidth':3}, font_scale=font_scale)  

	rp = sns.relplot(data=df, x=x, y=y, hue=hue, row=row, col=col, style=style, hue_order=hue_order, row_order=row_order, col_order=col_order,
		dashes=method2line, markers=method2marker, height=height, aspect=aspect, kind="line", palette=colors[:hue_nr], legend=legend, **kwargs)

	rp.set(ylim=(0,1), xlim=(0,1), xticks=xticks, yticks=yticks)

	if legend:
		# remove legend titles
		rp._legend.texts[0].set_text("")
		if hue:
			rp._legend.texts[hue_nr + 1].set_text("")

	axes = rp.axes

	# write labels per row
	if row:
		if row_labels:
			[y.set_ylabel(row_labels[i]) for i, y in enumerate(axes[:,0])]
		else:
			[[x.set_ylabel(x.title.get_text().split('=')[1].lstrip()) for x in y] for y in axes]
	else:
		[[x.set_ylabel('') for x in y] for y in axes]

	# write labels per columns
	[[x.set_title('') for x in y] for y in axes]
	if col_labels:
		[y.set_title(col_labels[i]) for i, y in enumerate(axes[0,:])]

	# write x labels
	[[x.set_xlabel('')  for x in y] for y in axes]	

	return rp



def grid_barplot(x, y, df, hue=None, row=None, col=None, kind='bar', height=6, aspect=3, x_order=None, hue_order=None, row_order=None, col_order=None, 
	colors=["#1b9e77", "#7570b3", "#d95f02"], ylim=None, x_labels=None, row_labels=None, col_labels=None, ylabel="", font_scale=3, legend_y=1,  glinewidth=3, legend=True, **kwargs):

	sns.set(style='whitegrid') 
	sns.set_context("paper", rc = {'grid.linewidth':glinewidth}, font_scale=font_scale)  

	cp = sns.catplot(x=x, y=y, hue=hue, row=row, col=col, data=df, kind=kind, ci=None, height=height, aspect=aspect, 
		order=x_order, hue_order=hue_order, row_order=row_order, col_order=col_order, palette=colors, legend=False, **kwargs)

	cp.set(ylim=ylim)

	# bbox_to_anchor sets the relative coordinates of the legend. loc defines which location of the legend is placed at bbox_to_anchor "coordinates"
	if hue:
		if legend:
			plt.legend(bbox_to_anchor=(1,legend_y), loc='center left', frameon=False)

	# write labels per row
	axes = cp.axes
	if row:
		if row_labels:
			[y.set_ylabel(row_labels[i]) for i, y in enumerate(axes[:,0])]
		else:
			[[x.set_ylabel(x.title.get_text().split('=')[1].lstrip())  for x in y] for y in axes]
	else:
		axes[0,0].set_ylabel(ylabel)

	# write labels per columns
	[[x.set_title('')  for x in y] for y in axes]
	if col_labels:
		[y.set_title(col_labels[i]) for i, y in enumerate(axes[0,:])]

	# write x labels
	[[x.set_xlabel('')  for x in y] for y in axes]

	# reset global settings
	sns.set(font_scale=1, style=None) 

	return cp


def grid_barplot_colored(df, x, x_names, y, row=None, row_names=None, col=None, col_names=None, color_map='bwr', 
	                     ylim=(-1, 1), y_tick_interval=0.2):

	# build color map
	absmax = np.abs(df[y].values).max()
	norm = plt.Normalize(-absmax, absmax)
	cmap = plt.get_cmap(color_map)

	plt.style.use('seaborn-whitegrid')
	plt.rcParams.update({'font.size': 1})
	fig, axs = plt.subplots(len(row_names), len(col_names), sharey=True, sharex=True, gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, figsize=(8, 10))

	for i in range(axs.shape[0]):
	    
	    row_label = row_names[i]
	    
	    for j in range(axs.shape[1]):
	        
	        ax = axs[i, j]
	        ax.plot([-0.5, 2.5], [0, 0], '-', color='black')
	        
	        col_label = col_names[j]
	        
	        # get color
	        sub_df = df[(df[row] == row_label) & (df[col] == col_label)]
	        colors = cmap(norm( - sub_df[y].values))
	        
	        # plot bar
	        ax.bar(range(len(x_names)), "delta_value", data=sub_df, color=colors, width=0.8)
	        ax.set_ylim(ylim)
	        ax.set_xlim(-0.5, len(x_names) - 0.5)

	        ax.label_outer()

	# dim names
	for ax, col in zip(axs[0], col_names):
	    ax.set_title(col, fontsize=24)

	for ax, row in zip(axs[:,0], row_names):
	    ax.set_ylabel(row, rotation=90, fontsize=24)
	    plt.sca(ax)
	    plt.yticks(np.arange(ylim[0], ylim[1]+0.000001, y_tick_interval), fontsize=16)

	for ax in axs[2]:
	    plt.sca(ax)
	    plt.xticks(range(len(x_names)), x_names, fontsize=18, rotation=45)

	return fig