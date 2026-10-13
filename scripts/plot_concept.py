"""Local conceptual SVG fallback after Whimsical quota failure; no measured claims."""
import pathlib,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=pathlib.Path(__file__).resolve().parents[1]
fig,ax=plt.subplots(figsize=(12,3.8));ax.set_xlim(0,12);ax.set_ylim(0,4);ax.axis('off')
labels=[('Current character\nFixed RF code',.1,1.3),('4 RF inputs\nBandwidth + limit',2.4,1.3),('YIG wave state\n32 spatial modes',4.7,1.3),('4 I/Q receivers\n16 observations',7.0,1.3),('Linear + softmax\nNext character',9.3,1.3)]
for text,x,y in labels:
    ax.add_patch(FancyBboxPatch((x,y),2.0,.95,boxstyle='round,pad=0.08',facecolor='#edf3f7',edgecolor='#35617c'))
    ax.text(x+1,y+.47,text,ha='center',va='center',fontsize=10)
for x in (2.15,4.45,6.75,9.05):ax.annotate('',xy=(x+.2,1.78),xytext=(x-.1,1.78),arrowprops=dict(arrowstyle='->',color='#35617c'))
ax.text(5.7,3.35,'8 DC coil currents: bounded trainable physical weights',ha='center',fontsize=11,color='#b54c37')
ax.annotate('',xy=(5.7,2.3),xytext=(5.7,3.0),arrowprops=dict(arrowstyle='->',color='#b54c37'))
ax.annotate('Cross-entropy gradient through causal dynamics',xy=(5.7,.85),xytext=(10.3,.35),ha='center',fontsize=10,color='#b54c37',arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-.15',color='#b54c37'))
ax.set_title('Conceptual architecture - not a measured hardware layout',fontsize=13)
(ROOT/'results/aggregate').mkdir(exist_ok=True)
fig.savefig(ROOT/'results/aggregate/concept.svg',bbox_inches='tight')
fig.savefig(ROOT/'results/aggregate/concept.png',dpi=180,bbox_inches='tight')
