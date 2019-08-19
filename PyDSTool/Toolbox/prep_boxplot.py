
from PyDSTool import remain, loadObjects, array, save_fig, arange, args
from matplotlib.font_manager import FontProperties
from PyDSTool.matplotlib_import import *
from scipy import mean

##symbol_map = {'isomap': {
##                    'K': {10: 'k<', 50: 'k>', 100: 'k^', 500: 'kv'},
##                    'eps': {20: 'w<', 50: 'w>', 100: 'w^'}
##                        },
##              'pca': {
##                  'knee': {1: 'ws', 2: 'ks'},
##                  'var': {80: 'wo', 90: 'ko'}
##                     }
##              }

symbol_map = {'isomap': {
                    'K': {10: ('ws', 'E'), 50: ('ws', 'F'),
                          100: ('ws', 'G'), 500: ('ws', 'H')},
                    'eps': {20: ('ws', 'I'), 50: ('ws', 'J'),
                            100: ('ws', 'K')}
                        },
              'pca': {
                  'knee': {1: ('wo', 'A'), 2: ('wo', 'B')},
                  'var': {80: ('wo', 'C'), 90: ('wo', 'D')}
                     }
              }


# For use with PD-E analysis
def prep_boxplots(data, xlabel_str, figname='', fignum=1, do_legend=1,
                  means=1, xlegoff=0, ylegstep=1, ylegoff=0, spacing=None):
    spacing_actual = {
        'width': 0.1,
        'wgapfac': 0.75,
        'markersize': 12,
        'off_fac': 0.7,
        'x_step': 0.9,   # 9 * width
        'x_off': 0,
        'box_to_marker': 1.1,
        'notch_size': 0.2}
    if spacing is not None:
        spacing_actual.update(spacing)
    width = spacing_actual['width']
    wgapfac = spacing_actual['wgapfac']
    markersize = spacing_actual['markersize']
    off_fac = spacing_actual['off_fac']
    x_step = spacing_actual['x_step']
    x_off = spacing_actual['x_off']
    box_to_marker = spacing_actual['box_to_marker']
    notch_size = spacing_actual['notch_size']

    n = len(data)
    x_min = -width*3.8 #3.75
    x_max = (n-1)*x_step+width*4.5 #3.75
    if n > 1:
        halfticks = arange(1,n)*x_step-x_step/2
    figure(fignum)
    # work out ordering of data from 'pos' key
    order = {}
    # `pos` position runs from 1 to n, `ns` runs from 0 to n-1
    ns = []
    for k, v in data.iteritems():
        order[v['pos']] = k
        ns.append(v['pos']-1)
    ns.sort()
    assert ns == range(n)
    maxD = 0
    max_dimval_markers = 0
    labels = []
    for pos in range(n):
        name = order[pos+1]
        pde_name = 'PD_E-'+name
        if 'known_dim' in data[name]:
            if n == 1:
                kdx1 = x_min
                kdx2 = x_max
            else:
                if pos == 0:
                    kdx1 = x_min
                    kdx2 = halfticks[0]
                elif pos == n-1:
                    kdx1 = halfticks[n-2]
                    kdx2 = x_max
                else:
                    kdx1 = halfticks[pos-1]
                    kdx2 = halfticks[pos]
            plot([[kdx1], [kdx2]],
                 [data[name]['known_dim'],data[name]['known_dim']],
                 'k', linewidth=1, zorder=0)
        slope_data = loadObjects(pde_name)[2]
        ds_mins = array(slope_data[:,0])#,shape=(len(slope_data),1))
        ds_mins.shape=(len(slope_data),1)
        ds_maxs = array(slope_data[:,1])#,shape=(len(slope_data),1))
        ds_maxs.shape=(len(slope_data),1)
        max_ds = max([max(ds_mins[:,0]),max(ds_maxs[:,0])])
        if max_ds > maxD:
            maxD = max_ds
        # limits args are ineffective here
        boxplot(ds_mins,positions=[pos*x_step-width*wgapfac+x_off],whis=100,
                means=means,monochrome=True,notch=2,notchsize=notch_size,
                limits=(),widths=width,fill=1)
        boxplot(ds_maxs,positions=[pos*x_step+width*wgapfac+x_off],whis=100,
                means=means,monochrome=True,notch=2,notchsize=notch_size,
                limits=(),widths=width,fill=1)
        if pos == 0:
            fa = figure(fignum).axes[0]
            fa.hold(True)
        if means:
            ds_all_mean = (mean(ds_mins[:,0])+mean(ds_maxs[:,0]))/2
            plot([pos*x_step+x_off], [ds_all_mean], 'k^',
                 markersize=markersize-2)
        pca_x = pos*x_step-width*(wgapfac+box_to_marker)+x_off
        isomap_x = pos*x_step+width*(wgapfac+box_to_marker)+x_off
        pca_ds = {}
        isomap_ds = {}
        try:
            pca_data = data[name]['pca']
        except KeyError:
            pca_data = []
        pca_ds, max_dimval_pca, pca_used = plot_markers(pca_data,
                                          pca_x, 'PCA',
                                          symbol_map['pca'], -1,
                                          width, off_fac, markersize)
        if max_dimval_pca > maxD:
            maxD = max_dimval_pca
        if max_dimval_pca > max_dimval_markers:
            max_dimval_markers = max_dimval_pca
        try:
            isomap_data = data[name]['isomap']
        except KeyError:
            isomap_data = []
        isomap_ds, max_dimval_iso, isomap_used = plot_markers(isomap_data,
                                             isomap_x, 'Isomap',
                                             symbol_map['isomap'], 1,
                                             width, off_fac, markersize)
        if max_dimval_iso > maxD:
            maxD = max_dimval_iso
        if max_dimval_iso > max_dimval_markers:
            max_dimval_markers = max_dimval_iso
        labels.append(data[name]['label'])
    ## legend
    if do_legend:
        font = FontProperties()
        font.set_family('sans-serif')
        font.set_size(11)
        x_legend = x_min + 3*width/4 + xlegoff
        y_legend = maxD+ylegoff
        # pca legend
        for k, s in pca_used:
            plot_markers([(k,s,y_legend)], x_legend, 'Legend', symbol_map['pca'],
                     1, width, off_fac, markersize)
            if k == 'var':
                legstr = "%s=%d%%"%(k,s)
            else:
                legstr = "%s=%d"%(k,s)
            text(x_legend+3*width/4, y_legend-width*2., legstr,
                 fontproperties=font)
            y_legend -= ylegstep
        # isomap legend
        isomap_leg_data = []
        for k, s in isomap_used:
            if y_legend-width*2. <= max_dimval_markers + 2:
                y_legend = maxD+ylegoff
                x_legend += x_step #-width*.75
            plot_markers([(k,s,y_legend)], x_legend, 'Legend', symbol_map['isomap'],
                      1, width, off_fac, markersize)
##        if k == 'eps':
##            kstr = '\\epsilon'
##        else:
##            kstr = k
            text(x_legend+3*width/4, y_legend-width*2., "%s=%d"%(k,s),
                 fontproperties=font)
            y_legend -= ylegstep
    ## tidy up axes, etc.
    fa.set_xticks(arange(n)*x_step)
    if n>1:
        for h in range(n-1):
            plot([halfticks[h], halfticks[h]], [0,maxD+1+ylegoff], 'k:')
    fa.set_xticklabels(labels)
    fa.set_position([0.07, 0.11, 0.9, 0.85])
    fa.set_xlim(x_min,x_max)
    fa.set_ylim(0,maxD+1+ylegoff)
    if xlabel_str != '':
        xlabel(r'$\rm{'+xlabel_str+r'}$',args(fontsize=20,fontname='Times'))
    ylabel(r'$\rm{Dimension}$',args(fontsize=20,fontname='Times'))
    draw()
    if figname != '':
        save_fig(fignum, figname)


def plot_markers(data, x_base, name, map, xoff_sgn, width, off_fac,
                 markersize):
    maxD = 0
    ds = {}
    used = []
    font = FontProperties()
    font.set_family('sans-serif')
    font.set_size(10)
    for (kind, subkind, dimval) in data:
        try:
            symb, lab = map[kind][subkind]
        except KeyError:
            raise KeyError("Invalid key for %s symbols"%name)
        used.append((kind, subkind))
        try:
            ds[dimval] += 1
            x_off = xoff_sgn*width*off_fac*(ds[dimval]-1)
        except KeyError:
            ds[dimval] = 1
            x_off = 0
        plot([x_base+x_off], [dimval], symb, markersize=markersize)
        # hack tweaks
    ##            if lab=='C':
    ##                x_off -= width/15
        if lab=='A':
            x_off += width/30
        text(x_base+x_off-width*.15, dimval-width*2., lab,
             fontproperties=font)
        if dimval > maxD:
            maxD = dimval
    return ds, maxD, used
