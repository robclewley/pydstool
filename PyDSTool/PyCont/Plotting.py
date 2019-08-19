""" Plotting class and function

    Drew LaMar, May 2006
"""


from PyDSTool.common import args
from PyDSTool.matplotlib_import import *
from functools import reduce

# THESE ARE REPEATS FROM CONTINUATION!  MAKE SURE AND UPDATE!!!
all_point_types = ['P', 'RG', 'LP', 'BP', 'H', 'BT', 'ZH', 'CP', 'GH', 'DH', 'LPC', 'PD',
                   'NS', 'MX', 'UZ', 'B']
all_curve_types = ['EP', 'LP', 'H', 'FP', 'LC']

#####
_classes = ['pargs', 'KeyEvent']

_functions = ['initializeDisplay']

__all__ = _classes + _functions
#####

class pargs(args):
    def __getitem__(self, key):
        if len(key.split(':')) > 1:
            (curve, pt) = key.split(':')
            return args.__getitem__(args.__getitem__(self, curve), pt)
        else:
            return args.__getitem__(self, key)

    def __repr__(self, ct=0, res=None):
        if res is None:
            res = []

        # This is so I can make things alphabetical
        all_pargs = []
        all_args = []
        deleted = 'deleted' in self and self.deleted
        if not deleted:
            for k, v in self.items():
                if k in ['point', 'curve', 'cycle']:
                    if v[0].get_label() != '_nolegend_':
                        all_args.append(('Legend', v[0].get_label()))
                elif k == 'text':
                    if v is not None:
                        all_args.append(('Label', v.get_text().lstrip(' ')))
                elif k == 'type':
                    all_args.append(('Type', self['type']))
                elif isinstance(v, pargs):
                    all_pargs.append((k, v))

        if len(all_args) == 0 and len(all_pargs) == 0:
            if 'deleted' in self:
                res.append(ct*4*' ' + 'Deleted\n')
            else:
                res.append(ct*4*' ' + 'Empty\n')
        else:
            all_args.sort()
            if len(all_args) > 0:
                for n in all_args:
                    res.append(ct*4*' ' + n[0] + ': ' + n[1] + '\n')

            all_pargs.sort()
            if len(all_pargs) > 0:
                ct += 1
                for n in all_pargs:
                    res.append((ct-1)*4*' ' + n[0] + ':' + '\n')
                    n[1].__repr__(ct=ct, res=res)

        if ct <= 1:
            return reduce(lambda x, y: x+y, res)

    __str__ = __repr__

    def fromLabel(self, label, key=None):
        point = None
        if 'text' in self and self.text.get_text().lstrip(' ') == label or \
           'point' in self and self.point[0].get_label() == label or \
           'curve' in self and self.curve[0].get_label() == label:
            return (key, self)
        for k, v in self.items():
            if isinstance(v, pargs):
                point = v.fromLabel(label, key=k)
                if point is not None:
                    break
        return point

    def get(self, objtype, bylabel=None, byname=None, bytype=None, bylegend=None, obj=None, ct=0):
        if objtype not in ['point', 'text', 'cycle', 'curve']:
            raise TypeError('Object type must be point, text, cycle, or curve')

        ct += 1
        if isinstance(bylabel, str):
            bylabel = [bylabel]
        if isinstance(byname, str):
            byname = [byname]
        if isinstance(bytype, str):
            bytype = [bytype]
        if isinstance(bylegend, str):
            bylegend = [bylegend]

        if obj is None:
            obj = []
        if bylabel is None and byname is None and bytype is None and bylegend is None:
            if objtype in ['point', 'text', 'cycle']:
                bytype = all_point_types
            elif objtype  == 'curve':
                bytype = all_curve_types
        for k, v in self.items():
            if isinstance(v, pargs):
                if objtype in v:
                    if bylabel is not None and 'text' in v and v.text.get_text().lstrip(' ') in bylabel or \
                       byname is not None and k in byname or \
                       bytype is not None and k.strip('0123456789') in bytype or \
                       bylegend is not None and (('curve' in v and v.curve[0].get_label() in bylegend) or ('cycle' in v and v.cycle[0].get_label() in bylegend)):
                           obj.append((k,v[objtype]))
                v.get(objtype, bylabel=bylabel, byname=byname, bytype=bytype, bylegend=bylegend, obj=obj, ct=ct)

        if ct == 1:
            return obj

    def toggleLabel(self, visible='on', refresh=True):
        if 'text' in self:
            self.text.set_visible(visible == 'on')

        if refresh:
            self.refresh()

    def toggleLabels(self, visible='on', bylabel=None, byname=None, bytype=None, ct=0):
        ct += 1
        if isinstance(bylabel, str):
            bylabel = [bylabel]
        if isinstance(byname, str):
            byname = [byname]
        if isinstance(bytype, str):
            bytype = [bytype]

        if bylabel is None and byname is None and bytype is None:
            bytype = all_point_types    # Turn off all points
        for k, v in self.items():
            if isinstance(v, pargs):
                if 'point' in v and (bytype is not None and k.strip('0123456789') in bytype or \
                   byname is not None and k in byname or \
                   bylabel is not None and v.text.get_text().lstrip(' ') in bylabel):
                    v.toggleLabel(visible=visible, refresh=False)
                else:
                    v.toggleLabels(visible=visible, bylabel=bylabel, byname=byname, bytype=bytype, ct=ct)

        if ct == 1:
            self.refresh()

    def togglePoint(self, visible='on', refresh=True):
        if 'point' in self:
            self.point[0].set_visible(visible == 'on')

        if refresh:
            self.refresh()

    def togglePoints(self, visible='on', bylabel=None, byname=None, bytype=None, ct=0):
        ct += 1
        if isinstance(bylabel, str):
            bylabel = [bylabel]
        if isinstance(byname, str):
            byname = [byname]
        if isinstance(bytype, str):
            bytype = [bytype]

        if bylabel is None and byname is None and bytype is None:
            bytype = all_point_types    # Turn off all points
        for k, v in self.items():
            if isinstance(v, pargs):
                if 'point' in v and (bytype is not None and k.strip('0123456789') in bytype or \
                   byname is not None and k in byname or \
                   bylabel is not None and v.text.get_text().lstrip(' ') in bylabel):
                    v.togglePoint(visible=visible, refresh=False)
                else:
                    v.togglePoints(visible=visible, bylabel=bylabel, byname=byname, bytype=bytype, ct=ct)

        if ct == 1:
            self.refresh()

    def toggleCurve(self, visible='on', refresh=True):
        if 'curve' in self:
            for k, v in self.items():
                if isinstance(v, pargs):
                    v.toggleLabel(visible=visible, refresh=False)
                    v.togglePoint(visible=visible, refresh=False)
            for line in self.curve:
                line.set_visible(visible == 'on')

        if refresh:
            self.refresh()

    def toggleCurves(self, visible='on', bylegend=None, byname=None, bytype=None, ct=0):
        ct += 1
        if isinstance(bylegend, str):
            bylegend = [bylegend]
        if isinstance(byname, str):
            byname = [byname]
        if isinstance(bytype, str):
            bytype = [bytype]

        if bylegend is None and byname is None and bytype is None:
            bytype = all_curve_types
        for k, v in self.items():
            if bytype is not None and isinstance(v, pargs) and 'type' in v and v.type in bytype or \
               byname is not None and k in byname or \
               bylegend is not None and isinstance(v, pargs) and 'curve' in v and v.curve[0].get_label() in bylegend:
                v.toggleCurve(visible=visible, refresh=False)
            elif isinstance(v, pargs):
                v.toggleCurves(visible=visible, bylegend=bylegend, byname=byname, bytype=bytype, ct=ct)

        if ct == 1:
            self.refresh()

    def toggleCycle(self, visible='on', refresh=True):
        if 'cycle' in self:
            for line in self.cycle:
                line.set_visible(visible == 'on')

        if refresh:
            self.refresh()

    def toggleCycles(self, visible='on', bylegend=None, byname=None, bytype=None, ct=0):
        ct += 1
        if isinstance(bylegend, str):
            bylegend = [bylegend]
        if isinstance(byname, str):
            byname = [byname]
        if isinstance(bytype, str):
            bytype = [bytype]

        if bylegend is None and byname is None and bytype is None:
            bytype = all_point_types
        for k, v in self.items():
            if bytype is not None and k.strip('0123456789') in bytype or \
               byname is not None and k in byname or \
               bylegend is not None and isinstance(v, pargs) and 'cycle' in v and v.cycle[0].get_label() in bylegend:
                v.toggleCycle(visible=visible, refresh=False)
            elif isinstance(v, pargs):
                v.toggleCycles(visible=visible, bylegend=bylegend, byname=byname, bytype=bytype, ct=ct)

        if ct == 1:
            self.refresh()

    def toggleAll(self, visible='on', bylabel=None, byname=None, bytype=None, ct=0):
        ct += 1
        if isinstance(bylabel, str):
            bylabel = [bylabel]
        if isinstance(byname, str):
            byname = [byname]
        if isinstance(bytype, str):
            bytype = [bytype]

        if bylabel is None and byname is None and bytype is None:
            bytype = all_point_types
        for k, v in self.items():
            if isinstance(v, pargs):
                if ('cycle' in v or 'point' in v) and (bytype is not None and k.strip('0123456789') in bytype or \
                   byname is not None and k in byname or \
                   bylabel is not None and (('point' in v and v.text.get_text().lstrip(' ') in bylabel) or ('cycle' in v and v.cycle[0].get_label() in bylabel))):
                    v.toggleLabel(visible=visible, refresh=False)
                    v.togglePoint(visible=visible, refresh=False)
                    v.toggleCycle(visible=visible, refresh=False)
                else:
                    v.toggleAll(visible=visible, bylabel=bylabel, byname=byname, bytype=bytype, ct=ct)

        if ct == 1:
            self.refresh()

    def setLabel(self, label, refresh=True):
        if 'text' in self:
            self.text.set_text('  '+label)

        if refresh:
            self.refresh()

    def setLabels(self, label, bylabel=None, byname=None, bytype=None, ct=0):
        ct += 1
        if isinstance(bylabel, str):
            bylabel = [bylabel]
        if isinstance(byname, str):
            byname = [byname]
        if isinstance(bytype, str):
            bytype = [bytype]

        if bylabel is None and byname is None and bytype is None:
            bytype = all_point_types    # Turn off all points
        for k, v in self.items():
            if bytype is not None and k.strip('0123456789') in bytype or \
               byname is not None and k in byname or \
               bylabel is not None and isinstance(v, pargs) and 'text' in v and v.text.get_text().lstrip(' ') in bylabel:
                v.setLabel(label, refresh=False)
            elif isinstance(v, pargs):
                v.setLabels(label, bylabel=bylabel, byname=byname, bytype=bytype, ct=ct)

        if ct == 1:
            self.refresh()

    def setLegend(self, legend, refresh=True):
        if 'curve' in self:
            for piece in self.curve:
                piece.set_label(legend)
        elif 'cycle' in self:
            for piece in self.cycle:
                piece.set_label(legend)

        if refresh:
            self.refresh()

    def setLegends(self, legend, bylegend=None, byname=None, bytype=None, ct=0):
        ct += 1
        if isinstance(bylegend, str):
            bylabel = [bylegend]
        if isinstance(byname, str):
            byname = [byname]
        if isinstance(bytype, str):
            bytype = [bytype]

        if bylegend is None and byname is None and bytype is None:
            if 'curve' in v:
                bytype = all_curve_types
            else:
                bytype = all_point_types    # Turn off all points
        for k, v in self.items():
            if bytype is not None and isinstance(v, pargs) and (('curve' in v and v.type in bytype) or ('cycle' in v and k.strip('0123456789') in bytype)) or \
               byname is not None and k in byname or \
               bylegend is not None and isinstance(v, pargs) and (('curve' in v and v.curve[0].get_label() in bylegend) or ('cycle' in v and v.cycle[0].get_label() in bylegend)):
                v.setLegend(legend, refresh=False)
            elif isinstance(v, pargs):
                v.setLegends(legend, bylegend=bylegend, byname=byname, bytype=bytype, ct=ct)

        if ct == 1:
            self.refresh()

    def refresh(self):
        fig = None  # Used to save current figure (reset when done)

        if 'deleted' not in self:
            if 'fig' in self.keys():
                plt.figure(self.fig.number)
                plt.draw()
            elif 'axes' in self.keys():
                plt.figure(self.axes.figure.number)
                #Would like to say: plt.axes(self.axes)  Doesn't work, though.
                plt.draw()
            elif 'curve' in self.keys():
                plt.figure(self.curve[0].figure.number)
                plt.draw()
            elif 'point' in self.keys():
                plt.figure(self.point[0].figure.number)
                plt.draw()
            elif 'cycle' in self.keys():
                plt.figure(self.cycle[0].figure.number)
                plt.draw()
            else:   # Only here in plot structure
                fig = plt.gcf()
                for k, v in self.items():
                    if isinstance(v, pargs):
                        v.refresh()

        if fig is not None: # Reset to current figure
            plt.figure(fig.number)

    def clean(self):
        """Cleans plotting structure (e.g. deleted)  Also gets rid of nonexisting figures
        if they've been deleted by an outside source.  This method is a little clumsy since
        it has to spawn a dummy plot if a plot has been deleted, but who cares, it works."""
        deleted = []
        for k, v in self.items():
            recursive_clean = True
            if isinstance(v, pargs):
                if 'deleted' in v:
                    deleted.append(k)
                    recursive_clean = False
                elif 'fig' in v:
                    fig_check = plt.figure(v.fig.number)
                    if fig_check != v.fig:
                        plt.close(fig_check)
                        deleted.append(k)
                        recursive_clean = False
                elif 'axes' in v:
                    try:
                        fig_axes = plt.axes(v.axes)
                    except:
                        fig_axes = None

                    if fig_axes is None:
                        deleted.append(k)
                        recursive_clean = False
                    elif fig_axes not in v.axes.figure.axes:
                        print('Warning:  Axes were deleted without using plt.delaxes().')
                        v.axes.figure.axes.append(fig_axes)
                        plt.draw()
                elif 'curve' in v:
                    for piece in v.curve:
                        if piece not in piece.axes.lines:
                            deleted.append(k)
                            v.delete()    # Remove remaining pieces of curve in pyplot
                            recursive_clean = False
                            break
                elif 'cycle' in v:
                    for piece in v.cycle:
                        if piece not in piece.axes.lines:
                            deleted.append(k)
                            v.delete()
                            recursive_clean = False
                            break
                elif 'point' in v:
                    if v.point[0] not in v.point[0].axes.lines or v.text not in v.text.axes.texts:
                        deleted.append(k)
                        if v.text in v.text.axes.texts:
                            v.text.axes.texts.remove(v.text)
                        if v.point[0] in v.point[0].axes.lines:
                            v.point[0].axes.lines.remove(v.point[0])
                        recursive_clean = False

                if recursive_clean:
                    v.clean()

        for k in deleted:
            self.pop(k)

    def clear(self, refresh=True):
        if 'deleted' not in self:
            if 'fig' in self:
                plt.figure(self.fig.number)
                plt.clf()
                remove = [k for k in self.keys() if k != 'fig']
            elif 'axes' in self:
                title = self.axes.title.get_text()
                self.axes.clear()
                plt.axes(self.axes)
                plt.title(title)
                remove = [k for k in self.keys() if k != 'axes']
                if refresh:
                    self.refresh()
            elif 'curve' in self:
                remove = [k for k in self.keys() if k != 'curve']
                for k in remove:
                    if isinstance(self[k], pargs):
                        self[k].clear(refresh=False)
                if refresh:
                    self.refresh()
            elif 'point' in self:
                if self.point[0] in self.point[0].axes.lines:
                    self.point[0].axes.lines.remove(self.point[0])
                if self.text in self.point[0].axes.texts:
                    self.point[0].axes.texts.remove(self.text)
                remove = [k for k in self.keys() if k != 'point']
                if refresh:
                    self.refresh()
                self.deleted = True
            else:
                remove = []

            if remove != []:
                for k in remove:
                    self.pop(k)
        else:
            print('Object is deleted.')

    def clearall(self):
        for v in self.values():
            if isinstance(v, pargs):
                v.clear(refresh=False)
        self.refresh()

    def delete(self, refresh=True):
        if 'deleted' not in self:
            if 'fig' in self:
                self.clear(refresh=False)
                plt.close()
                self.deleted = True
            elif 'axes' in self:
                self.clear(refresh=False)
                plt.delaxes()
                self.deleted = True
            elif 'curve' in self:
                self.clear(refresh=False)
                for curve in self.curve:
                    if curve in self.curve[0].axes.lines:
                        self.curve[0].axes.lines.remove(curve)
                if refresh:
                    self.refresh()
                self.deleted = True
            elif 'cycle' in self:
                self.clear(refresh=False)
                for cycle in self.cycle:
                    if cycle in self.cycle[0].axes.lines:
                        self.cycle[0].axes.lines.remove(cycle)
                if refresh:
                    self.refresh()
                self.deleted = True
            elif 'point' in self:
                self.clear(refresh=refresh)
            else:
                for v in self.values():
                    if isinstance(v, pargs):
                        v.delete(refresh=refresh)
        else:
            print('Object is already deleted.')

    def deleteall(self):
        for v in self.values():
            if isinstance(v, pargs):
                v.delete(refresh=False)
        self.refresh()

class KeyEvent(object):
    """Used in 'highlight' method of plot_cycles."""
    def __init__(self, paxes):
        self.curr = 0
        self.axes = paxes.axes

        # Get list of cycles from pargs class axes
        cycles = [thing.cycle[0] for k, thing in paxes.items() if isinstance(thing, pargs) and 'cycle' in thing]
        # Put in order as in axes.lines (this should be same as order along curve, or user
        #   specified in cycles when plot_cycles was called)
        self.cycles = []
        for line in self.axes.lines:
            if line in cycles:
                self.cycles.append(line)
        self.axes.set_title(self.cycles[0].get_label())

        self.bgd_lw = 0.2
        self.fgd_lw = 1.5
        for line in self.cycles:
            line.set(linewidth=self.bgd_lw)
        self.cycles[0].set(linewidth=self.fgd_lw)
        plt.draw()

        # NOT WORKING!
        # currfig = plt.gcf()
        # currfig.canvas.mpl_connect('key_press_event', self.__call__)
        plt.connect('key_press_event', self.__call__)

    def __call__(self, event):
        if event.key == 'right': self.change_curr(1)
        elif event.key == 'left': self.change_curr(-1)
        elif event.key == 'up': self.change_bgd(1)
        elif event.key == 'down': self.change_bgd(-1)
        elif event.key in ('+', '='): self.change_fgd(1)
        elif event.key in ('-', '_'): self.change_fgd(-1)

    def change_bgd(self, up):
        if up == 1:
            if self.bgd_lw+0.1 > self.fgd_lw:
                self.bgd_lw = self.fgd_lw
            else:
                self.bgd_lw += 0.1
        else:
            if self.bgd_lw-0.1 < 0:
                self.bgd_lw = 0
            else:
                self.bgd_lw -= 0.1

        for ct, line in enumerate(self.cycles):
            if ct != self.curr:
                line.set(linewidth=self.bgd_lw)
        plt.draw()

        print('Background linewidth = %f' % self.bgd_lw)

    def change_fgd(self, up):
        if up == 1:
            self.fgd_lw += 0.1
        else:
            if self.fgd_lw-0.1 < self.bgd_lw:
                self.fgd_lw = self.bgd_lw
            else:
                self.fgd_lw -= 0.1

        self.cycles[self.curr].set(linewidth=self.fgd_lw)
        plt.draw()

        print('Foreground linewidth = %f' % self.fgd_lw)

    def change_curr(self, up):
        self.cycles[self.curr].set(linewidth=self.bgd_lw)
        if up == 1:
            self.curr = (self.curr+1) % len(self.cycles)
        else:
            self.curr = (self.curr-1) % len(self.cycles)
        self.cycles[self.curr].set(linewidth=self.fgd_lw)
        self.axes.set_title(self.cycles[self.curr].get_label())
        plt.draw()

def initializeDisplay(plot, figure=None, axes=None):
    """If figure = 'new', then it will create a new figure with name fig#
    If figure is an integer, that figure # will be selected
    """

    plot.clean()   # Clean up plot structure
    plot._cfl = None
    plot._cal = None

    # Handle figure
    if figure is None:
        if len(plot) <= 3:
            figure = plt.gcf()
        else:
            raise ValueError('Please specify a figure.')

    cfl = None
    if isinstance(figure, plt.Figure):
        for k, v in plot.items():
            if isinstance(v, pargs) and v.fig == figure:
                cfl = k
                break
    elif isinstance(figure, str):
        if figure == 'new':
            figure = plt.figure()
        else:
            cfl = figure
            if cfl not in plot.keys():
                plot[cfl] = pargs()
                plot[cfl].fig = plt.figure()
    elif isinstance(figure, int):
        fighandle = plt.figure(figure)
        cfl = 'fig' + str(figure)
        if cfl not in plot.keys():
            plot[cfl] = pargs()
            plot[cfl].fig = fighandle
    else:
        raise TypeError("Invalid type for figure argument")

    if cfl is None:
        cfl = 'fig1'
        ct = 1
        while cfl in plot.keys():
            ct += 1
            cfl = 'fig'+repr(ct)
        plot[cfl] = pargs()
        plot[cfl].fig = figure

    plt.figure(plot[cfl].fig.number)

    # Handle axes
    if not axes:
        if len(plot[cfl]) <= 2:
            axes = plt.gca()
        else:
            raise ValueError('Please specify axes.')
    elif isinstance(axes, tuple):
        if len(axes) == 3:
            axes = plt.subplot(axes[0],axes[1],axes[2])
        else:
            raise TypeError('Tuple must be of length 3')

    cal = None
    if isinstance(axes, plt.Axes):
        for k, v in plot[cfl].items():
            if isinstance(v, pargs) and v.axes == axes:
                cal = k
                break

        if cal is None:
            cal = 'axes1'
            ct = 1
            while cal in plot[cfl].keys():
                ct += 1
                cal = 'axes'+repr(ct)
            plot[cfl][cal] = pargs()
            plot[cfl][cal].axes = axes
            plot[cfl][cal].axes.set_title(cal)
    elif isinstance(axes, str):
        cal = axes
        if cal not in plot[cfl].keys():
            plot[cfl][cal] = pargs()
            plot[cfl][cal].axes = plt.axes()
            plot[cfl][cal].axes.set_title(cal)

    plt.axes(plot[cfl][cal].axes)

    plot._cfl = cfl
    plot._cal = cal
