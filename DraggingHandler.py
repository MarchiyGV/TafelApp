"""
Dragging event handler
"""
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np

class DraggingScatter(pg.GraphItem):
    def __init__(self, xs, ys, parent):
        self.xs = xs
        self.ys = ys
        self.parent = parent
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.clicked)
        
    def setData(self, **kwds):
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.updateGraph()

        
    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i,item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])
        
        
    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        
        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first 
            # pressed:
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - pos
            self.flag = True
        elif ev.isFinish():
            self.dragPoint = None
            self.parent.on_change_selection()
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return
        
        ind = self.dragPoint.data()[0]
        dx = self.dragOffset[0]
        x0, y0 = ev.pos()
        x1 = x0+dx
        i1 = np.sum(self.xs<=x1)-1
        y1 = self.ys[i1]
        r1 = np.array([x1, y1])
        self.data['pos'][ind] = r1
        self.updateGraph()
        ev.accept()
        
    def clicked(self, pts):
        print("clicked: %s" % pts)


if __name__ == '__main__':
    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)

    w = pg.GraphicsLayoutWidget(show=True)
    w.setWindowTitle('pyqtgraph example: CustomGraphItem')
    v = w.addViewBox()
    v.setAspectLocked()

    g = DraggingScatter()
    v.addItem(g)

    ## Define positions of nodes
    pos = np.array([
        [0,0],
        [10,0],
        [0,10],
        [10,10],
        [5,5],
        [15,5]
        ], dtype=float)
        

    ## Update the graph
    g.setData(pos=pos, size=1, symbol='o', pxMode=False)
    pg.exec()
