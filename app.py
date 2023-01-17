from PyQt5.QtCore import (Qt, QSortFilterProxyModel, pyqtSignal, pyqtSlot, 
QThread, QModelIndex, QAbstractTableModel, QVariant, QPoint)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QInputDialog,
    QFileDialog,
    QAbstractItemView,
    QErrorMessage,
    QMessageBox,
    QShortcut,
    QDialog,
    QLabel,
    QTreeWidgetItem
)
import pickle
from PyQt5.QtGui import QCursor
from PyQt5 import QtWidgets, uic, QtCore
#from pyqtgraph import PlotWidget
import os
import design
from tafel_app import Model
import numpy as np
from pyqtgraph import PlotWidget, mkPen
import pyqtgraph as pg
from DraggingHandler import DraggingScatter
import exception_hooks


def pretty_round(num):
    if np.isnan(num):
        return num
    if np.isinf(num):
        return num
    else:
        working = str(num-int(num))
        if np.sign(num)<0:
            n = 3
        else: 
            n = 2
        for i, e in enumerate(working[n:]):
            if e != '0':
                return round(num, i+1)
    


class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()][index.column()]
        
    @property
    def df(self):
        return self._data
        
    @df.setter
    def df(self, df):
        self.beginResetModel()
        self._data = df
        self.endResetModel()

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])
    
    def export(self):
        return self._data

class App(QMainWindow, design.Ui_MainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi('design.ui', self)
        self.actionLoad.triggered.connect(self.open_file)
        self.actionSave.triggered.connect(self.save)
        self.actionPlus.clicked.connect(self.plus)
        self.actionMinus.clicked.connect(self.minus)
        self.btnNewDataset.clicked.connect(self.load_dataset)
        self.btnExport.clicked.connect(self.export)
        self.model = Model()
        self.clickedPen = pg.mkPen('b', width=2)
        self.lastClicked = []
        self.info = TableModel([['', ''], ['', '']])
        self.output_widget.setModel(self.info)
        self.tafel_widget.setBackground('w')
        self.data = {}
        self.data_disp = {}
        self.u_saved = {}
        self.selected_data = ''
        self.input_tree.itemSelectionChanged.connect(self.data_selection)
        self.auto_selection()
        
        
    @pyqtSlot()      
    def load_dataset(self):
        fnames = QFileDialog.getOpenFileNames(self, 'Open file', os.getcwd())[0]
        for fname in fnames:
            self.model.load_data(fname)
            name = f'Data {len(self.data)+1}'
            self.data.update({name: [fname]})
            self.genDataDisp()
            self.upd_tree()
            self.auto_selection()
            self.u_saved.update({name: self.u})
            self.selected_data = name
        if len(fnames)>0:
            self.tafel()
        
    def save(self):
        name = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\', 'Tafel app files (*.tfl)')[0]
        with open(name, 'wb') as handle:
            pickle.dump([self.data, self.u_saved], handle, protocol=pickle.HIGHEST_PROTOCOL)
       
    def export(self):
        name = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\', 'Text files (*.txt)')[0]
        np.savetxt(name, self.info.export(), fmt="%s")
            
    def open_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'C:\\', 'Tafel app files (*.tfl)')[0]
        with open(fname, 'rb') as handle:
            self.data, self.u_saved = pickle.load(handle)
        self.selected_data = list(self.data.keys())[0]
        self.genDataDisp()
        self.upd_tree()
        self.do_select()
    
    def upd_tree(self):
        self.input_tree.clear()
        items = []
        for key, values in list(self.data_disp.items()):
            item = QTreeWidgetItem([key])
            for value in values:
                child = QTreeWidgetItem([value])
                child.setToolTip(0, value)
                item.addChild(child)
            items.append(item)
        self.input_tree.insertTopLevelItems(0, items)
        
    def data_selection(self):
        getSelected = self.input_tree.selectedItems()
        if getSelected:
            baseNode = getSelected[0]
            getChildNode = baseNode.text(0)
            if getChildNode:
                if ('Data ' in getChildNode):
                    self.selected_data = getChildNode
                    self.do_select()
                    # self.selected_higlight:
                      #  pass#self.tafel()
                else:
                    name = baseNode.parent().text(0)
                    self.selected_data = name
                    self.do_select()
                    
                    
                
                
    def do_select(self, auto_select=False):
        if len(self.data)>0:
            keys = list(self.data.keys())
            if not self.selected_data in keys:
                self.selected_data = keys[0]
            for i, fname in enumerate(self.data[self.selected_data]):
                if i==0:
                    self.model.load_data(fname)
                else:
                    self.model.add_data(fname)
            if auto_select:
                self.auto_selection()
            else:
                self.u = self.u_saved[self.selected_data]
            self.tafel()
        
    def auto_selection(self):
        u0 = self.model.xmin
        u3 = self.model.xmax
        u1 = u0+(u3-u0)/3
        u2 = u0+2*(u3-u0)/3
        self.u = np.array([u0, u1, u2, u3])
        self.u_saved[self.selected_data]=self.u
    
    def plus(self):
        self.add_file()
            
    def minus(self):
        getSelected = self.input_tree.selectedItems()
        if getSelected:
            auto_select = False
            baseNode = getSelected[0]
            getChildNode = baseNode.text(0)
            if getChildNode and ('Data ' in getChildNode):
                self.data.pop(getChildNode)
                self.data_disp.pop(getChildNode)
                self.u_saved.pop(getChildNode)
            else:
                dataset = baseNode.parent().text(0)
                i = self.data_disp[dataset].index(getChildNode)
                self.data[dataset].pop(i)
                self.data_disp[dataset].pop(i)
                if len(self.data[dataset]) == 0:
                    del self.data[dataset]
                    del self.data_disp[dataset]
                auto_select = True
            self.upd_tree()
            self.do_select(auto_select)
        
    def genDataDisp(self):
        self.data_disp = {}
        for name, fnames in self.data.items():
            self.data_disp.update({name: []})
            for fname in fnames:
                fname_disp = fname.split('/')[-1]
                self.data_disp[name].append(fname_disp)
            
        
    @pyqtSlot()      
    def add_file(self):
        if 'Data ' in self.selected_data:
            fname = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd())[0]
            if fname:
                self.model.add_data(fname)
                self.data[self.selected_data].append(fname)
                fname_disp = fname.split('/')[-1]
                self.data_disp[self.selected_data].append(fname_disp)
                self.upd_tree()
                #self.auto_selection()
                self.tafel()
            else:
                pass
        else:
            print("You don't select dataset!")
            self.open_file()
        
    def convert_u_from_slider(self, n):
        return n*self.model.dx/len(self.model.x) + self.model.xmin
        
    def convert_u_to_slider(self, us):
        q = np.zeros_like(us)
        for i, u in enumerate(us):
            q[i] = np.sum(u>=self.model.x)-1
            q[i] = max(i, q[i])
            q[i] = min(len(self.model.x)-1-(3-i), q[i])
        return q
    
    def on_change_selection(self):
        pos = self.selection.data['pos']
        pos = np.sort(np.transpose(pos)[0, :])
        if pos[0] < self.model.xmin or pos[-1] > self.model.xmax:
            return False
        dx = 2*self.model.dx/len(self.model.x)
        if abs(pos[0] - pos[1])<=dx:
            if pos[0]>self.model.xmin:
                pos[0] -= dx
            else:
                pos[1] += dx
        if abs(pos[2] - pos[3])<=dx:
            if pos[3]<self.model.xmax:
                pos[3] += dx
            else:
                pos[2] -= dx
        self.u = pos
        self.u_saved[self.selected_data] = self.u
        self.tafel(upd_range=False)
        return True

    def output(self):
        i0 = self.model.i_corr
        units_i = ['', 'm', 'u', 'n', 'p']
        n = -np.floor(np.log10(i0))
        if n <= 0:
           unit_i = units_i[0]
           i = i0
        elif n <= 12:
            k = int(1+n//3)
            unit_i = units_i[k]
            i = i0 * 10**(k*3)
        else:
            i = 0
            unit_i = units_i[-1]
            
        e0 = self.model.E_corr
        units_e = ['', 'm', 'u', 'n', 'p']
        n = -np.floor(np.log10(abs(e0)))
        if n <= 0:
           unit_e = units_e[0]
           e = e0
        elif n <= 12:
            k = int(1+n//3)
            unit_e = units_e[k]
            e = e0 * 10**(k*3)
        else:
            e = 0
            unit_e = units_e[-1]
        self.info.df = [
            ['E_corr', str(pretty_round(e))+unit_e+'V'],
            ['i_corr', str(pretty_round(i))+unit_i+'A'],
            ['alpha', str(pretty_round(self.model.alpha))],
            ['beta_1', str(pretty_round(self.model.a1))+'A/V'],
            ['beta_2', str(pretty_round(self.model.a2))+'A/V']
            ]
        
    def tafel(self, upd_range=True):
        i_s = self.convert_u_to_slider(self.u)
        i_s = list(map(int, i_s))
        self.tafel_widget.clear()
        x = self.model.x.reshape((-1,))
        ylog = self.model.ylog.reshape((-1,))
        ecorr, icorr, alpha, a1, b1, a2, b2 = self.model.tafel(*i_s)
        
        self.tafel_widget.plot(x, ylog, 
                               pen=mkPen('black', width=3))

        self.selection = DraggingScatter(x, ylog, self)
        self.tafel_widget.addItem(self.selection)
        pos = np.transpose(np.array([x[i_s], ylog[i_s]]))
        self.selection.setData(pos=pos, size=20, symbol='o', pxMode=True)
        
        self.tafel_widget.plot(x[i_s[0]:i_s[1]], ylog[i_s[0]:i_s[1]], symbol='o')
        self.tafel_widget.plot(x[i_s[2]:i_s[3]], ylog[i_s[2]:i_s[3]], symbol='o')

        self.tafel_widget.plot(x, a1*x+b1, 
                               pen=mkPen('r', width=3, style=QtCore.Qt.DashLine))
        self.tafel_widget.plot(x, a2*x+b2, 
                               pen=mkPen('r', width=3, style=QtCore.Qt.DashLine))
        if upd_range:
            self.tafel_widget.setYRange(ylog.min(), ylog.max(), padding=0.2)
            self.tafel_widget.setXRange(x.min(), x.max(), padding=0.2)
        
        self.output()
        
        
if __name__ == '__main__': 
    import sys
    app = QApplication(sys.argv)  
    pg.setConfigOptions(antialias=True)
    window = App() 
    window.show()  # Показываем окно
    sys.exit(app.exec_())