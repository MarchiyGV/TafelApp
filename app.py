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
import pandas as pd
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
#import exception_hooks


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
    
class DialogNewP(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi('dialog_new_project.ui', self)
        self.open_path_btn.clicked.connect(self.open_folder)
        
    def open_folder(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.ppath.setText(folder)


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
        self.actionNew.triggered.connect(self.new_project)
        self.actionLoad.triggered.connect(self.open_file)
        self.actionSave.triggered.connect(self.save)
        self.actionPlus.clicked.connect(self.add_to_dataset)
        self.actionMinus.clicked.connect(self.minus)
        self.btnNewDataset.clicked.connect(self.load_dataset)
        self.btnExport.clicked.connect(self.export)
        self.model = Model(self)
        self.clickedPen = pg.mkPen('b', width=2)
        self.lastClicked = []
        self.info = TableModel([['', ''], ['', '']])
        self.output_widget.setModel(self.info)
        self.tafel_widget.setBackground('w')
        self.datasets = pd.DataFrame(columns=['name', 'data', 'data_disp', 'u_saved'])
        self.selected_data = ''
        self.input_tree.itemSelectionChanged.connect(self.data_selection)
        self.input_tree.itemChanged[QTreeWidgetItem, int].connect(self.rename_dataset)
        self.auto_selection()
        self.pname = ''
        self.ppath = ''
        self.btns_enabled(False)
        
    
    def btns_enabled(self, flag):
        self.actionSave.setEnabled(flag)
        self.actionPlus.setEnabled(flag)
        self.actionMinus.setEnabled(flag)
        self.btnNewDataset.setEnabled(flag)
        self.btnExport.setEnabled(flag)
        
        
    def rename_dataset(self, item, column):    
        new_name = item.text(0)
        
        if np.any(new_name==self.datasets['name']):
            old_name = self.datasets['name'][self.selected_data].values[0]
            self.input_tree.blockSignals(True)
            item.setText(0, old_name)
            self.input_tree.blockSignals(False)
            self.error_msg(f'Dataset with name "{new_name}" is already exist!')
            return
            #raise ValueError(f'Dataset with name "{new_name}" is already exist!')
            
        self.datasets['name'][self.selected_data] = new_name
        
    def error_msg(self, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error:")
        msg.setInformativeText(text)
        msg.setWindowTitle("Error")
        msg.exec_()
    
    @pyqtSlot()      
    def new_project(self):
        dlg = DialogNewP(self)
        dlg.exec()
        if dlg.result():
            self.pname = dlg.pname.text()
            self.ppath = dlg.ppath.text()
            try:
                os.mkdir(f'{self.ppath}/{self.pname}')
            except FileExistsError:
                self.error_msg(f'Project "{self.ppath}/{self.pname}" is already exists!')
                return 
            self.btns_enabled(True)
        
    
    @pyqtSlot()      
    def load_dataset(self):
        fnames = QFileDialog.getOpenFileNames(self, 'Open file', os.getcwd())[0]
        for fname in fnames:
            new_fname = self.model.load_data(fname, )
            name = f'Data {len(self.datasets)+1}'
            self.auto_selection()
            data_disp = new_fname.split('/')[-1]
            row = pd.DataFrame({'name': [name], 
                                  'data': [[new_fname]], 
                                  'data_disp':[[data_disp]], 
                                  'u_saved': [np.array(self.u)]})
            self.datasets = pd.concat([self.datasets, row], ignore_index = True)
            self.upd_tree()
            self.selected_data = (self.datasets['name']==name)
        if len(fnames)>0:
            self.tafel()
        
        
    @pyqtSlot()      
    def add_to_dataset(self):
        if np.any(self.selected_data):
            fname = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd())[0]
            if fname:
                new_fname = self.model.add_data(fname)
                self.datasets['data'][self.selected_data][0].append(new_fname)
                fname_disp = new_fname.split('/')[-1]
                self.datasets['data_disp'][self.selected_data][0].append(fname_disp)
                self.upd_tree()
                self.tafel()
            else:
                pass
        else:
            print("You don't select dataset!")
            
    def save(self):
        name = f'{self.ppath}/{self.pname}/project.tfl'
        with open(name, 'wb') as handle:
            pickle.dump((self.datasets), handle, protocol=pickle.HIGHEST_PROTOCOL)
       
    def open_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'C:\\', 'Tafel app files (*.tfl)')[0]
        if fname:
            with open(fname, 'rb') as handle:
                self.datasets = pickle.load(handle)
            ppath = ''
            for s in fname.split('/')[:-2]:
                ppath += s
            self.ppath = ppath
            self.pname = fname.split('/')[-2]
            self.selected_data = (self.datasets['name']==self.datasets['name'].iloc[0]) #select first element
            self.upd_tree()
            self.do_select()
            self.btns_enabled(True)
            
    def export(self):
        name = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\', 'Text files (*.txt)')[0]
        np.savetxt(name, self.info.export(), fmt="%s")
    
    def upd_tree(self):
        self.input_tree.clear()
        items = []
        for i in range(len(self.datasets)):
            key = self.datasets['name'].iloc[i]
            values = self.datasets['data_disp'].iloc[i]
            item = QTreeWidgetItem([key])
            for value in values:
                child = QTreeWidgetItem([value])
                child.setToolTip(0, value)
                item.addChild(child)
            item.setFlags(QtCore.Qt.ItemIsEditable |
                        QtCore.Qt.ItemIsEnabled |
                        QtCore.Qt.ItemIsSelectable)
            items.append(item)
            
        self.input_tree.insertTopLevelItems(0, items)
        
    def data_selection(self):
        getSelected = self.input_tree.selectedItems()
        if getSelected:
            baseNode = getSelected[0]
            getChildNode = baseNode.text(0)
            if getChildNode:
                if (getChildNode in self.datasets['name'].values):
                    select = getChildNode
                else:
                    name = baseNode.parent().text(0)
                    select = name
                    
                self.selected_data = (self.datasets['name']==select)
                self.do_select()
                
    def do_select(self, auto_select=False):
        if len(self.datasets)>0:
            if not np.any(self.selected_data):
                self.selected_data = (self.datasets['name']==self.datasets['name'].iloc[0]) #select first element
            select = self.datasets['data'][self.selected_data].values[0]
            for i, fname in enumerate(select):
                if i==0:
                    flag = self.model.load_data(fname)
                    if isinstance(flag, FileNotFoundError):
                        self.error_msg(f'File {fname} was not found!')
                        return
                else:
                    flag = self.model.add_data(fname)
                    if isinstance(flag, FileNotFoundError):
                        self.error_msg(f'File {fname} was not found!')
                        return
            if auto_select:
                self.auto_selection()
            else:
                self.u = np.array(self.datasets['u_saved'][self.selected_data].values[0])
            self.tafel()
        
    def auto_selection(self):
        u0 = self.model.xmin
        u3 = self.model.xmax
        u1 = u0+(u3-u0)/3
        u2 = u0+2*(u3-u0)/3
        self.u = np.array([u0, u1, u2, u3])
    
            
    def minus(self):
        getSelected = self.input_tree.selectedItems()
        if getSelected:
            auto_select = False
            baseNode = getSelected[0]
            getChildNode = baseNode.text(0)
            if getChildNode and (getChildNode in self.datasets['name'].values):
                ind = self.datasets.index[self.selected_data].tolist()[0]
                self.datasets = self.datasets.drop(ind)
            else:
                dataset = baseNode.parent().text(0)
                ind = (self.datasets['name']==dataset)
                i = np.where(np.array(self.datasets['data_disp'][ind].values[0])==getChildNode)[0][0]
                self.datasets['data_disp'][ind].values[0].pop(i)
                self.datasets['data'][ind].values[0].pop(i)
                
                if len(self.datasets['data_disp'][ind].values[0]) == 0:
                    ind = self.datasets.index[self.selected_data].tolist()[0]
                    self.datasets = self.datasets.drop(ind)
                    
                auto_select = True
                
            self.selected_data = (self.datasets['name']==self.datasets['name'].iloc[0]) #select first element
            self.upd_tree()
            self.do_select(auto_select)
        
        
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
        self.u = np.array(pos)
        self.datasets.loc[self.selected_data, 'u_saved'] = [self.u]
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