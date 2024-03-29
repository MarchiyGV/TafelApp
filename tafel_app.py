import numpy as np
from sklearn.linear_model import LinearRegression
import os


class Model:
    k = 8.617333262e-5
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.E_corr = 0
        self.i_corr = 0
        self.alpha = 0
        self.b1 = 0
        self.b2 = 0
        alpha = 0.8
        ocv = 0
        self.x = np.linspace(-0.2, 0.3, num=100).reshape((-1, 1))
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.dx = self.xmax-self.xmin
        self.y = self.j(alpha, 300, self.x-ocv)
        self.ylog = np.log(np.abs(self.y))
        
    def _read(self, fname):
        try:
            s = ''
            metadata2w = '\n'
            metadata = ''
            with open(fname) as f:
                for line in f:
                    if '#' in line:
                        metadata += line.replace('#', '')
                        metadata2w += line
                    else:
                        s += line.replace(",", ".") 
        except FileNotFoundError:
            self.parent.error_msg(f'File "{fname}" was not found!')
            return 
        return s, metadata, metadata2w
    
    def read_data(self, fname, flag):
        temp = fname.split('.')
        temp = temp[-2].split('/')
        new_fname = (str(temp[-1])+'.tdata')
        new_fpath = self.parent.ppath + '/' + new_fname
        if flag:
            s, metadata, metadata2w = self._read(fname)
            f = open(new_fpath, 'w+')
            f.write(s)
            f.close()
            f = open(new_fpath, 'a')
            f.write(metadata2w)
            f.close()
            data = np.loadtxt(new_fpath, skiprows=1, delimiter='\t')
            return data, new_fname, metadata
        else:
            s, metadata, metadata2w = self._read(new_fpath)
            data = np.loadtxt(new_fpath, skiprows=1, delimiter='\t')
            return data, fname, metadata
    
    def load_data(self, fname):
        flag = True 
        if fname.split('.')[-1] == 'tdata':
            flag = False
        data, new_fname, metadata = self.read_data(fname, flag)
            
        self.x = np.transpose(data[:, 1]).reshape((-1, 1))
        ind = np.argsort(self.x, axis=0)
        self.x = self.x[ind][:,:,0]
        self.y = np.transpose(data[:, 2]).reshape((-1, 1))
        self.y = self.y[ind][:,:,0]
        mask = (self.y==0)
        self.y[mask] = np.mean(self.y)*1e-16
        self.ylog = np.log10(np.abs(self.y))
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.dx = self.xmax-self.xmin
        return new_fname, metadata
        
    def add_data(self, fname):
        flag = True 
        if fname.split('.')[-1] == 'tdata':
            flag = False
        data, new_fname, metadata = self.read_data(fname, flag)
        
        x_new = np.transpose(data[:, 1]).reshape((-1, 1))
        self.x = np.concatenate((self.x, x_new))
        ind = np.argsort(self.x, axis=0)
        self.x = self.x[ind][:,:,0]
        y_new = np.transpose(data[:, 2]).reshape((-1, 1))
        mask = (y_new==0)
        y_new[mask] = np.mean(y_new)*1e-16
        self.y = np.concatenate((self.y, y_new))
        self.y = self.y[ind][:,:,0]
        self.ylog = np.log10(np.abs(self.y))
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.dx = self.xmax-self.xmin
        return new_fname, metadata
        
    def j(self, alpha, T, eta):
        f = 1/(Model.k*T)
        c = 0.05
        j = np.exp((1-alpha)*f*eta) - np.exp(-alpha*f*eta)
        eta_c = np.log(1-j*c)/f
        eta = eta + eta_c
        j = np.exp((1-alpha)*f*eta) - np.exp(-alpha*f*eta)
        return j

    
    def fit(self, i):
        i0, i1 = i
        model = LinearRegression()
        ind = slice(i0,i1)
        model.fit(self.x[ind], self.ylog[ind])
        r_sq = model.score(self.x[ind], self.ylog[ind])
        return r_sq
    
    def linreg_plus(self, u0, u1, du):
        x = self.x
        ylog = self.ylog
        dx = x.max()-x.min()
        di = round(du/dx*len(x))
        _i0 = max(round((u0-x.min())/dx*len(x)), 0)
        _i1 = min(round((u1-x.min())/dx*len(x)), len(x)-di)
        i0s = np.arange(_i0, _i1)
        arg_i1s = []
        r_sq_1 = []
        for i0 in i0s:
            i1s = np.arange(i0+di, len(x))
            r_sq_2 = []
            arg_2 = []
            for i1 in i1s:
                r_sq = self.fit([i0, i1])
                r_sq_2.append(r_sq)
                arg_2.append(i1)
            arg_i1s.append(arg_2[np.argmax(r_sq_2)])
            r_sq_1.append(np.max(r_sq_2))
            
        i0 = i0s[np.argmax(r_sq_1)]
        i1 = arg_i1s[np.argmax(r_sq_1)]
        
        model1 = LinearRegression()
        ind = slice(i0,i1)
        model1.fit(x[ind], ylog[ind])
        r_sq = model1.score(x[ind], ylog[ind])
        y_pred = model1.predict(x)
        a1 = model1.coef_[0][0]
        b1 = model1.intercept_[0]
        return a1, b1

    def tafel_plus(self, u0, u1, u2, u3, du1, du2):
        a1, b1 = self.linreg_plus(u0, u1, du1)
        a2, b2 = self.linreg_plus(u2, u3, du2)
    
        E_corr = (b2-b1)/(a1-a2)
        i_corr = a1*E_corr+b1
        alpha = -a1/(a2-a1)
        self.E_corr = E_corr
        self.i_corr = i_corr
        self.alpha = alpha
        self.a1 = a1
        self.a2 = a2
        return E_corr, i_corr, alpha, a1, b1, a2, b2
        
    def linreg(self, i0, i1):
        x = self.x
        ylog = self.ylog
        
        model = LinearRegression()
        ind = slice(i0,i1)
        model.fit(x[ind], ylog[ind])
        a = model.coef_[0][0]
        b = model.intercept_[0]
        return a, b
        
    def tafel(self, i0, i1, i2, i3):
        a1, b1 = self.linreg(i0, i1)
        a2, b2 = self.linreg(i2, i3)
    
        E_corr = (b2-b1)/(a1-a2)
        i_corr = np.power(10, a1*E_corr+b1)
        alpha = -a1/(a2-a1)
        self.E_corr = E_corr
        self.i_corr = i_corr
        self.alpha = alpha
        self.a1 = a1
        self.a2 = a2
        return E_corr, i_corr, alpha, a1, b1, a2, b2