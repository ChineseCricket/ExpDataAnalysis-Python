# -*- coding: utf-8 -*-
"""
version 5.6-a 发行版

@author: 张靖毅，牛嘉豪

调用时请一并作如下调用：
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sympy import *
from scipy.stats import norm
from sklearn.metrics import r2_score
from scipy.stats import t
from scipy.optimize import curve_fit
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sympy import *
from scipy.stats import norm
from sklearn.metrics import r2_score
from scipy.stats import t
from scipy.optimize import curve_fit

class ExpDataAnalysis:
    '''
    功能：
        BasicAnalysis 基本统计量：
            计算因变量平均值、测量列的贝塞尔标准差、平均值的贝塞尔标准差、自由度；
        Plot 进行原始数据折线作图；
        CurvePlot 插值光滑曲线作图；
        LinearPlot 线性拟合作图并计算最小二乘参数的精度、残差列及相关系数；
        UncertaintyReport 完整的markdown格式不确定度报告。
    请按照如下示例传入：
    ExpData(因变量测量列，自变量测量列)
    自变量测量列应当是和因变量一一对应的n*n二维np矩阵
    因变量测量列应当是在各因变量行向量组成的二维np矩阵，矩阵缺失值处请用nan占位，确保每行等长
    '''
    def __init__(self, L, A = 0):
        if type(A) != int:
            self.A = A.astype(np.float64)
        self.L = L.astype(np.float64)

    def BasicAnalysis(self):
        '''
        分析全部传入的因变量数据的均值、测量列的贝塞尔标准差和平均值的贝塞尔标准差及自由度
        '''
        Average = np.empty((0,1))
        for l in self.L:
            l = np.array(pd.DataFrame(l).dropna()).flatten()
            Average = np.append(Average, [[np.average(l)]])
        BslStd = np.empty((0,1))
        for l in self.L:
            l = np.array(pd.DataFrame(l).dropna()).flatten()
            BslStd = np.append(BslStd, [[np.sqrt(sum([x**2 for x in
                                              (l-np.average(l))])
                                         /(l.shape[0]-1))]])
        AvgBslStd = BslStd/np.sqrt(self.L.shape[1])
        DegreeofFreedom = np.empty((0,1))
        for l in self.L:
            l = np.array(pd.DataFrame(l).dropna())
            DegreeofFreedom = np.append(DegreeofFreedom, [[l.shape[0] - 1]])
        print('各平均值为:', Average,
              '各测量列的标准差为:', BslStd,
              '各平均值的标准差为:', AvgBslStd,
              '各测量列的自由度为:',DegreeofFreedom,
              sep='\n')
        return Average, BslStd, AvgBslStd, DegreeofFreedom

    def Plot(self, labels = [0], xlabel = '', ylabel = '',xlim=0, ylim=0, xscale='linear', yscale='linear',title = '', figsize = (6,4), FileAdress = '', scatter=True):
        '''
        直接作图。
        labels:list，存放对应各测量列的标签；
        xlabel:图片的x坐标轴标签；
        ylabel:图片的y坐标轴标签；
        xscale:图片的x轴形式，如线性linear或对数log；
        yscale:图片的y轴形式，同上；
        title:图片上的标题；
        figsize:画布大小；
        FileAdress:图片保存地址；
        scatter:是否绘制数据点，默认绘制。
        返回：画布，子图
        '''
        if pd.isnull(xlabel):
            xlabel = ''
        if pd.isnull(ylabel):
            ylabel = ''
        if pd.isnull(title):
            title = ''
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(1,1, figsize = figsize)
        i = 0
        if all(labels):
            for l in self.L:
                if scatter:
                    ax.plot(self.A[i], l, marker = '+', label=labels[i])
                else:
                    ax.plot(self.A[i], l, label=labels[i])
                i+=1
            ax.legend(frameon=False)
        else:
            for l in self.L:
                if scatter:
                    ax.plot(self.A[i], l, marker = '+', label=labels[i])
                else:
                    ax.plot(self.A[i], l, label=labels[i])
        ax.grid()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        fig.suptitle(title)
        if FileAdress != '':
            fig.savefig(FileAdress)
        return fig, ax

    def LinearfitPlot(self, xname = 'x', yname = 'y', xlabel = '', ylabel = '', title = '', xscale='linear', yscale='linear', figsize = (6, 4), ci = 99.74, FileAdress = '',pattern=1,Nofigmode=False):
        '''
        返回：画布，子图，线性最小二乘参数，线性最小二乘参数误差，残差列。默认置信概率99.74%。
        '''
        '''
        if pd.isnull(xlabel):
            xlabel = ''
        if pd.isnull(ylabel):
            ylabel = ''
        if pd.isnull(title):
            print('pass')
        if pd.isnull(xname):
            xname = 'x'
        if pd.isnull(yname):
            yname = 'y'
        if pd.isnull(ci):
            ci = 99.74
        '''
        color=['b','g','r','c','y']
        A = self.A
        L = self.L
        n=len(L)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        '''
        最小二乘估计量的精度估计函数
        '''
        def LinearVarience(A, L):
            '''
            等精度测量列的线性最小二乘估计值和测量值的精度计算函数
            示例：LinearVarience(自变量测量列，因变量测量列）
            返回：[斜率精度,截距精度]，残差列
            '''
            line_A = np.polyfit(A, L, 1)
            v = [(L[i]-(line_A[0]*A[i] + line_A[1]))**2 for i in range(L.shape[0])]
            sigma = sum(v)/(L.shape[0]-2)
            A = np.vstack([A, np.ones(A.shape)]).T
            return np.sqrt(np.diag(sigma * np.linalg.inv(np.matmul(A.T, A)))), np.array(v)

        if pattern==1:
            Line=np.array([0,0])
            D=np.array([0,0])
            R2=np.array([])
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            for j in range(n):
                L[j] = L[j].astype(np.float64)
                line = np.polyfit(A[0], L[j], 1)
                r2 = r2_score(L[j], line[0]*A[0]+line[1])
                d, v = LinearVarience(A[0], L[j])
                R2=np.append(R2,r2)
                D=np.vstack([D,d])
                Line=np.vstack([Line,line])
                a = round(norm.ppf(ci / 100 + (1 - ci / 100) / 2), 2)
                if Nofigmode: 
                    continue
                else:
                    sns.regplot(x=A[0], y=L[j], ax = ax, line_kws = {'label':'%s = (%f$\pm$%f)%s + (%f$\pm$%f)\n$R^2=%f$'
                                %(yname[j], line[0], a*d[0], xname[j], line[1], a*d[1], r2)}, ci = ci, marker='+')
                    ax.scatter(A[0], L[j], marker = '+')
            if Nofigmode: 
                pass
            else: 
                ax.grid()
                ax.legend(frameon = False)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                fig.suptitle(title)

        else:
            ar=[i for i in range(n)]
            fig, axs = plt.subplots(n, 1, figsize=figsize)
            pairs=zip(ar,axs)
            Line=np.array([0,0])
            D=np.array([0,0])
            R2=np.array([])
            i = 0
            for (j,ax) in pairs:
                L[j] = L[j].astype(np.float64)
                line = np.polyfit(A[0], L[j], 1)
                r2 = r2_score(L[j], line[0]*A[0]+line[1])
                d, v = LinearVarience(A[0], L[j])
                R2=np.append(R2,r2)
                D=np.vstack([D,d])
                Line=np.vstack([Line,line])
                a = round(norm.ppf(ci / 100 + (1 - ci / 100) / 2), 2)
                sns.regplot(x=A[j], y=L[j], ax = ax, line_kws = {'label':'%s = (%f$\pm$%f)%s + (%f$\pm$%f)\n$R^2=%f$'
                            %(yname[i], line[0], a*d[0], xname[i], line[1], a*d[1], r2)}, ci = ci, marker='+',color=color[i])
                ax.scatter(A[0], L[j], marker = '^',c=color[i])
                ax.grid(c=color[i],linestyle="-.")
                ax.legend(frameon = False)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                fig.suptitle(title)
                i+=1
        plt.show()

        if FileAdress != '':
            fig.savefig(FileAdress)
        if Nofigmode:
            return Line[1:], D[1:], v, R2
        else:
            return fig, ax, Line[1:], D[1:], v, R2

    def CurvefitPlot(self, func, init, method = 'trf', xname = 'x', yname = 'y', xlabel = '', ylabel = '', labels=[''], xscale='linear', yscale='linear', title = '', figsize = (6,4), ci = 95, FileAdress = ''):
        '''
        单变量非线性函数拟合，并给出指定置信概率下的拟合参数精度估计值。
        func：欲拟合之函数，其参数的第一项为自变量，后各项为欲拟合之参数；
        init：欲拟合之参数的迭代初值。
        默认置信概率为ci=95（%）。
        返回：画布，子图，拟合参数，拟合参数精度估计。
        '''
        L=self.L
        A=self.A
        if pd.isnull(xlabel):
            xlabel = ''
        if pd.isnull(ylabel):
            ylabel = ''
        if pd.isnull(title):
            title = ''
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(1,1, figsize = figsize)
        para=np.zeros([L.shape[0],func.__code__.co_argcount-1]).astype(float)
        para_std=np.zeros([L.shape[0],func.__code__.co_argcount-1]).astype(float)
        No=0
        a = round(norm.ppf(ci / 100 + (1 - ci / 100) / 2), 2)
        for i,l in enumerate(L):
            para[No], para_cov = curve_fit(func, A[i], l, init, method=method)
            para_std[No] = a*np.sqrt(para_cov.diagonal())
            ax.scatter(A[i],l,marker='+',c='r',label='实验数据')
            if xscale=='linear':
                X=np.linspace(A[i][0],A[i][-1],num=10000)
            if xscale=='log':
                X=np.exp(np.linspace(np.log(A[i][0]),np.log(A[[i]-1]),num=10000))
            ax.plot(X,func(X,*para[No]),label='拟合曲线')
            No=No+1
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.legend(frameon=False)
        fig.suptitle(title)
        ax.grid()
        if FileAdress != '':
            fig.savefig(FileAdress)
        return fig, ax, para, para_std

    def UnequalAccuracyMeasurement(exData, data=[]):
        '''
        data与exData都是np.array形式
        data测量列平均值汇总
        '''
        if (len(data) != 0):
            mean_data = []
            std_data=[]
            for item in data:
                n=len(item)
                mean_data.append(np.array(item).mean())
                std_data.append(np.array(item).std()*(n/(n-1))**0.5)
            mean_set = np.array(mean_data + list(exData[0]))
            std_set = np.array(std_data + list(exData[1]))
        else:
            mean_set = np.array(list(exData[0]))
            std_set = np.array(list(exData[1]))
        # 权重设置
        weight = 1 / std_set ** 2
        v_i = mean_set - mean_set.mean()
        m = len(data) + len(exData[0])
        numerator = (v_i ** 2 * weight).sum()
        denominator = weight.sum()
        sigma_st = (numerator / denominator) ** 0.5
        print("测量列的均值为：", mean_set.mean(), "测量列的标准差为：", sigma_st)
        return mean_set.mean(), sigma_st

    def UncertaintyReport(self, exData = pd.DataFrame([]), func = 0, ci = 99.73, reslab = 'X',resunit = '',precision=[2,2,2,2],standard=False,Unicode=False):
        '''
        间接测量量的不确定度报告生成程序。
        可选传入参数及其默认值：
        exData:额外传入的参数数据，同一数据的平均值、标准差和自由度应分别在同一列的第一行、第二行和第三行，默认为空DataFrame；
        func:间接测量量函数名，应当为一个外部定义的函数，将所有的参数打包传入，示例如下：
                def func(L):
                    m,s = L
                    return m/s
            注意应当将确保函数中定义参数的顺序和给入本类内部的数据的行顺序一致，并将需要额外给入的数据放在最后面。
            默认为传入的单变量自身；
        ci:置信概率，默认为99.73%；
        reslab:间接测量量的变量名，默认为'X'；
        resunit:间接测量量的单位，默认为空；
        precision:一维list，格式为[间接测量量小数保留位数，间接测量量的展伸不确定度结果小数保留位数，间接测量量的展伸不确定度报告小数保留位数，合成标准不确定度小数保留位数]，默认[2,2,2,2]；
        standard:非0的间接测量量的预期真值，给入后可以附加计算实验结果和预期值的相对误差，默认为False，即不计算相对误差；
        Unicode:此项传入True时为Unicode格式输出模式，默认为False，即默认用Latex格式输出。
        依次返回：间接测量量值，包含因子，合成标准不确定度，合成自由度。
        '''
        '''
        if exData.empty:
            exData = pd.DataFrame([])
        if pd.isnull(ci):
            ci = 99.73
        if pd.isnull(reslab):
            reslab = 'X'
        if pd.isnull(resunit):
            resunit = ''
        '''
        Average, BslStd, AvgBslStd, DFA = self.BasicAnalysis()
        #对单变量函数进行单独分析
        dfres=DFA[0]
        print(len(self.L))
        if (len(self.L)==1 and exData.shape[1]==0):
            if func != 0:
                para = func.__code__.co_varnames
                x=Symbol(para[1])
                value=func(x).evalf(subs={x:Average[0]})
                #定义传递系数
                av=diff(func(x)).evalf(subs={x:Average[0]})
                print("传递系数为：",av)
                k = t.ppf(ci / 100 + (1 - ci / 100) / 2, float(dfres))
                ures=av*AvgBslStd[0]
                print('包含因子k =', k)
            else:
                value=Average[0]
                ures=AvgBslStd[0]
                k = t.ppf(ci / 100 + (1 - ci / 100) / 2, float(dfres))
                print('包含因子k =', k)
            if Unicode:
                if standard:
                    error=((standard-value)/standard)*100
                    print('不确定度报告：\n%s=%s_aver \\pm U_%s \\approx (' % (reslab, reslab, reslab),round(value, precision[0]), '\\pm ', round(k * ures, precision[1]),')%s\n\delta_{%s}=\\frac{%s-%s_T}{%s_T}=%.2f%%\n'%(resunit,reslab,reslab,reslab,reslab,error),'\n其中展伸不确定度U_%s =k\\times u_%s \\approx ' % (reslab, reslab), round(k * ures, precision[2]),"%s" % resunit, "是由合成标准不确定度u_%s\\approx " % reslab, round(ures, precision[3]),'%s' % resunit, '和k\\approx %.2f确定的，k是依据置信概率P=%.2f%%和自由度\\nu \\approx %.2f查t分布表得到的。'% (k, ci, dfres),'其相对误差为%.2f%%'%error)
                else:
                    print('不确定度报告：\n%s=%s_aver \\pm U_%s \\approx (' % (reslab, reslab, reslab),round(value, precision[0]), '\\pm ', round(k * ures, precision[1]),')%s' % (resunit),'\n其中展伸不确定度U_%s =k\\times u_%s \\approx ' % (reslab, reslab), round(k * ures, precision[2]),"%s" % resunit, "是由合成标准不确定度u_%s\\approx " % reslab, round(ures, precision[3]),'%s' % resunit, '和k\\approx %.2f确定的，k是依据置信概率P=%.2f%%和自由度\\nu \\approx %.2f查t分布表得到的。'% (k, ci, dfres))
            else:
                if standard:
                    error=((value-standard)/standard)*100
                    print('不确定度报告：\n\\begin{align}\n%s&=\\bar{%s}\\pm U_{%s}\\approx(' % (reslab, reslab, reslab),round(value, precision[0]), '\\pm ', round(k * ures, precision[1]), ')%s\\\\\n\delta_{%s}&=\\frac{%s-%s_T}{%s_T}=%.2f\\%%\n'%(resunit,reslab,reslab,reslab,reslab,error),'\\end{align}''\n其中展伸不确定度$U_{%s}={k}\\times{u_{%s}}\\approx' % (reslab, reslab), round(k * ures, precision[2]),"%s$" % resunit, "是由合成标准不确定度$u_{%s}\\approx" % reslab, round(ures, precision[3]),'%s$' % resunit, '和$k\\approx%.2f$确定的，k是依据置信概率$P=%.2f\\%%$和自由度$\\nu\\approx%.2f$查t分布表得到的，'% (k, ci, dfres),'其相对误差为%.2f\\%%。'%error)
                else:
                    print('不确定度报告：\n\\begin{equation}\n%s=\\bar{%s}\\pm U_{%s}\\approx(' % (reslab, reslab, reslab),
                        round(value, precision[0]), '\\pm ', round(k * ures, precision[1]), ')%s\n\\end{equation}' % (resunit),
                        '\n其中展伸不确定度$U_{%s}={k}\\times{u_{%s}}\\approx' % (reslab, reslab), round(k * ures, precision[2]),
                        "%s$" % resunit, "是由合成标准不确定度$u_{%s}\\approx" % reslab, round(ures, precision[3]),
                        '%s$' % resunit, '和$k\\approx%.2f$确定的，k是依据置信概率$P=%.2f\\%%$和自由度$\\nu\\approx%.2f$查t分布表得到的。'
                        % (k, ci, dfres))


        elif (len(self.L)==0 and exData.shape[1]==1):
            dfres=exData[2][0]
            print('平均值为:', exData[0][0],
                  '平均值的标准差为:', exData[1][0],
                  '测量列的自由度为:', exData[2][0])
            if func != 0:
                para = func.__code__.co_varnames
                x=Symbol(para[1])
                value=func(x).evalf(subs={x:exData[0][0]})
                #定义传递系数
                av=diff(func(x)).evalf(subs={x:exData[0][0]})
                print("传递系数为：",av)
                k = t.ppf(ci / 100 + (1 - ci / 100) / 2, float(exData[2][0]))
                ures=av*exData[1][0]
                print('包含因子k =', k)
            else:
                value=exData[0][0]
                ures=exData[1][0]
                k = t.ppf(ci / 100 + (1 - ci / 100) / 2, float(exData[2][0]))
                print('包含因子k =', k)
            if Unicode:
                if standard:
                    error=((standard-value)/standard)*100
                    print('不确定度报告：\n%s=%s_aver \\pm U_%s \\approx (' % (reslab, reslab, reslab),round(value, precision[0]), '\\pm ', round(k * ures, precision[1]),')%s\n\delta_{%s}=\\frac{%s-%s_T}{%s_T}=%.2f%%\n'%(resunit,reslab,reslab,reslab,reslab,error),'\n其中展伸不确定度U_%s =k\\times u_%s \\approx ' % (reslab, reslab), round(k * ures, precision[2]),"%s" % resunit, "是由合成标准不确定度u_%s\\approx " % reslab, round(ures, precision[3]),'%s' % resunit, '和k\\approx %.2f确定的，k是依据置信概率P=%.2f%%和自由度\\nu \\approx %.2f查t分布表得到的。'% (k, ci, dfres),'其相对误差为%.2f%%'%error)
                else:
                    print('不确定度报告：\n%s=%s_aver \\pm U_%s \\approx (' % (reslab, reslab, reslab),round(value, precision[0]), '\\pm ', round(k * ures, precision[1]),')%s' % (resunit),'\n其中展伸不确定度U_%s =k\\times u_%s \\approx ' % (reslab, reslab), round(k * ures, precision[2]),"%s" % resunit, "是由合成标准不确定度u_%s\\approx " % reslab, round(ures, precision[3]),'%s' % resunit, '和k\\approx %.2f确定的，k是依据置信概率P=%.2f%%和自由度\\nu \\approx %.2f查t分布表得到的。'% (k, ci, dfres))
            else:
                if standard:
                    error=((value-standard)/standard)*100
                    print('不确定度报告：\n\\begin{align}\n%s&=\\bar{%s}\\pm U_{%s}\\approx(' % (reslab, reslab, reslab),round(value, precision[0]), '\\pm ', round(k * ures, precision[1]), ')%s\\\\\n\delta_{%s}&=\\frac{%s-%s_T}{%s_T}=%.2f\\%%\n'%(resunit,reslab,reslab,reslab,reslab,error),'\\end{align}''\n其中展伸不确定度$U_{%s}={k}\\times{u_{%s}}\\approx' % (reslab, reslab), round(k * ures, precision[2]),"%s$" % resunit, "是由合成标准不确定度$u_{%s}\\approx" % reslab, round(ures, precision[3]),'%s$' % resunit, '和$k\\approx%.2f$确定的，k是依据置信概率$P=%.2f\\%%$和自由度$\\nu\\approx%.2f$查t分布表得到的，'% (k, ci, dfres),'其相对误差为%.2f\\%%。'%error)
                else:
                    print('不确定度报告：\n\\begin{equation}\n%s=\\bar{%s}\\pm U_{%s}\\approx(' % (reslab, reslab, reslab),
                        round(value, precision[0]), '\\pm ', round(k * ures, precision[1]), ')%s\n\\end{equation}' % (resunit),
                        '\n其中展伸不确定度$U_{%s}={k}\\times{u_{%s}}\\approx' % (reslab, reslab), round(k * ures, precision[2]),
                        "%s$" % resunit, "是由合成标准不确定度$u_{%s}\\approx" % reslab, round(ures, precision[3]),
                        '%s$' % resunit, '和$k\\approx%.2f$确定的，k是依据置信概率$P=%.2f\\%%$和自由度$\\nu\\approx%.2f$查t分布表得到的。'
                        % (k, ci, dfres))
        #多变量函数分析
        else:
            if exData.shape[0] != 0:
                Average = np.append(Average, exData[0])
                AvgBslStd = np.append(AvgBslStd, exData[1])
                DFA = np.append(DFA, exData[2])
                Average = np.array(pd.DataFrame(Average).dropna().T.iloc[0])
                AvgBslStd = np.array(pd.DataFrame(AvgBslStd).dropna().T.iloc[0])
                DFA = np.array([x for x in DFA if x != -1])
            if func != 0:
                value = func(Average)
                print('间接测量量：',value)
                para = func.__code__.co_varnames
                X = symbols(','.join(para[1:]))
                if type(X) == Symbol:
                    X = [X]
                print('所有参数为:',','.join(para[1:]))
                sigma = dict(zip(X,AvgBslStd))
                print(sigma)
                values = dict(zip(X,Average))
                dfas = dict(zip(X,DFA))
                ures = 0
                print('传递系数和各项标准不确定度：')
                i = 1
                for x in X:
                    a = diff(func(X),x)
                    av = diff(func(X),x).subs(values)
                    print('a{}'.format(x),'=',a,'=',av)
                    uv = av*sigma[x]
                    print('u{}'.format(i),'=a{}*u{}'.format(x,x),'=',uv)
                    ures += uv**2
                    i += 1
                i = 1
            else:
                raise ValueError('传入了太多变量！')
            ures=sqrt(ures)
            print('合成标准不确定度u =',ures)
            print('各自由度分量:')
            if DFA.shape[0] == Average.shape[0]:
                if DFA.shape[0] > 1:
                    dfres = 0
                    for x in X:
                        print('ν{}'.format(x),'=',dfas[x])
                        a = diff(func(X),x)
                        av = diff(func(X),x).subs(values)
                        uv = av*sigma[x]
                        dfres += (uv**4)/dfas[x]
                if DFA.shape[0] == 1:
                    print('ν','=',DFA[0])
                    dfres = ures**4/DFA[0]
            else:
                raise ValueError('测量列自由度数量不匹配！')
            dfres = (ures**4)/dfres
            print('合成自由度ν =', dfres)
            k = t.ppf(ci/100 + (1 - ci/100) / 2, float(dfres))
            print('包含因子k =', k)
            if Unicode:
                if standard:
                    error=((standard-value)/standard)*100
                    print('不确定度报告：\n%s=%s_aver \\pm U_%s \\approx (' % (reslab, reslab, reslab),round(value, precision[0]), '\\pm ', round(k * ures, precision[1]),')%s\n\delta_{%s}=\\frac{%s-%s_T}{%s_T}=%.2f%%\n'%(resunit,reslab,reslab,reslab,reslab,error),'\n其中展伸不确定度U_%s =k\\times u_%s \\approx ' % (reslab, reslab), round(k * ures, precision[2]),"%s" % resunit, "是由合成标准不确定度u_%s\\approx " % reslab, round(ures, precision[3]),'%s' % resunit, '和k\\approx %.2f确定的，k是依据置信概率P=%.2f%%和自由度\\nu \\approx %.2f查t分布表得到的。'% (k, ci, dfres),'其相对误差为%.2f%%'%error)
                else:
                    print('不确定度报告：\n%s=%s_aver \\pm U_%s \\approx (' % (reslab, reslab, reslab),round(value, precision[0]), '\\pm ', round(k * ures, precision[1]),')%s' % (resunit),'\n其中展伸不确定度U_%s =k\\times u_%s \\approx ' % (reslab, reslab), round(k * ures, precision[2]),"%s" % resunit, "是由合成标准不确定度u_%s\\approx " % reslab, round(ures, precision[3]),'%s' % resunit, '和k\\approx %.2f确定的，k是依据置信概率P=%.2f%%和自由度\\nu \\approx %.2f查t分布表得到的。'% (k, ci, dfres))
            else:
                if standard:
                    error=((value-standard)/standard)*100
                    print('不确定度报告：\n\\begin{align}\n%s&=\\bar{%s}\\pm U_{%s}\\approx(' % (reslab, reslab, reslab),round(value, precision[0]), '\\pm ', round(k * ures, precision[1]), ')%s\\\\\n\delta_{%s}&=\\frac{%s-%s_T}{%s_T}=%.2f\\%%\n'%(reslab,reslab,reslab,reslab,reslab,error),'\\end{align}''\n其中展伸不确定度$U_{%s}={k}\\times{u_{%s}}\\approx' % (reslab, reslab), round(k * ures, precision[2]),"%s$" % resunit, "是由合成标准不确定度$u_{%s}\\approx" % reslab, round(ures, precision[3]),'%s$' % resunit, '和$k\\approx%.2f$确定的，k是依据置信概率$P=%.2f\\%%$和自由度$\\nu\\approx%.2f$查t分布表得到的，'% (k, ci, dfres),'其相对误差为%.2f\\%%。'%error)
                else:
                    print('不确定度报告：\n\\begin{equation}\n%s=\\bar{%s}\\pm U_{%s}\\approx(' % (reslab, reslab, reslab),
                        round(value, precision[0]), '\\pm ', round(k * ures, precision[1]), ')%s\n\\end{equation}' % (resunit),
                        '\n其中展伸不确定度$U_{%s}={k}\\times{u_{%s}}\\approx' % (reslab, reslab), round(k * ures, precision[2]),
                        "%s$" % resunit, "是由合成标准不确定度$u_{%s}\\approx" % reslab, round(ures, precision[3]),
                        '%s$' % resunit, '和$k\\approx%.2f$确定的，k是依据置信概率$P=%.2f\\%%$和自由度$\\nu\\approx%.2f$查t分布表得到的。'
                        % (k, ci, dfres))


        return value,k,ures,dfres
