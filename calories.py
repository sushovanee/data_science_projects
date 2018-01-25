import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from pandas import DataFrame
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLars

data = read_csv('calories_consumed.csv')

# shape
print(data.describe())

# head
print(data.head(20))

# correlation
set_option('precision', 2)
print(data.corr(method='pearson'))

# histograms
data.hist()
pyplot.show()
pyplot.savefig('plots/cal_hist_plot.png')

# density
data.plot(kind='density', subplots=True, layout=(2,2), sharex=False, legend=False, fontsize=1)
pyplot.show()
pyplot.savefig('plots/cal_density_plot.png')

# box and whisker plots
data.plot(kind='box', sharex=False, sharey=False,
fontsize=8)
pyplot.show()
pyplot.savefig('plots/cal_box_plot.png')

# scatter plot matrix
scatter_matrix(data)
pyplot.show()
pyplot.savefig('plots/cal_scatter_plot.png')

# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = numpy.arange(0,14,1)
pyplot.show()
pyplot.savefig('plots/cal_corr_plot.png')

Y = data['Weight gained (grams)'].values
X = data['Calories Consumed'].values.reshape(-1,1)

lr = LinearRegression()
lr.fit(X, Y, sample_weight=None)

y_predict = lr.predict(X)

pyplot.scatter(Y, y_predict)
pyplot.show()
pyplot.savefig('plots/cal_predict_scatter_plot.png')

d = {'Calories Consumed':data['Calories Consumed'].values , 'Weight gained (grams)': Y}

# Spot-Check Algorithms
models = []
models.append(('LinearRegression', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Lasso', Lasso()))
models.append(('ElasticNet', ElasticNet()))
models.append(('LassoLars', LassoLars()))

# evaluate each model in turn
for name, model in models:
	model.fit(X, Y)
	y_predict = model.predict(X)
	d[name] = y_predict
	y_predict = []

df = DataFrame(data=d)
print(df.head())
df.to_csv('cal_output.csv', sep=',')