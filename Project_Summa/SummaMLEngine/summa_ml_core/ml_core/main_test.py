from .feature_controller import *
from scipy import stats


import numpy as np
import pandas as pd



def feature_selection_test():
	df = pd.read_csv("./raw_data/brain_size.csv", sep=";", na_values=".")

	#print(df.info())
	df.dropna(inplace=True)

	fc = feature_controller(df, df['Height'])
	#resultsets = fc.feature_selection_statistics(df.loc[:,'FSIQ':], df['Height'])
	fc.feature_selection_statistics()

def feature_covariance_matrix_plot_test():
	dh1 = data_holder()
	vt = visualization_toolbox(dh1, 'brain_size.csv')

	vt.feature_covariance_matrix_plot()


def feature_dim_reduction_plot_test():
	dh1 = data_holder()
	vt = visualization_toolbox(dh1, 'brain_size.csv')
	digits = load_digits(n_class=6)
	X = digits.data
	y = digits.target
	vt.feature_dim_reduction_plot(X, y, (8,8))


def main():
	#feature_selection_test()
	#feature_dim_reduction_test()
	feature_covariance_matrix_plot_test()	
	


if __name__ == '__main__':
	main()
