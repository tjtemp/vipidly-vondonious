import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob


class data_holder:
    """
        dataset : indicates individual datafile
        datatable : 2 dimensional table with indexes and columns,
            columns contains categorical and continuous variables as features
            indexs contains number of samples

        self.datatable_names : only contains after read_data()
                basename of the table. left out './raw_data/'
    """

    def __init__(self, *args):
        self.datatable_names = list(args)
        self.dataset_num = len(self.datatable_names)
        self.datatables = []
        self.native_flag = False # on interactive mode

        print('='*50)
        print('data_holder is created..')
        print()
        for _ in args:
            self.dataset_num += 1
        print("number of data sets : ", self.dataset_num)
        print("datatable_names : ", self.datatable_names)


    def inspect_data_integrity(self, table, table_name):
        print('Table contains NaN value : ',np.any(table.isnull()),
            )


    def read_data(self, filepath=None):
        """
        read data
        currently implemented only for csv file
        in ./raw_data directory
        """

        # if user didn't specify a dataset folder
        # select all dataset in ./raw_data
        if self.dataset_num == 0:
            # csv data input
            # caution with django framework : "." directory is in manage.py file
            path = os.path.abspath(".")

            if filepath == None:
                path += "/SummaMLEngine/summa_ml_core/ml_core"
                path += "/raw_data/"
                datatable_names = glob.glob(path + "*.csv")

                for table_name in datatable_names:
                    # table name registration
                    table_name = os.path.basename(table_name)
                    self.datatable_names.append(table_name)

                    # table data registration
                    df = pd.read_csv("./SummaMLEngine/summa_ml_core/ml_core/raw_data/" + table_name, sep=',',
                                     na_values=';')
                    self.datatables.append(df)

                # json data input
                # excluding trailing data \n for nested json structure
                datatable_names = glob.glob(path + "*.json")

                for table_name in datatable_names:
                    # table name registration
                    table_name = os.path.basename(table_name)
                    self.datatable_names.append(table_name)

                    # table data registration
                    with open("./SummaMLEngine/summa_ml_core/ml_core/raw_data/" + table_name, 'rb') as f:
                        data = f.readlines()
                    data = map(lambda x: x.decode("utf-8").strip(), data)
                    data_str = "[" + ",".join(data) + "]"
                    df_json = pd.read_json(data_str)
                    self.datatables.append(df_json)

            else:
                datatable_names = glob.glob(filepath)


        else:
            print("user specified folder/file")

        print(self.datatable_names)


    def col_exclude(self, table, ex_col_names):
        return table


    def trim_data(self, datasets):
        """
        Trim datasets. NaN
        """
        return 1


    def categorize_data(self, table_name):
        """
        differ data with categorical and continuous data columns

        input : (self, table_name):
        return : categorical_data_columns, continuous_data_columns
        """
        #TODO: currently only works for csv file
        categorical_data_columns=[]
        continuous_data_columns=[]
        table_index = self.datatable_names.index(str(table_name))
        colnames = self.datatables[table_index].columns

        for colname in colnames:
            if 'object' == self.datatables[table_index][colname].dtype:
                categorical_data_columns.append(colname)
            else:
                continuous_data_columns.append(colname)

        return categorical_data_columns, continuous_data_columns


    def print_data_table_info(self, table_name):
        print("print_data_table_info call")
        cr = ''
        delim = '<br/>'
        print(self.datatable_names)
        table_index = self.datatable_names.index(str(table_name))
        table = self.datatables[table_index]
        if self.native_flag == True:
            print(table_index, table_name)
            print('='*50)
            print("table name : ", str(table_name))
            print("table shape : ", table.shape)
            print("table info : "), table.info()
            print("table columns : \n", table.columns)
            print("table index : \n", table.index)
            print("table head : \n", table.head())
        else:
            cr += '='*50 + delim
            cr += "table name : {}".format(str(table_name)) + delim
            cr += "table info : {}".format(table.info()) + delim
            cr += "table columns : \n{}".format(table.columns) + delim
            cr += "table index : \n{}".format(table.index) + delim
            cr += "table head : \n{}".format(table.head()) + delim

        if table.shape[1] <= 30:
            if self.native_flag == True:
                print('='*50)
                print("column-wise info .. ")
            else:
                cr += '='*50 + delim
                cr += "column-wise info .. "  + delim


            categorical_data_columns, continuous_data_columns = self.categorize_data(table_name)
            print("categorical features : ", categorical_data_columns)

            for cat_col in categorical_data_columns:
                uniq = np.unique(table[cat_col])
                if self.native_flag == True:
                    print('-'*50)
                    print('# col {}\t unique_num {}\t unique_val {}\t'.format(
                            cat_col, len(uniq), uniq))
                else:
                    cr += '-'*50 + delim
                    cr += '# col {}\t unique_num {}\t unique_val {}\t'.format(
                            cat_col, len(uniq), uniq) + delim
            if self.native_flag == True:
                print("continuous features : ", continuous_data_columns)
                print(table.describe())
            else:
                cr += "continuous features : ".format(continuous_data_columns) + delim
                cr += str(table.describe().to_html()) + delim
        if self.native_flag == True:
            print('='*50)
        else:
            cr += '='*50 + delim

        return cr


    def print_data_table_statistic_info(self, table_name):
        """
            print out basic statistic info of the table
        """
        table_index = self.datatable_names.index(str(table_name))
        print("table correlation : ", self.datatables[table_index].corr())