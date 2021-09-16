
"""
This a demo script used to show the use of GPs on the G-Sobol function.

Some terminology on the directories:
Base_Path: the base path is where the initial data is stored.
Store: inside the base path there maybe multiple stores depending on how the data has been "store_and_fold".
Split: the next directory will usually be the splits if the data has more than 1 output.
Fold: Each fold will then be found.
Models: The models can be the GP (e.g. "ARD") or a ROM (e.g. "ROM.optimized"). Each rotation of the ROM will be in here too (e.g. "ROM.0").
After that, files are collected together back down these directories.
"""


import time
import numpy as np
import pandas as pd
from pathlib import Path
from ROMCOMMA_Student_Code.romcomma import model
from ROMCOMMA_Student_Code.romcomma.data import Store, Fold, Frame


def g_sobol_to_csv(n_samples: int, dim: int = 5, upper_bound: float = 1.0, lower_bound: float = 0.0, csv_name: str = "g_sobol.csv"):
    """
    This uses the Sobol' G function as described in https://www.sfu.ca/~ssurjano/gfunc.html to create a csv file of training data ready to use
    :param n_samples: The number of samples used for each input variable in between the lower and upper bound. The total number of datapoints will be
    n_samples**dim.
    :param dim: The number of input variables used in the input grid.
    :param upper_bound: The upper bound is the maximum value the inputs can be.
    :param lower_bound: The lower bound is the lowest value the inputs can be.
    :param csv_name: Choose a sensible name for the csv file and ensure it includes ".csv",
    :return: A frame containing the input grid and the G_Sobol output as a csv file saved in BASE_PATH ready to train a GP.
    """
    # Create ndarray of input values.
    space = np.linspace(lower_bound, upper_bound, n_samples)
    a_tuple = (space,) * dim
    mesh = np.ravel(np.meshgrid(*a_tuple))
    mesh.shape = (dim, n_samples ** dim)
    x = np.transpose(mesh)
    # Save ndarray of input values as a dataframe with two labelled header columns (Input, Input, ..., Input) & (x1, x2, ..., xdim)
    df_x = pd.DataFrame(x)
    inputs_label = ['x' + str(i - 1) for i in np.arange(1, len(df_x.columns) + 1)]
    inputs_dict = dict(zip(df_x.columns, inputs_label))
    df_x = df_x.rename(columns=inputs_dict)
    list_input_header = list(df_x.columns.values)
    for idx, val in enumerate(list_input_header):
        list_input_header[idx] = "Input"
    df_x.columns = pd.MultiIndex.from_tuples(zip(list_input_header, df_x.columns))
    # Calculate the Sobol G function as described in https://www.sfu.ca/~ssurjano/gfunc.html using the ndarray of input values X.
    prod = 1
    a = np.linspace(1, dim, dim)
    for j in range(dim):
        j = int(j)
        xi = x[:, [j]]
        ai = a[j]
        new1 = np.abs(4 * xi - 2) + ai
        new2 = 1 + ai
        prod = prod * new1 / new2
    y = prod
    # Save(1 x dim) ndarray of output values as a dataframe with two labelled header columns (Output) & (y)
    df_y = pd.DataFrame(y)
    output_label = ['y' + str(i - 1) for i in np.arange(1, len(df_y.columns) + 1)]
    output_dict = dict(zip(df_y.columns, output_label))
    df_y = df_y.rename(columns=output_dict)
    list_output_header = list(df_y.columns.values)
    for idx, val in enumerate(list_output_header):
        list_output_header[idx] = "Output"
    df_y.columns = pd.MultiIndex.from_tuples(zip(list_output_header, df_y.columns))
    # concatenate both df's and save is as frame
    df = pd.concat([df_x, df_y], axis=1)
    frame = Frame(BASE_PATH / csv_name, df)
    return frame


def store_and_fold(source: str, store_name: str, k: int, replace_empty_test_with_data_: bool = True, is_split: bool = True,
                   origin_csv_parameters=None) -> Store:
    """
    Store and Fold is to be used to take a csv file and organise it ready for k-fold cross validation.
    Args:
        source: The source csv file name to be used as training data for the GP or ROM.
        store_name: The name of the folder where the data will be stored ready to train a GP or ROM.
        k: The amount of folds to be used in cross-validation.
        replace_empty_test_with_data_: Whether to replace an empty test file with the training data when K==1.
        is_split: Whether the store needs splitting for multiple outputs.
        origin_csv_parameters: A dictionary stating the index column.
    Returns:
        A ``store'' object which contains a data_csv file, a meta_json file and a standard_csv file. The files contain the global dataset which have
        been split into folds and will be analysed.
    """
    if origin_csv_parameters is None:
        origin_csv_parameters = {'index_col': 0}
    store = Store.from_csv(BASE_PATH / store_name, BASE_PATH / source, **origin_csv_parameters)
    store.standardize(standard=Store.Standard.mean_and_std)
    Fold.into_K_folds(store, k, shuffled_before_folding=True, standard=Store.Standard.none,
                      replace_empty_test_with_data_=replace_empty_test_with_data_)
    if is_split is True:
        store.split()
    return store


def run_gp(store: Store, gp_name: str = "ARD", model_name: str = "ARD", optimize: bool = True, test: bool = True,
           sobol: bool = True, is_split: bool = True, kerneltypeidentifier: str = "gpy_.ExponentialQuadratic"):
    """
    Args:
        store:
        gp_name: The GP name.
        model_name: The name of the Model where the results are being collected.
        optimize:
        test:
        sobol:
        is_split:
        kerneltypeidentifier:
    Returns:
    """
    # params = model.base.GP.Parameters(kernel=kernel_parameters, e_floor=-1.0E-2, f=1, e=0.01, log_likelihood=None)
    gp_optimizer_options = {'optimizer': 'bfgs', 'max_iters': 10000, 'gtol': 1E-32}
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=np.full((1, STORE.M), 1, dtype=float))
    parameters = model.gpy_.GP.DEFAULT_PARAMETERS._replace(kernel=kernel_parameters, e_floor=-1E-2, e=0.01)
    model.run.GPs(module=model.run.Module.GPY_, name=gp_name, store=store, M_Used=-1, parameters=parameters, optimize=optimize,
                  test=test, sobol=sobol, optimizer_options=gp_optimizer_options)
    model.run.collect_GPs(store=store, model_name=model_name, test=test, sobol=sobol, is_split=is_split, kernelTypeIdentifier=kerneltypeidentifier)
    return


if __name__ == '__main__':
    start_time = time.time()
    BASE_PATH = Path("Z:\\comma_group1\\User\\fcp18asy\\PhD Work\\Python\\Data\\GSA_Toy_Data\\Demos")
    g_sobol_to_csv(n_samples=5, dim=5, csv_name="G_Sobol.csv")
    STORE_NAME = "G_Sobol-5_Folds"
    STORE = store_and_fold("G_Sobol.csv", STORE_NAME, 5, is_split=False)
    # STORE = Store(BASE_PATH / STORE_NAME)
    run_gp(store=STORE, gp_name="ARD", model_name="ARD", optimize=True, test=True, sobol=True, is_split=False,
           kerneltypeidentifier="gpy_.ExponentialQuadratic")
    total_time = (time.time() - start_time)/60
    print("Code finished running in {:.2f} minutes.".format(total_time))