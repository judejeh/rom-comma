import time
from numpy import full
import romcomma.model as model
import numpy as np
import pandas as pd
from romcomma.data import Store, Fold, Frame
from romcomma.typing_ import NP, Union, Tuple, Sequence
from shutil import rmtree, copytree
from pathlib import Path
from scipy import optimize
from pyDOE import lhs
from scipy.stats.distributions import norm
from sklearn.metrics import r2_score


def store_and_fold(SOURCE: str, store_name: str, K: int, replace_empty_test_with_data_: bool = True, is_split: bool = True,
                   ORIGIN_CSV_PARAMETERS=None) -> Store:
    """
    Args:
        SOURCE: The source csv file name to be used as training data for the GP.
        store_name: The name of the folder where the GP will be stored.
        K: The amount of folds to be used in cross-validation.
        replace_empty_test_with_data_: Whether to replace an empty test file with the training data when K==1.
        is_split: Whether the store needs splitting for multiple outputs.
        ORIGIN_CSV_PARAMETERS: A dictionary stating the index column.
    Returns:
        A ``store'' object which contains a data_csv file, a meta_json file and a standard_csv file. The files contain the global dataset which have
        been split into folds and will be analysed.
    """
    if ORIGIN_CSV_PARAMETERS is None:
        ORIGIN_CSV_PARAMETERS = {'index_col': 0}
    store = Store.from_csv(BASE_PATH / store_name, BASE_PATH / SOURCE, **ORIGIN_CSV_PARAMETERS)
    store.standardize(standard=Store.Standard.mean_and_std)
    Fold.into_K_folds(store, K, shuffled_before_folding=True, standard=Store.Standard.none,
                      replace_empty_test_with_data_=replace_empty_test_with_data_)
    if is_split is True:
        store.split()
    return store


def run_GP(store: Store, GP_NAME_STEM: str = "ARD", optimize: bool = True, test: bool = True, sobol: bool = True, is_split: bool = True):
    """
    Args:
        store:
        GP_NAME_STEM:
        optimize:
        test:
        sobol:
        is_split:
    Returns:

    """
    # kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, store.M), 0.2, dtype=float))
    kernel_parameters = model.gpy_.Kernel.RationalQuadratic.Parameters(lengthscale=2.0, power=1) # attempt 1
    # kernel_parameters = model.gpy_.Kernel.RatQuad2.Parameters(lengthscale=2.0, power=1)  # attempt 2
    params = model.base.GP.Parameters(kernel=kernel_parameters, e_floor=-1.0E-3, f=10, e=0.1, log_likelihood=None)
    model.run.GPs(module=model.run.Module.GPY_, name=GP_NAME_STEM, store=store, M_Used=-1, parameters=params, optimize=optimize,
                  test=test, sobol=sobol, optimizer_options=model.gpy_.GP.DEFAULT_OPTIMIZER_OPTIONS)
    if test is True:
        model.run.collect_tests(store=store, model_name=GP_NAME_STEM, is_split=is_split)
    model.run.collect(store=store, model_name=GP_NAME_STEM, parameters=model.gpy_.GP.DEFAULT_PARAMETERS, is_split=is_split)
    model.run.collect(store=store, model_name=Path(GP_NAME_STEM) / model.gpy_.GP.KERNEL_NAME, parameters=kernel_parameters, is_split=is_split)
    model.run.collect(store=store, model_name=Path(GP_NAME_STEM) / model.base.Sobol.NAME, parameters=model.base.Sobol.DEFAULT_PARAMETERS,
                      is_split=is_split)
    return


def run_ROM(store: Store, ROM_NAME_STEM: str = "ROM", GP_NAME_STEM: str = "rom.reorder.optimized", It: int = 4, guess_It: int = -1, is_split: bool = False):
    """
    Args:
        store:
        ROM_NAME_STEM:
        GP_NAME_STEM:
        It:
        guess_It:
        is_split:
    Returns:

    """
    ROM_NAME = ROM_NAME_STEM + model.base.ROM.OPTIMIZED_GP_EXT
    kernel_parameters = model.gpy_.Kernel.ExponentialQuadratic.Parameters(lengthscale=full((1, store.M), 0.2, dtype=float))
    sobol_options = {'semi_norm': model.base.Sobol.SemiNorm.DEFAULT_META, 'N_exploit': 0, 'N_explore': 1,
                     'options': {'gtol': 1.0E-16}}
    # change N_exploit to below 1 for re-ordering of input basis only instead of rotation
    rom_options = dict(iterations=It, guess_identity_after_iteration=guess_It, sobol_optimizer_options=sobol_options,
                       gp_initializer=model.base.ROM.GP_Initializer.CURRENT_WITH_GUESSED_LENGTHSCALE,
                       gp_optimizer_options=model.run.Module.GPY_.value.GP.DEFAULT_OPTIMIZER_OPTIONS)
    model.run.ROMs(module=model.run.Module.GPY_, name=ROM_NAME_STEM, store=store, source_gp_name=GP_NAME_STEM, Mu=-1, Mx=-1,
                   optimizer_options=rom_options)
    model.run.collect(store=store, model_name=ROM_NAME_STEM, parameters=model.base.ROM.DEFAULT_PARAMETERS, is_split=is_split)
    model.run.collect(store=store, model_name=ROM_NAME, parameters=model.gpy_.GP.DEFAULT_PARAMETERS, is_split=is_split)
    model.run.collect(store=store, model_name=Path(ROM_NAME) / model.gpy_.GP.KERNEL_NAME, parameters=kernel_parameters,
                      is_split=is_split)
    model.run.collect(store=store, model_name=Path(ROM_NAME) / model.base.Sobol.NAME,
                      parameters=model.base.Sobol.DEFAULT_PARAMETERS, is_split=is_split)
    return


def Test_Data(gp_path: Union[str, Path], test_data_file: str, standardized: bool, k: int, standard_data_file: str = "__standard__.csv"):
    """ Loads and saves the test data as standardized inputs (X) and standardized observed outputs (Y).

    Args:
        k:
        standardized: Is the test data already standardised?
        gp_path: Path to the folder (inside a split) where the test data and standard data files are stored.
        test_data_file: The name of the test data file, e.g. 'test_data.csv'.
        standard_data_file: The name of the standard data file which contains the mean and standard deviation used to standardize the test data.
    Returns: A tuple containing the NP.Matrix of inputs (X) and the NP.Vector of outputs (Y).
    """

    GP_PATH = Path(gp_path)
    Split_Store = Store(GP_PATH)
    fold = Fold(Split_Store, k)
    fold_dir = fold.dir
    test_data = pd.read_csv(BASE_PATH / test_data_file, header=[0, 1], index_col=0)
    if standardized is True:
        Dataset = test_data.values
        Stand_Inputs = Dataset[:, :-1]
        Stand_Observed_Outputs = Dataset[:, -1]
        return Stand_Inputs, Stand_Observed_Outputs
    else:
        mean_std = pd.read_csv(GP_PATH / standard_data_file, index_col=0)
        Dataset = test_data.values
        AVG_STD = mean_std.values
        Stand_Dataset = (Dataset - AVG_STD[1].astype(float)) / AVG_STD[2].astype(float)
        Stand_Inputs = Stand_Dataset[:, :-1]
        Stand_Observed_Outputs = Stand_Dataset[:, -1]
        return Stand_Inputs, Stand_Observed_Outputs


def Test_Data_No_Outputs(gp_path: Union[str, Path], test_data_file: str, standardized: bool, k: int, standard_data_file: str = "__standard__.csv"):
    """ Loads and saves the test data as standardized inputs (X) and standardized observed outputs (Y).

    Args:
        k:
        standardized: Is the test data already standardised?
        gp_path: Path to the folder (inside a split) where the test data and standard data files are stored.
        test_data_file: The name of the test data file, e.g. 'test_data.csv'.
        standard_data_file: The name of the standard data file which contains the mean and standard deviation used to standardize the test data.
    Returns: A tuple containing the NP.Matrix of inputs (X) and the NP.Vector of outputs (Y).
    """
    GP_PATH = Path(gp_path)
    Split_Store = Store(GP_PATH)
    fold = Fold(Split_Store, k)
    fold_dir = fold.dir
    test_data = pd.read_csv(BASE_PATH / test_data_file, header=[0, 1], index_col=0)
    if standardized is True:
        Dataset = test_data.values
        Stand_Inputs = Dataset[:, :]
        return Stand_Inputs
    else:
        mean_std = pd.read_csv(GP_PATH / standard_data_file, index_col=0)
        Dataset = test_data.values
        AVG_STD = mean_std.values
        Stand_Dataset = (Dataset - AVG_STD[1].astype(float)) / AVG_STD[2].astype(float)
        Stand_Inputs = Stand_Dataset[:, :]
        return Stand_Inputs


def Predict(gp_path: Union[str, Path], gb_source: str, gb_destination: str, Rotated_Inputs: NP.Matrix, Us_taken: int, k: int = 0)\
            -> Tuple[NP.Matrix, NP.Matrix, NP.Tensor3]:
    """ Prediction using a GaussianBundle.

    Args:
        gp_path: The path that takes you into the trained Gaussian process' Split folder where the Fold can be found.
        gb_source: The name of the trained GB e.g. "ROM.optimized" or "ARD".
        gb_destination: The name of the new folder where the GB will be copied and predictions made e.g. "ROM_Predictions".
        Rotated_Inputs: The rotated inputs that are predicted on.
        Us_taken: The amount of Us taken from the ROM to be used in predicting.
        k: The fold that contains the data that has been used to train the GP's.
    Returns: A pair (predictive_mean, predictive_std) of (N,1) numpy arrays.

    """
    GP_PATH = Path(gp_path)
    Split_Store = Store(GP_PATH)
    fold = Fold(Split_Store, k, Us_taken)
    gb_dir = fold.dir / gb_destination
    rmtree(gb_dir, ignore_errors=True)
    copytree(src=fold.dir / gb_source, dst=gb_dir)
    gb = model.gpy_.GP(fold=fold, name=gb_destination, parameters=None)
    Predicted_Output = gb.predict(Rotated_Inputs)
    return Predicted_Output


def Create_CSVs(gp_path: Union[str, Path], gb_destination: str, Predicted_Outputs: Tuple[NP.Vector, NP.Vector, NP.Tensor3], Stand_Inputs: NP.Matrix,
                Rotated_Inputs: NP.Matrix, k: int = 0, csv_name: str = "__predictions__"):
    """ Saves the inputs, rotated inputs and predictions into a CSV file.

    Args:
        gp_path: The path that takes you into the trained Gaussian process' Split folder where the Fold can be found.
        gb_destination: The name of the new folder where the GB will be copied and predictions made e.g. "ROM_Predictions".
        Predicted_Outputs: A pair (predictive_mean, predictive_std) of (N,1) numpy arrays.
        Stand_Inputs: The standardized input values - an (N,M) numpy array, consisting of N test inputs, each of dimension M.
        Rotated_Inputs: The rotated inputs that are predicted on.
        k: The fold that contains the data that has been used to train the GP's.
        csv_name: The name of the csv file that has been created. Default is "__predictions__".
    Returns: A frame containing the inputs, the rotated inputs and predictions for each split directory.

    """
    GP_PATH = Path(gp_path)
    Split_Store = Store(GP_PATH)
    fold = Fold(Split_Store, k)
    gb_dir = fold.dir / gb_destination
    mean_prediction_T = Predicted_Outputs[0]
    std_prediction_T = np.sqrt(Predicted_Outputs[1])
    df_X = pd.DataFrame(Stand_Inputs)
    inputs_label = ['x' + str(i - 1) for i in np.arange(1, len(df_X.columns) + 1)]
    inputs_dict = dict(zip(df_X.columns, inputs_label))
    df_X = df_X.rename(columns=inputs_dict)
    list_input_header = list(df_X.columns.values)
    for idx, val in enumerate(list_input_header):
        list_input_header[idx] = 'Stand_Input'
    df_X.columns = pd.MultiIndex.from_tuples(zip(list_input_header, df_X.columns))
    """df_U = pd.DataFrame(Rotated_Inputs)
    rotated_label = ['u' + str(i - 1) for i in np.arange(1, len(df_U.columns) + 1)]
    rotated_dict = dict(zip(df_U.columns, rotated_label))
    df_U = df_U.rename(columns=rotated_dict)
    list_input_header = list(df_U.columns.values)
    for idx, val in enumerate(list_input_header):
        list_input_header[idx] = 'Rotated_Input'
    df_U.columns = pd.MultiIndex.from_tuples(zip(list_input_header, df_U.columns))"""
    df_mean = pd.DataFrame(mean_prediction_T)
    mean_label = ['Predicted_Mean']
    mean_dict = dict(zip(df_mean.columns, mean_label))
    df_mean = df_mean.rename(columns=mean_dict)
    df_std = pd.DataFrame(std_prediction_T)
    std_label = ['Predicted_STD']
    std_dict = dict(zip(df_std.columns, std_label))
    df_std = df_std.rename(columns=std_dict)
    df_outputs = pd.concat([df_mean, df_std], axis=1)
    list_output_header = list(df_outputs.columns.values)
    for idx, val in enumerate(list_output_header):
        list_output_header[idx] = 'Output'
    df_outputs.columns = pd.MultiIndex.from_tuples(zip(list_output_header, df_outputs.columns))
    df = pd.concat([df_X, df_outputs], axis=1)  # removed df_U
    frame = Frame(gb_dir / csv_name, df)
    return frame


def Create_CSV_with_Observed(gp_path: Union[str, Path], gb_destination: str, Predicted_Outputs: Tuple[NP.Matrix, NP.Matrix, NP.Tensor3],
                             Stand_Inputs: NP.Matrix, Rotated_Inputs: NP.Matrix, Stand_Observed_Outputs: NP.Vector, k: int = 0,
                             csv_name: str = "__predictions__.csv"):
    """ Saves the inputs, rotated inputs, predictions and observed output into a CSV file.

    Args:
        gp_path: The path that takes you into the trained Gaussian process' Split folder where the Fold can be found.
        gb_destination: The name of the new folder where the GB will be copied and predictions made e.g. "ROM_Predictions".
        Predicted_Outputs: A pair (predictive_mean, predictive_std) of (N,1) numpy arrays.
        Stand_Inputs: The standardized input values - an (N,M) numpy array, consisting of N test inputs, each of dimension M.
        Rotated_Inputs: The rotated inputs that are predicted on.
        Stand_Observed_Outputs: A NP.Vector of standardized observed outputs from the inputs.
        k: The fold that contains the data that has been used to train the GP's.
        csv_name: The name of the csv file that has been created. Default is "__predictions__".
    Returns: A frame containing the inputs, the rotated inputs, the predictions and the observed outputs.

    """
    GP_PATH = Path(gp_path)
    Split_Store = Store(GP_PATH)
    fold = Fold(Split_Store, k)
    gb_dir = fold.dir / gb_destination
    mean_prediction_T = Predicted_Outputs[0]
    std_prediction_T = np.sqrt(Predicted_Outputs[1])
    df_X = pd.DataFrame(Stand_Inputs)
    inputs_label = ['x' + str(i - 1) for i in np.arange(1, len(df_X.columns) + 1)]
    inputs_dict = dict(zip(df_X.columns, inputs_label))
    df_X = df_X.rename(columns=inputs_dict)
    list_input_header = list(df_X.columns.values)
    for idx, val in enumerate(list_input_header):
        list_input_header[idx] = 'Input'
    df_X.columns = pd.MultiIndex.from_tuples(zip(list_input_header, df_X.columns))
    """df_U = pd.DataFrame(Rotated_Inputs)
    rotated_label = ['u' + str(i - 1) for i in np.arange(1, len(df_U.columns) + 1)]
    rotated_dict = dict(zip(df_U.columns, rotated_label))
    df_U = df_U.rename(columns=rotated_dict)
    list_input_header = list(df_U.columns.values)
    for idx, val in enumerate(list_input_header):
        list_input_header[idx] = 'Input'
    df_U.columns = pd.MultiIndex.from_tuples(zip(list_input_header, df_U.columns))"""
    df_mean = pd.DataFrame(mean_prediction_T)
    mean_label = ['Predicted_Mean']
    mean_dict = dict(zip(df_mean.columns, mean_label))
    df_mean = df_mean.rename(columns=mean_dict)
    df_std = pd.DataFrame(std_prediction_T)
    std_label = ['Predicted_STD']
    std_dict = dict(zip(df_std.columns, std_label))
    df_std = df_std.rename(columns=std_dict)
    df_Y = pd.DataFrame(Stand_Observed_Outputs)
    Y_label = ['Observed_Output']
    Y_dict = dict(zip(df_Y.columns, Y_label))
    df_Y = df_Y.rename(columns=Y_dict)
    df_outputs = pd.concat([df_mean, df_std, df_Y], axis=1)
    list_output_header = list(df_outputs.columns.values)
    for idx, val in enumerate(list_output_header):
        list_output_header[idx] = 'Output'
    df_outputs.columns = pd.MultiIndex.from_tuples(zip(list_output_header, df_outputs.columns))
    df = pd.concat([df_X, df_outputs], axis=1)  # removed df_U
    frame = Frame(gb_dir / csv_name, df)
    return frame


def collect_predictions(store: Store, folder_name: str, csv_name: str, is_split: bool = True) -> Sequence[Path]:
    """Service routine to instantiate the collection of prediction results.

        Args:
            csv_name:
            store: The Store containing the global dataset to be analyzed.
            folder_name: The name of the folder where the results are being collected from.
            is_split: True or False, whether splits have been used in the model.
        Returns: The split directories collected.
    """
    final_frame = frame = None
    if is_split:
        final_destination = store.dir / folder_name
        final_destination.mkdir(mode=0o777, parents=True, exist_ok=True)
        splits = store.splits
    else:
        final_destination = None
        splits = [(None, store.dir)]
    for split in splits:
        split_store = Store(split[-1])
        K = split_store.meta['K']
        destination = split_store.dir / folder_name
        destination.mkdir(mode=0o777, parents=True, exist_ok=True)
        for k in range(K):
            fold = Fold(split_store, k)
            source = (fold.dir / folder_name) / csv_name
            #PARAMETERS = {'sep': ',',
            #              'header': [0],
            #              'index_col': 0, }
            result = Frame(source).df
            result.insert(0, "Fold", np.full(result.shape[0], k), True)
            # out = result.iloc[:, -1]
            # std = result.iloc[:, -2]
            # mean = result.iloc[:, -3]
            # result.iloc[:, -1] = (out - mean) / std
            if k == 0:
                frame = Frame(destination / csv_name, result.copy(deep=True))
            else:
                frame.df = pd.concat([frame.df, result.copy(deep=True)], axis=0, ignore_index=False)
        frame.write()
        if is_split:
            result = frame.df
            #rep = dict([(result['Predicted_Mean'].columns[0], "Observed_Output")])
            #result.rename(columns=rep, level=1, inplace=True)
            result.insert(0, "Split", np.full(result.shape[0], split[0]), True)
            # result = result.reset_index()
            if split[0] == 0:
                final_frame = Frame(final_destination / csv_name, result.copy(deep=True))
            else:
                final_frame.df = pd.concat([final_frame.df, result.copy(deep=True)], axis=0, ignore_index=True)
    if is_split:
        final_frame.write()
    return splits


def RMSE_Exp_Function(X):
    """
    A function that will calculate the RMSE between the predicted mean value and the experimental value given four modelling parameters, while 3
    experimental parameters are held constant at the the values set by the experiment sets directly paired with the experimental outputs.
    Args:
        X: a ndarray consisting of 4 elements, X[0], X[1], X[2], X[3]:
            Agglomeration_Custom_kernel_parameter_values_1_Rate_Type_I_II
            Breakage_Custom_selection_function_parameters_Granulate_Rate_coefficient
            Consolidation_custom_kinetic_parameter_values_Granulate_Rate_coefficient
            Critical_pore_saturation

    Returns: The RMSE value.

    """
    # label the files, paths and folders which will be needed
    # pred_folder = "Trial-ROM_Prediction_for_Optimization3" - this is only for when creating a new folder for example, may want to save predictions
    exp_data_file = "Stand_Exp_Data.csv"
    # set the initial sums to equal 0
    total_sum = 0
    # total_output = 0
    # split_sum = 0
    # give values to the variables which are held constant
    Mu = -1     # the number of dimensions the ROM will use for predictions
    M_exp = 3  # number of experimental input variables
    N_exp = 11          # number of experimental sets
    # Collect the standardised experimental data and split it into inputs and outputs
    Exp_Data = pd.read_csv(BASE_PATH / exp_data_file, header=[0, 1], index_col=0)
    Exp_Input_Values = Exp_Data.iloc[:, :M_exp]
    Exp_Output_Values = Exp_Data.iloc[:, M_exp:]
    # Create the standardized inputs to be used in the GP for predictions by combining the parameter variables and the experimental variables
    Param_Inputs = np.tile(np.array([X[0], X[1], X[2], X[3]]), (N_exp, 1))     # Ensure the initial parameter values are correct size
    Stand_Inputs = np.concatenate((Param_Inputs, Exp_Input_Values), axis=1)
    # begin the standard procedure of setting the store, the amount of splits and folds so it can loop through each GP.
    splits = store.splits
    K = store.meta['K']
    for split_dir in splits:
        GP_Path = Path(store_path / split_dir[1])
        for k in range(K):
            # in each fold, the procedure is to rotate the standardized inputs by theta, then use the ROM to predict on the rotated inputs
            Split_Store = Store(GP_Path)
            fold = Fold(Split_Store, k)
            fold_dir = fold.dir
            # find the folder Sobol which holds the CSV file with the theta values in
            """sobol = fold_dir / "ROM.optimized" / "sobol"
            theta = pd.read_csv(sobol / "Theta.csv", header=[0], index_col=0)
            theta_T = np.transpose(theta.values)
            Stand_Inputs.astype(float)
            M = store.M
            if 0 < Mu < M:
                Rotated_Inputs = Stand_Inputs @ theta_T[:, 0:Mu]
            else:
                Rotated_Inputs = Stand_Inputs @ theta_T
            # copy the ROM into a new folder and use it to predict on the rotated inputs
            # gb_dir = fold.dir / pred_folder
            # rmtree(gb_dir, ignore_errors=True)
            # copytree(src=fold.dir / "ROM.optimized", dst=gb_dir)"""
            gb = model.gpy_.GP(fold=fold, name="ARD", parameters=None)
            Predicted_Output = gb.predict(Stand_Inputs)
            # take the mean values.
            mean_output = Predicted_Output[0]
            # Ensure the correct split has the correct experimental output values in the correct shape (a column vector).
            if split_dir[0] == 0:
                exp_output = Exp_Output_Values.iloc[:, 0].values.reshape(-1, 1)
            elif split_dir[0] == 1:
                exp_output = Exp_Output_Values.iloc[:, 1].values.reshape(-1, 1)
            elif split_dir[0] == 2:
                exp_output = Exp_Output_Values.iloc[:, 2].values.reshape(-1, 1)
            elif split_dir[0] == 3:
                exp_output = Exp_Output_Values.iloc[:, 3].values.reshape(-1, 1)
            else:
                print("Error")
            error = exp_output - mean_output
            error_squared = error**2
            sum_error_squared = np.sum(error_squared)
            # calculate the sum of the error squares for every fold by adding them onto the previous fold_sum
            total_sum += sum_error_squared
    # calculate RMSE
    L = 4  # number of splits
    error_count = K*N_exp*L  # the total number of errors summed together
    # add a penalty for constraints x < 3.5 and x > -3.5
    # - for each parameter set, there is 11 exp sets, which predict through K folds and L outputs
    RMSE = np.sqrt(total_sum/error_count)
    return RMSE


if __name__ == '__main__':
    """# start = time.time()
    BASE_PATH = Path("X:\\comma_group1\\Rom\\dat\\AaronsTraining\\Prices\\LLOY\\Attempt_1")
    store_name = "1_Fold-GP_Only"
    store = store_and_fold("Data.csv", store_name, 1)
    run_GP(store)
    # end_GP_time = (time.time() - start) / 60
    # print("GP 1 Fold finished in {:.2f} minutes.".format(end_GP_time))
    X = Test_Data_No_Outputs(gp_path=Path(BASE_PATH / store_name), test_data_file="Test_Data.csv", standardized=True, k=0,
                             standard_data_file="__standard__.csv")
    Pred_Outputs = Predict(gp_path=Path(BASE_PATH / store_name / "split.0"), gb_source="ARD", gb_destination="GP_Forecasting", Rotated_Inputs=X,
                           Us_taken=-1, k=0)
    Create_CSVs(gp_path=Path(BASE_PATH / store_name / "split.0"), gb_destination="GP_Forecasting", Predicted_Outputs=Pred_Outputs,Stand_Inputs=X,
                Rotated_Inputs=X, k=0,csv_name="forecast_results_1.csv")
    collect_predictions(store=store, folder_name="GP_Forecasting", csv_name="forecast_results_1.csv", is_split=True)



    """
    # start = time.time()
    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Silica")
    store_name = "Rat_Quad_Trial"
    store = store_and_fold("Data_Inc_Amine.csv", store_name, 5)
    # store = Store(BASE_PATH / store_name)
    GP_name_stem = "RQF_Kernel"
    start_GP_time = time.time()
    run_GP(store=store, GP_NAME_STEM=GP_name_stem, optimize=True, test=True, sobol=True, is_split=True)
    end_GP_time = (time.time() - start_GP_time)/60
    print("GP finished in {:.2f} minutes.".format(end_GP_time))
    start_ROM_time = time.time()
    run_ROM(store=store, ROM_NAME_STEM="ROM_Reorder", GP_NAME_STEM=GP_name_stem, It=4, guess_It=-1, is_split=True)
    end_ROM_time = (time.time() - start_ROM_time) / 60
    print("ROM finished in {:.2f} minutes.".format(end_ROM_time))
    """
    BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Silica")
    store_name = "82_Exc_Amine"
    store = store_and_fold("Data_Exc_Amine.csv", store_name, 82)
    # store = Store(BASE_PATH / store_name)
    start_GP_time = time.time()
    run_GP(store)
    end_GP_time = (time.time() - start_GP_time)/60
    print("GP finished in {:.2f} minutes.".format(end_GP_time))
    start_ROM_time = time.time()
    run_ROM(store=store, ROM_NAME_STEM="ROM_Reorder", GP_NAME_STEM="ARD", It=4, guess_It=-1, is_split=True)
    end_ROM_time = (time.time() - start_ROM_time) / 60
    print("ROM finished in {:.2f} minutes.".format(end_ROM_time))"""
    """
    store_name = "Design_82folds"
    store = store_and_fold("Design.csv", store_name, 82)
    # store = Store(BASE_PATH / store_name)
    start_GP_time = time.time()
    run_GP(store)
    end_GP_time = (time.time() - start_GP_time)/60
    print("GP finished in {:.2f} minutes.".format(end_GP_time))

    store_name = "Actual_82folds"
    store = store_and_fold("Actual.csv", store_name, 82)
    # store = Store(BASE_PATH / store_name)
    start_GP_time = time.time()
    run_GP(store)
    end_GP_time = (time.time() - start_GP_time)/60
    print("GP finished in {:.2f} minutes.".format(end_GP_time))
    """
    """start_ROM_time = time.time()
    run_ROM(store=store, ROM_NAME_STEM="ROM", GP_NAME_STEM="rom.reorder.optimized", It=4, guess_It=-1, is_split=False)
    end_ROM_time = (time.time() - start_ROM_time) / 60
    print("ROM finished in {:.2f} minutes.".format(end_ROM_time))"""
    """
    store_name = "VSA"
    store = store_and_fold("Data-VSA.csv", store_name, 5)
    # store = Store(BASE_PATH / store_name)
    start_GP_time = time.time()
    run_GP(store)
    end_GP_time = (time.time() - start_GP_time)/60
    print("GP finished in {:.2f} minutes.".format(end_GP_time))

    store_name = "H5"
    store = store_and_fold("Data-H5.csv", store_name, 5)
    # store = Store(BASE_PATH / store_name)
    start_GP_time = time.time()
    run_GP(store)
    end_GP_time = (time.time() - start_GP_time)/60
    print("GP finished in {:.2f} minutes.".format(end_GP_time))

    store_name = "LCD"
    store = store_and_fold("Data-LCD.csv", store_name, 5)
    # store = Store(BASE_PATH / store_name)
    start_GP_time = time.time()
    run_GP(store)
    end_GP_time = (time.time() - start_GP_time)/60
    print("GP finished in {:.2f} minutes.".format(end_GP_time))

    store_name = "VF"
    store = store_and_fold("Data-VF.csv", store_name, 5)
    # store = Store(BASE_PATH / store_name)
    start_GP_time = time.time()
    run_GP(store)
    end_GP_time = (time.time() - start_GP_time)/60
    print("GP finished in {:.2f} minutes.".format(end_GP_time))
    """



    """
    store_name = "Total_Indices_x1"
    store = store_and_fold("Data_x1.csv", store_name, 1)
    # store = Store(BASE_PATH / store_name)
    start_GP_time = time.time()
    run_GP(store)
    end_GP_time = (time.time() - start_GP_time)/60
    print("GP 1 Fold finished in {:.2f} minutes.".format(end_GP_time))

    store_name = "Total_Indices_x2"
    store = store_and_fold("Data_x2.csv", store_name, 1)
    # store = Store(BASE_PATH / store_name)
    start_GP_time = time.time()
    run_GP(store)
    end_GP_time = (time.time() - start_GP_time)/60
    print("GP 1 Fold finished in {:.2f} minutes.".format(end_GP_time))

    store_name = "Total_Indices_x3"
    store = store_and_fold("Data_x3.csv", store_name, 1)
    # store = Store(BASE_PATH / store_name)
    start_GP_time = time.time()
    run_GP(store)
    end_GP_time = (time.time() - start_GP_time)/60
    print("GP 1 Fold finished in {:.2f} minutes.".format(end_GP_time))
    """


    """
    store_name2 = "1_Fold-GP_Only"
    store2 = store_and_fold("Data.csv", store_name2, 1)
    # store = Store(BASE_PATH / store_name)
    start_GP_time2 = time.time()
    run_GP(store2)
    end_GP_time2 = (time.time() - start_GP_time2) / 60
    print("GP 1 Fold finished in {:.2f} minutes.".format(end_GP_time2))
    """
    """
    store_name3 = "5_Folds-Rotated_GP"
    store3 = store_and_fold("Data.csv", store_name3, 5)
    # store = Store(BASE_PATH / store_name)
    start_GP_time3 = time.time()
    run_GP(store3)
    end_GP_time3 = (time.time() - start_GP_time3) / 60
    print("GP 1 Fold finished in {:.2f} minutes.".format(end_GP_time3))
    start_ROM_time = time.time()
    run_ROM(store3)
    end_ROM_time = (time.time() - start_ROM_time)/60
    print("ROM finished in {:.2f} minutes.".format(end_ROM_time))
    """


    # store_path = Path(BASE_PATH / store_name)
    """ # opt0
    start_Opt0_time = time.time()
    x0 = np.array([0, 0, 0, 0])
    RMSE0 = RMSE_Exp_Function(X=x0)
    print("Begin the first optimisation with initial guess of", x0, "resulting in an RMSE of", RMSE0)
    result0 = optimize.minimize(RMSE_Exp_Function, x0, method='BFGS', options={'gtol': 1e-10, 'disp': True})
    print(result0.x)
    print(result0.message)
    for i in range(len(result0.x)):
        hess_inv_i = result0.hess_inv[i, i]
        #ftol = 2.220446049250313e-09
        uncertainty_i = np.sqrt(hess_inv_i)
        print('x^{0} = {1:12.4e} ± {2:.3e}'.format(i, result0.x[i], uncertainty_i))
    end_Opt0_time = (time.time() - start_Opt0_time) / 60
    print("Opt_0 finished in {:.2f} minutes.".format(end_Opt0_time))
    print()
    print()
    # opt 1
    start_Opt1_time = time.time()
    x1 = result0.x
    RMSE1 = RMSE_Exp_Function(X=x1)
    print("Begin the second optimisation with initial guess of", x1, "resulting in an RMSE of", RMSE1)
    result1 = optimize.minimize(RMSE_Exp_Function, x1, method='BFGS', options={'gtol': 1e-10, 'disp': True})
    print(result1.x)
    print(result1.message)
    for i in range(len(result1.x)):
        hess_inv_i = result1.hess_inv[i, i]
        #ftol = 2.220446049250313e-09
        uncertainty_i = np.sqrt(hess_inv_i)
        print('x^{0} = {1:12.4e} ± {2:.3e}'.format(i, result1.x[i], uncertainty_i))
    end_Opt1_time = (time.time() - start_Opt1_time) / 60
    print("Opt_2 finished in {:.2f} minutes.".format(end_Opt1_time))
    print()
    print()
    # opt 2 
    start_Opt2_time = time.time()
    x2 = np.array([0.231727076, -0.952684399, 2.294705629, -0.536301781])
    RMSE2 = RMSE_Exp_Function(X=x2)
    print("Begin the second optimisation with initial guess of", x2, "resulting in an RMSE of", RMSE2)
    result2 = optimize.minimize(RMSE_Exp_Function, x2, method='BFGS', options={'gtol': 1e-10, 'disp': True})
    print(result2.x)
    print(result2.message)
    for i in range(len(result2.x)):
        hess_inv_i = result2.hess_inv[i, i]
        #ftol = 2.220446049250313e-09
        uncertainty_i = np.sqrt(hess_inv_i)
        print('x^{0} = {1:12.4e} ± {2:.3e}'.format(i, result2.x[i], uncertainty_i))
    end_Opt2_time = (time.time() - start_Opt2_time) / 60
    print("Opt_2 finished in {:.2f} minutes.".format(end_Opt2_time))
    print()
    print()
    # opt 3
    start_Opt3_time = time.time()
    x3 = np.array([-0.009361,   -1.23891363,  2.61768944,  0.02540098])
    RMSE3 = RMSE_Exp_Function(X=x3)
    print("Begin the least squares optimisation with initial guess of", x3, "resulting in an RMSE of", RMSE3)
    result3 = optimize.least_squares(RMSE_Exp_Function, x3)
    print(result3.x)
    print(result3.message)
    for i in range(len(result3.x)):
        hess_inv_i = result3.hess_inv[i, i]
        #ftol = 2.220446049250313e-09
        uncertainty_i = np.sqrt(hess_inv_i)
        print('x^{0} = {1:12.4e} ± {2:.3e}'.format(i, result3.x[i], uncertainty_i))
    end_Opt3_time = (time.time() - start_Opt3_time) / 60
    print("Opt_0 finished in {:.2f} minutes.".format(end_Opt3_time))
    print()
    print()

    total_time_mins = (time.time() - start) / 60
    print("Code finished in {:.2f} minutes.".format(total_time_mins))"""
