import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import time 
import numpy as np 
import winsound
import os
import pickle
import gc

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)[:4] == 'bool'  :
            continue
            
        if col_type != object and str(col_type)[:3] != 'str'  :
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def zamiana_wartosci_w_kolumnie_map_dict(df,column_name,map_dict):
    for klucz, wartosc in map_dict.items():
        mask = (df[column_name] == klucz)
        df.loc[mask, column_name] = wartosc

    return(df)

def zapisz_model_do_pliku(model_name,model,zbiorcza_nazwa_katalogu):
    import joblib
    if model_name =='model_lgbm_reg' : 
        nazwa_pliku_modelu = model_name + '.txt'
        # Tworzenie ścieżki do katalogu 'submissions' i nowego katalogu
        path = os.path.join('submissions', zbiorcza_nazwa_katalogu, model_name)
        
        # Tworzenie katalogów, jeśli nie istnieją
        os.makedirs(path, exist_ok=True)
        
        # Zapisywanie modelu LightGBM do pliku
        model.booster_.save_model(os.path.join(path, nazwa_pliku_modelu))
        
    elif 'Neuron_Network' not in model_name: 
        nazwa_pliku_modelu = model_name+'.pkl'
        # Tworzenie ścieżki do katalogu 'submissions' i nowego katalogu
        path = os.path.join('submissions', zbiorcza_nazwa_katalogu, model_name)
        
        # Tworzenie katalogów, jeśli nie istnieją
        os.makedirs(path, exist_ok=True)
        
        # Zapisywanie modelu do pliku w nowym katalogu
        with open(os.path.join(path, nazwa_pliku_modelu), 'wb') as f:
            pickle.dump(model, f)
    else: 

        nazwa_pliku_modelu = model_name + '.keras'
        path = os.path.join('submissions', zbiorcza_nazwa_katalogu, model_name)
        os.makedirs(path, exist_ok=True)
        
        # Save the model using Keras' built-in method
        model.save(os.path.join(path, nazwa_pliku_modelu))
        
        nazwa_pliku_modelu = model_name + '.h5'

        os.makedirs(path, exist_ok=True)
        model.save(os.path.join(path, nazwa_pliku_modelu))
    print('model_name',model_name)

    return(nazwa_pliku_modelu)

def zapisz_DF_do_Excel(df_dane_modelu ,nazwa ,model_name,model,zbiorcza_nazwa_katalogu):
    path = os.path.join('submissions', zbiorcza_nazwa_katalogu, model_name)
    # Tworzenie katalogów, jeśli nie istnieją
    os.makedirs(path, exist_ok=True)
    df_dane_modelu.to_excel(os.path.join(path, nazwa+'.xlsx'), index=False)

def transform (df, data_name , print_transform = False):

    map_dict = {'Yes':1,'No':0}
    df = zamiana_wartosci_w_kolumnie_map_dict(df_train_test,'Vehicle_Damage',map_dict)
    
    map_dict = {'Male':1,'Female':0}
    df = zamiana_wartosci_w_kolumnie_map_dict(df_train_test,'Gender',map_dict)
    
    map_dict = {'1-2 Year':0,'< 1 Year':1,'> 2 Years':2}
    df = zamiana_wartosci_w_kolumnie_map_dict(df_train_test,'Vehicle_Age',map_dict)
    
    print('1 ',df.info())

    df['Gender'] = df['Gender'].astype('bool')
    df['Driving_License'] = df['Driving_License'].astype('bool')
    df['Previously_Insured'] = df['Previously_Insured'].astype('bool')
    df['Vehicle_Damage'] = df['Vehicle_Damage'].astype('bool')
    df['Vehicle_Age'] = df['Vehicle_Age'].astype('int8')
    
    df['Region_Code'] = df['Region_Code'].astype('int16')    
    df['Annual_Premium'] = df['Annual_Premium'].astype('int32')
    df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('int16')   
    
    print('2 ',df.info())

    print(df.head())

    return(df)

def zapis_kopi_pliku_z_programem(path):
    # =============================================================================
    # ZAPIS KOPI PLIKU Z PROGRAMEM
    # =============================================================================

    import os
    import shutil
    
    # Ścieżka do pliku, który chcesz skopiować
    current_file = os.path.basename(__file__)
    print(current_file)
    
    # Ścieżka, gdzie chcesz skopiować plik
#    path = 'ścieżka_do_folderu_docelowego'  # Zmień na właściwą ścieżkę
    destination_file = os.path.join(path, current_file)
    
    # Sprawdzenie, czy plik już istnieje
    if not os.path.exists(destination_file):
        # Kopiowanie pliku
        shutil.copy(current_file, destination_file)
        print(f"Plik {current_file} został skopiowany do {destination_file}.")
        return(True)
    else:
        print(f"Plik {current_file} już istnieje w {destination_file}. Kopiowanie pominięte.")
        return(False)

def zestawienie_wynikow (df_wyniki,zbiorcza_nazwa_katalogu, X_train, y_train,
                         X_valid, y_valid,X_test,
                         model_name,model,param={},
                         p_test=None,scoring=None,execution_time=None, 
                         fitted_lambda_train=None):

    start_time = time.time()


    # =============================================================================
    #         PARAMETRY MODELU
    # =============================================================================

    if scoring is None:

        model.fit(X_train, y_train,**param)
        p_train  = model.predict_proba(X_train)[:, 1]
        p_valid  = model.predict_proba(X_valid)[:, 1]
    
        end_time = time.time()
        execution_time = end_time - start_time
        
        from sklearn.metrics import roc_auc_score
        auc_roc = roc_auc_score(y_valid, p_valid)
    else :
        auc_roc = scoring        
        

# =============================================================================
#     ZAPIS MODELU DO PLIKU
# =============================================================================
    nazwa_pliku_modelu = zapisz_model_do_pliku(model_name,model,zbiorcza_nazwa_katalogu) 
# =============================================================================
#         PARAMETRY MODELU
# =============================================================================
    model_parameters = str(model.get_all_params())
    df_wyniki.loc[len(df_wyniki)] = [model_name, auc_roc,
                                     model_parameters,
                                     nazwa_pliku_modelu,
                                     execution_time]

# =============================================================================
#     ZAPIS WYNIKÓW DO PLIKU CSV
# =============================================================================
    zapis_predykcji_do_pliku(zbiorcza_nazwa_katalogu,model_name, 
                             nazwa_pliku_modelu,p_test)
    
# =============================================================================
#     ZAPIS TABELKI DO PLIKU XLS
# =============================================================================
    path = os.path.join('submissions', zbiorcza_nazwa_katalogu)
    df_wyniki.to_excel(os.path.join(path, 'wyniki.xlsx'), index=False)
# =============================================================================
# ZAPIS KOPI PLIKU Z PROGRAMEM
# =============================================================================
    pierwsza_proba_zapisu = zapis_kopi_pliku_z_programem(path)
    if pierwsza_proba_zapisu :
        global split_valid_size
        global wynik_column
        df_split = pd.DataFrame(columns=['valid_size','split_random_state','X_train.info()'])
        df_split.loc[len(df_split)] = [split_valid_size,
                                       split_random_state,
                                       str(X_train.info())]
        
        df_split.to_excel(os.path.join(path, 'dane_pliku.xlsx'), index=False)

    return (df_wyniki)

def zapis_predykcji_do_pliku(zbiorcza_nazwa_katalogu, model_name ,
                             plik = None, p_test = None ):
    global id_column
    global wynik_column
    global df_test_received
    
    path = os.path.join('submissions', zbiorcza_nazwa_katalogu, model_name)

    if not os.path.exists(path):
        os.makedirs(path)

    if plik is not None :
        print('zapis_predykcji_do_pliku plik \n',' model_name ',model_name)
        if model_name =='model_lgbm_reg':
            nazwa_pliku_modelu = model_name + '.txt'
            plik_path = os.path.join(path, nazwa_pliku_modelu)
            try:
                # Wczytaj model LightGBM z pliku
                loaded_model = lgbm.Booster(model_file=plik_path)
            except FileNotFoundError:
                print(f'zpdp Nie można otworzyć pliku: {plik_path}')
                print('zpdp Zawartość katalogu:')
                print(os.listdir(os.path.dirname(plik_path)))
            except Exception as e:
                print(f'zpdp Wystąpił błąd: {e}')
        
        elif 'Neuron_Network' not in model_name:
            nazwa_pliku_modelu = model_name + '.pkl'
            plik_path = os.path.join(path, nazwa_pliku_modelu)
            print('zpdp plik_path ',plik_path)
            try:
                # Wczytaj model z pliku
                with open(plik_path, 'rb') as f:
                    loaded_model = pickle.load(f)
                print('zpdp udało sie załadować plik ',nazwa_pliku_modelu)
            except FileNotFoundError:
                print(f'zpdp Nie można otworzyć pliku: {plik_path}')
                print('zpdp Zawartość katalogu:')
                print(os.listdir(os.path.dirname(plik_path)))
            except Exception as e:
                print(f'zpdp Wystąpił błąd: {e}')
        
        else:
            nazwa_pliku_modelu = model_name + '.keras'
            try:
                # Wczytaj model z pliku
                loaded_model = tf.keras.models.load_model(nazwa_pliku_modelu)
                print('zpdp udało sie załadować plik ',plik)
            except Exception as e:
                print(f'zpdp Wystąpił błąd podczas ładowania modelu Keras: {e}')
    
    if p_test is None :
        #y_pred_test = loaded_model.predict(X_test)

        y_pred_test = loaded_model.predict_proba(X_test)[:, 1]
        print('zpdp y_pred_test\n',y_pred_test)
    else :
        y_pred_test = p_test
    
    output = pd.DataFrame({id_column:indexy_test, wynik_column: y_pred_test})

    print(model_name,'output.head()\n',output.head())
    output.to_csv(os.path.join(path, 'submission.csv'), index=False)
    
def podstawowe_dane_df(df_trein_R, df_test_R ,podstawowe_print):

    df = df_trein_R
    
    pelne_wyswietlanie_tabelek = True
    if pelne_wyswietlanie_tabelek :
    # Ustawienie opcji wyświetlania
        pd.set_option('display.max_columns', None)  # Pokaż wszystkie kolumny
        pd.set_option('display.expand_frame_repr', False)  # Zapobiegaj łamaniu wierszy

    # Nie trzeba tworzyć słownika jeżeli jasne jest co jest w kolumnach
    dict_columns={}
    
    # Podzielenie kolumn ze względu na typ danych numeryczne kategoryczne
    list_num_columns = ['Age','Annual_Premium','Policy_Sales_Channel','Vintage']

    list_cat_columns = ['Gender','Driving_License','Region_Code','Previously_Insured',
                         'Vehicle_Damage','Vehicle_Age']
    
    if (len(list_num_columns) + len(list_cat_columns))!= len(df.columns.values)-1:
        
        print('list_num_columns ', list_num_columns)
        print('len(list_num_columns) ',len(list_num_columns))
        print('list_cat_columns ',list_cat_columns)
        print('len(list_cat_columns) ',len(list_cat_columns))
        print('len(df.columns.values) ', len(df.columns.values))
        print('df.columns.values ', df.columns.values)
        
        input('podstawowe_dane_df cos sie nie zgadza')
    
    return(list_num_columns,list_cat_columns,dict_columns)

def koniec():
    import winsound
    import time
    
    E5 = (659, 250)
    D5 = (587, 250)
    C5 = (523, 250)
    F5 = (698, 250)
    G5 = (784, 250)
    A5 = (880, 250)
    Bb = (932, 250)  # Nuta B (si bemol)
    D6 = (1175, 250) # 1174.66
    A4 = (440, 250)
    # Definiuj nuty (częstotliwość w Hz) i ich długość (czas trwania w milisekundach)
    notes = [
    D5, F5, D5, D5, D5, G5, D5, C5, D5,
        A5, D5, D5, D5, Bb, A5, F5, D5,
        A5, D6, F5, D5, A5, D6, D5, C5,
        C5, C5, A4, E5, D5,
    ]
    
    
    # Odtwarzaj nuty z różnymi przerwami
    licznik =0
    for freq, duration in notes:
        
        winsound.Beep(freq, duration)
        if licznik+1 < len(notes) and notes[licznik] == notes[licznik+1]:
            sleep_time = 0.05
            time.sleep(sleep_time)
        elif freq == D5[0] or freq == A5[0]:
            sleep_time = 0.3
            time.sleep(sleep_time)  # Dłuższa pauza po nutach D5 i A5
        else:
            sleep_time = 0.2
            time.sleep(sleep_time)  # Krótsza pauza po pozostałych nutach
        # print('notes[licznik] ',notes[licznik],' sleep_time ',sleep_time)
        licznik +=1
    print("Koniec odtwarzania melodi Axel F w tonacji C-moll!")
    print('KONIEC KONIEC KONIEC KONIEC KONIEC KONIEC KONIEC KONIEC KONIEC KONIEC')

def start():
    # Define the notes (frequency in Hz) and their durations (milliseconds)
    notes = [
        (523, 500),  # C5
        (587, 500),  # D5
        (659, 500),  # E5
        (698, 500),  # F5
        (784, 500),  # G5
        (880, 500),  # A5
        (988, 500),  # B5
        (1047, 500)  # C6
    ]
    
    # Play the notes
    for freq, duration in notes:
        winsound.Beep(freq, duration)
        time.sleep(0.1)  # Short pause between notes
        
    

# =============================================================================
#     ZMIENNE GLOBALNE ZMIENNE GLOBALNE ZMIENNE GLOBALNE ZMIENNE GLOBALNE 
# =============================================================================
id_column = 'id'
wynik_column = 'Response'
df_test_received = pd.read_csv(r"C:\kursy\Jupiter_files\kaggle_files\playground-series-s4e7\test.csv")
df_test_received = df_test_received.set_index(id_column)
indexy_test = df_test_received.index.values

# =============================================================================
# SPLIT PARAMETRY
# =============================================================================

split_valid_size =0.25
split_random_state =42

# =============================================================================
# MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN MAIN 
# =============================================================================

if __name__ == '__main__':
    
    # =============================================================================
    # DANE
    # =============================================================================
    # Tworzenie nazwy katalogu z datą, godziną i słowem 'submission'
    zbiorcza_nazwa_katalogu = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_submission')
    print('df_train_received')
    # Read CSV train data file into DataFrame
    df_train_received = pd.read_csv(r"C:\kursy\Jupiter_files\kaggle_files\playground-series-s4e7\train.csv")
    df_train_received = df_train_received.set_index(id_column)
    df_train_received = df_train_received[:int(len(df_train_received)/1)]
    # =============================================================================
    # ANALIZA PODSTAWOWYCH DANYCH
    # =============================================================================\
    print('podstawowe dane')
    # =============================================================================
    # SPLIT DATA
    # =============================================================================
    
    print('SPLIT DATA SPLIT DATA SPLIT DATA SPLIT DATA SPLIT DATA SPLIT DATA ')
    
    X_train_full_data = df_train_received.drop([wynik_column],axis = 1)
    y_train_full_data = df_train_received[wynik_column].astype('int8')

    X_train_split, X_valid_split, y_train, y_valid = train_test_split(
        X_train_full_data, y_train_full_data,
        test_size=split_valid_size,
        random_state=split_random_state,
        stratify=y_train_full_data
    )
    
    indexy_train = X_train_split.index.values
    indexy_valid = X_valid_split.index.values
    
    del X_train_split
    del X_valid_split 
    del X_train_full_data 
    # =============================================================================
    # Łączenie df TEST i TRAIN
    # =============================================================================
    df_train_test = pd.concat([df_train_received, df_test_received])
    df_train_test = df_train_test.drop([wynik_column],axis = 1)
    #, random_state=42
    # =============================================================================
    # TRANSFORM ALL_DATA, TRAIN_DATA, VALID_DATA, TEST_DATA
    # =============================================================================
    del df_train_received 
    del df_test_received 
    
    
    
    df_wyniki = pd.DataFrame(columns=['Model','roc_auc_score', 
                                          'model_parameters',
                                          'plik','execution_time'])
        
    df_all_data = transform(df_train_test,
                                  'ALL DATA',print_transform = True)

    X_train= df_all_data.loc[indexy_train]
    y_train= y_train_full_data.loc[indexy_train].values

    X_valid= df_all_data.loc[indexy_valid]
    y_valid= y_train_full_data.loc[indexy_valid].values

    X_test= df_all_data.loc[indexy_test]

    
    df_all_data = None
    df_train_test = None
    
    
    
    
    catboost_clas = True
    
    catboost_clas_tuning = False
    folding_model = True
    print('X_train.shape ',X_train.shape)
    start()
    if catboost_clas :
        start_time = time.time()
        wartosci_pozytywne  = y_train.sum()
        wartosci_negatywne  = len(y_train)- y_train.sum()
        scale_pos_weight    = int(wartosci_negatywne/wartosci_pozytywne)

        
        

    # =============================================================================
    # CatBoostClassifier
    # ============================================================================= 
        print('\nooo____catboost________oooooooooooooooooo ')
        
        
        from catboost import CatBoostClassifier
        
# =============================================================================
#         cat_params = {
#             'loss_function': 'Logloss',
#             'eval_metric': 'AUC',
#             'class_names': [0, 1],
#             'learning_rate': 0.075,
#             'iterations': 3000,
#             'depth': 9,
#             'random_strength': 0,
#             'l2_leaf_reg': 0.5,
#             'max_leaves': 512,
#             'fold_permutation_block': 64,
#             'task_type': 'GPU',
#             'random_seed': 42,
#             'verbose': False,
#             'allow_writing_files': False
#         }
# =============================================================================
        cat_params = {
            'loss_function': 'Logloss',
            #'loss_function': 'QueryCrossEntropy',

            #'eval_metric': 'AUC',
            'eval_metric': 'CrossEntropy',
            'scale_pos_weight' :scale_pos_weight,

            #'class_names': [0, 1],
            'learning_rate': 0.075,
            'iterations': 15000,
            'depth': 7,
            #'random_strength': 0,
            'l2_leaf_reg': 3,
            'bagging_temperature':0,
            #'max_leaves': 512,
            #'fold_permutation_block': 64,
            'task_type': 'GPU',
            #'random_seed': 42,
            #'verbose': False,
            #'plot':True,
            #'metric_period':10
            #'allow_writing_files': False
        }

        from catboost import Pool
        list_cat_columns = X_train.columns.values
        X_train_pool = Pool(X_train.astype(str), y_train, cat_features = list_cat_columns)
        X_valid_pool = Pool(X_valid.astype(str), y_valid, cat_features = list_cat_columns)
        X_test_pool  = Pool(X_test.astype(str), cat_features= list_cat_columns)

        
        cat_clf = CatBoostClassifier(**cat_params)
# =============================================================================
#         cat_clf.eval_metrics(data=X_valid_pool, 
#                              metrics=['Logloss','AUC'],
#                              eval_period=10)
# =============================================================================
        cat_clf = cat_clf.fit(X=X_train_pool,
                              eval_set=X_valid_pool,
                              verbose=500,
                              early_stopping_rounds=200)
        
                                        
        p_valid  = cat_clf.predict_proba(X_valid_pool)[:, 1]
        


        
        if np.isnan(p_valid).any():
            print("p_valid contains NaN values")
            print("Indices of NaN in p_valid:", np.where(np.isnan(p_valid)))
            
        
        from sklearn.metrics import roc_auc_score
        auc_roc = roc_auc_score(y_valid, p_valid)
        print('auc_roc',auc_roc)
        
        p_test  = cat_clf.predict_proba(X_test_pool)[:, 1]
        
        zapis_predykcji_do_pliku(zbiorcza_nazwa_katalogu, 'catboost_kagel_scale_pos_weight' ,
                               plik = None, p_test = p_test)
        end_time = time.time()
        execution_time = (end_time - start_time)/60
        zestawienie_wynikow(df_wyniki=df_wyniki, zbiorcza_nazwa_katalogu=zbiorcza_nazwa_katalogu, 
                            X_train=X_train, y_train=y_train,
                            X_valid=X_valid, y_valid=y_valid, X_test=X_test,
                            model_name='model_catboost_kagel_scale_pos_weight', model=cat_clf,
                            p_test=p_test,
                            scoring =auc_roc,execution_time=execution_time)
        
        
        
    if catboost_clas_tuning:
        
        winsound.PlaySound('Welcome.wav', winsound.SND_FILENAME) 
        
        
        from catboost import CatBoostClassifier
        from catboost import Pool

        
        iterations = [3000]
       # learning_rates = [0.07, 0.055, 0.05  ]
        #depths=[5,6]
        learning_rates = [ 0.05,0.75]
        depths=[6,8,9]
        l2_leaf_regs=[0.5, 6]
        bagging_temperatures=[1]
        border_counts= [254]
        
        wartosci_pozytywne  = y_train.sum()
        wartosci_negatywne  = len(y_train)- y_train.sum()
        scale_pos_weight    = int(wartosci_negatywne/wartosci_pozytywne)
        
        list_cat_columns = X_train.columns.values
        X_train_pool = Pool(X_train.astype(str), y_train, cat_features = list_cat_columns)
        X_valid_pool = Pool(X_valid.astype(str), y_valid, cat_features = list_cat_columns)
        X_test_pool  = Pool(X_test.astype(str), cat_features= list_cat_columns)
        
        print('Pool    Pool    Pool')
        best_auc_roc = 0
        best_params = {}
        df_model_tuning = pd.DataFrame()
        #y_valid_cb, _ = transform_Y(y_valid, fitted_lambda_train)
        start_time = time.time()
        licznik=0

        df_model_tuning = pd.DataFrame(columns=['best_auc_roc','cv','iteration',
                    'learning_rate','depth','bagging_temperature','l2_leaf_reg','border_count'])
        df_params = pd.DataFrame()
        
        for iterations_ in iterations:
            print('iterations ',iterations_)
            for lr in learning_rates:
                for depth in depths:
                    depth_start_time = time.time()
                    #print('depth ',depth)
                    for l2_leaf_reg_ in l2_leaf_regs:
                        #print('l2_leaf_reg ',l2_leaf_reg_)
                        for bagging_temperature_ in bagging_temperatures:
                            for border_count in border_counts:
                                #print('lr ',lr)
                                #for border_count_ in border_counts :
                                licznik= licznik+1
                                print(licznik,' lr ',lr,' depth ',depth,' l2_leaf_reg ',l2_leaf_reg_,' bagging_temperature_ ',bagging_temperature_)
                                model = CatBoostClassifier(learning_rate=lr, 
                                                           depth=depth, 
                                                          l2_leaf_reg=l2_leaf_reg_,
                                                          iterations=iterations_,
                                                          bagging_temperature=bagging_temperature_,
                                                          border_count=border_count,
                                                          # task_type='GPU',
                                                          eval_metric='AUC',
                                                          scale_pos_weight =scale_pos_weight,
                                                          early_stopping_rounds=200, verbose=500,
                                                          )
    
                                model.fit(X_train_pool, eval_set=(X_valid_pool))
                                p_valid  = model.predict_proba(X_valid_pool)[:, 1]
                                cv_mean = 0
                                
                                if np.isnan(p_valid).any():
                                    print("p_valid contains NaN values")
                                    print("Indices of NaN in p_valid:", np.where(np.isnan(p_valid)))
                                    
                                
                                from sklearn.metrics import roc_auc_score
                                auc_roc = roc_auc_score(y_valid, p_valid)
                                
                                model_params = {'learning_rate': lr, 
                                               'depth': depth,
                                               'iteration': iterations_,
                                               'bagging_temperature':bagging_temperature_,
                                                'l2_leaf_reg': l2_leaf_reg_,
                                                'border_count':border_count
                                               }
    # =============================================================================
    #                              df_model_tuning = pd.DataFrame(columns=['best_auc_roc','cv','iteration',
    #                                          'learning_rate','depth','bagging_temperature','l2_leaf_reg'])
    # =============================================================================
                                          
                                df_model_tuning.loc[len(df_model_tuning)] = [auc_roc,0,iterations_,lr,depth,bagging_temperature_,l2_leaf_reg_,border_count]
    
                                zapisz_DF_do_Excel(df_model_tuning ,'df_model_tuning' ,'model_catboost_T',model,zbiorcza_nazwa_katalogu)
                                  
                                if auc_roc > best_auc_roc:
                                    
                                    best_auc_roc = auc_roc
                                    model_catboost_T = model
                                    best_params=model_params
                                    print(f"Najlepszy wynik best_auc_roc : {best_auc_roc}")
                                    print(f"Najlepsze parametry: {best_params}")
                                    winsound.Beep(1000, 500)
                                    
                                del model
                    depth_end_time=time.time()
                    execution_depth_time = depth_end_time - depth_start_time
                    print('execution_depth_time ',execution_depth_time/60,' min')
        
            end_time=time.time()
            execution_time = (end_time - start_time)/60
            
            p_test  = model_catboost_T.predict_proba(X_test_pool)[:, 1]
            
            zestawienie_wynikow(df_wyniki=df_wyniki, zbiorcza_nazwa_katalogu=zbiorcza_nazwa_katalogu, 
                                X_train=X_train, y_train=y_train,
                                X_valid=X_valid, y_valid=y_valid, X_test=X_test,
                                model_name='model_catboost_T', model=model_catboost_T,
                                param = { 'eval_set':(X_valid, y_valid)},p_test=p_test,
                                scoring =best_auc_roc,execution_time=execution_time)
    
    
    # =============================================================================
    #             del X_train_pool, X_valid_pool
    # =============================================================================
    
        gc.collect()
    if folding_model :
        
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        scores = []
        
        #params = (model_catboost_T.get_all_params())
        
        y_train_df = pd.DataFrame(y_train)
        y_train_df.columns=[wynik_column]

        y_train_df_ind = pd.DataFrame(indexy_train,columns=[id_column])
        y_train_df = y_train_df_ind.join(y_train_df)

        y_train_df.set_index(id_column, inplace=True)

        
        start_time = time.time()
        FOLDS = 2
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
        test_preds = np.zeros((len(X_test), FOLDS), dtype=np.float32)
        valid_preds = np.zeros((len(X_valid), FOLDS), dtype=np.float32)
        from catboost import CatBoostClassifier
        from catboost import Pool
        wartosci_pozytywne  = y_train.sum()
        wartosci_negatywne  = len(y_train)- y_train.sum()
        scale_pos_weight    = int(wartosci_negatywne/wartosci_pozytywne)
        
# =============================================================================
#         model = CatBoostClassifier(learning_rate=0.05, 
#                                    depth=6, 
#                                   l2_leaf_reg=6,
#                                   iterations=10000,
#                                   bagging_temperature=1,
#                                   border_count=254,
#                                   # task_type='GPU',
#                                   eval_metric='AUC',
#                                   scale_pos_weight =scale_pos_weight,
#                                   early_stopping_rounds=200, verbose=500,
#                                   )
# =============================================================================
        model = cat_clf#CatBoostClassifier(**params, verbose=False)
        print('model ok')
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_df)):
            print("#" * 50)
            print(f"# Fold {fold + 1}")
            print("#" * 50)


            X_train_fold = X_train.iloc[train_idx]
            print(X_train_fold.shape)
            print(y_train_df.iloc[train_idx].shape)
            y_train_fold = y_train_df.iloc[train_idx].values

            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train_df.iloc[val_idx].values


            list_cat_columns = X_train.columns.values

# =============================================================================
#             X_train_pool = Pool(X_train_fold.astype(int), y_train_fold, cat_features = list_cat_columns)
#             X_val_pool   = Pool(X_val_fold.astype(int), y_val_fold, cat_features = list_cat_columns)
#             X_test_pool  = Pool(X_test.astype(int), cat_features= list_cat_columns)
# =============================================================================
            X_train_pool = Pool(X_train_fold, y_train_fold, cat_features = list_cat_columns)
            X_val_pool   = Pool(X_val_fold, y_val_fold, cat_features = list_cat_columns)
            X_test_pool  = Pool(X_test, cat_features= list_cat_columns)
            # model_catboost_T
            

            model.fit(X=X_train_pool, 
                      eval_set=X_val_pool, 
                      verbose=500, 
                      #eval_metric='AUC',
                      early_stopping_rounds=200
                      
                     )

            test_preds[:, fold]  = model.predict_proba(X_test_pool)[:, 1]
            
            valid_preds = model.predict_proba(X_val_pool)[:, 1]
            auc_roc = roc_auc_score(y_val_fold, valid_preds)
            scores.append(auc_roc)
            print('scores  ',scores)
            
            del X_train_fold, y_train_fold
            del X_val_fold, y_val_fold
            del X_train_pool, X_val_pool, X_test_pool
            
            gc.collect()
        
            

            p_test = test_preds.mean(axis=1)
    

        # =============================================================================
        #         
        #         zapis_predykcji_do_pliku(zbiorcza_nazwa_katalogu, 'catboost_fold' ,
        #                                      plik = None, p_test = test_preds.mean(axis=1))
        # =============================================================================
            score = sum(scores) / len(scores)
            end_time = time.time()
            execution_time = (end_time - start_time)/60
            print(f"Czas wykonania: {execution_time} min")

        # =============================================================================
        #     zestawienie_wynikow(df_wyniki, zbiorcza_nazwa_katalogu, X_train, y_train,
        #                         X_valid, y_valid, X_test,
        #                         'model_catboost_fold', model,p_test ,auc_roc,execution_time)
        # =============================================================================
            zestawienie_wynikow(df_wyniki=df_wyniki, zbiorcza_nazwa_katalogu=zbiorcza_nazwa_katalogu, 
                                X_train=X_train, y_train=y_train,
                                X_valid=X_valid, y_valid=y_valid, X_test=X_test,
                                model_name='model_catboost_fold', model=model,
                                p_test=p_test,
                                scoring =scores,execution_time=execution_time)
        del model

    koniec()


