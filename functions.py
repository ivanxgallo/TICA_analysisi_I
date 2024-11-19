import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyemma.coordinates import tica
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

def normalize_dataset(dataset):
    # Inicializa el escalador MinMaxScaler
    scaler = MinMaxScaler()

    # Escala los valores del dataset manteniendo los índices
    dataset_scaled = dataset.copy()
    dataset_scaled.iloc[:, :] = scaler.fit_transform(dataset.values)

    return dataset_scaled


def get_eigen(df_i, rep=1, lag=50, dim=-1):
    #df_i = dataset.loc[dataset.index.get_level_values('R') == rep]
    X_scaled = df_i.values
    p_t_i = tica(X_scaled, lag=50, dim=-1)
    embedding_i = p_t_i.get_output()[0]
    eig_vect_i = p_t_i.eigenvectors
    eig_val_i = p_t_i.eigenvalues


    return eig_val_i, eig_vect_i

def get_vars_freq(dataset, reps=100, lag=50, n_contributions=3,
                    cut=0.1, criterion="at_least_one"):
    ds = normalize_dataset(dataset)
    imp_per_rep = {}  # dimensiones importantes por repeticion
    imp_all =[]  # todas las dimensiones importantes (se repiten por cada valor ppio de cada Simulación)
    imp = []  # Dims importantes
    imp_pondered = {}
    for i in range(reps):
        try:
            # Calculamos los valores y vectores propios
            ds = dataset.xs(i+1, level="R")
            eigen_vals, eigen_vecs = get_eigen(ds, rep=i+1, lag=lag)
            imp_per_rep[i+1] = {}

            # Definimos una lista que contendrá solo una vez una
            # variable importante para una repeticion i en particular
            a = []

            # Solo considero los 3 valores ppios mas grandes por defecto
            # pero este criterio puede cambiar con n_contributions
            for j in range(n_contributions):

                # Solo nos importa la magnitud de la contribución
                modulos = np.abs(eigen_vecs[:, j])

                # Definimos un criterio para elegir las variables más relevantes
                # Nuevamente este criterio se puede redefinir pero a mano
                umbral = np.mean(modulos) + 1 * np.std(modulos)
                indices_greater = np.where(modulos > umbral)
                imp_per_rep[i+1][j] = [dataset.columns[x] for x in indices_greater[0]]
                for id_column in indices_greater[0]:
                    col = dataset.columns[id_column]
                    imp_all.append(col)
                    a.append(col)
                    if col not in imp_pondered.keys():
                        imp_pondered[col] = modulos[id_column]*eigen_vals[j]
                    else:
                        imp_pondered[col] += modulos[id_column]*eigen_vals[j]

            for x in set(a):
                imp.append(x)

        except KeyError:
            pass

    # se podrían retornar también las demás,
    # pero para efectos de lo que presentaré (y usaré) solo me interesa esta
    if criterion == "at_least_one":
        imp_final = imp
        cutter = reps*cut
    elif criterion == "all_freq":
        imp_final = imp_all
        cutter = reps*n_contributions*cut
    elif criterion == "pondered":

        # Ordenar el diccionario por sus valores de mayor a menor
        # Y hacemos un corte en el 10% del maximo valor
        max_val = max(imp_pondered.values())
        sorted_filtered_vals = sorted(filter(lambda item: item[1] > max_val*cut, imp_pondered.items()),
                                key=lambda item: item[1], reverse=True)

        frequencies = dict(sorted_filtered_vals)
        return frequencies


    # Creamos un diccionario ordenado según la frecuencia
    # que se presenta en todas las repeticiones
    most_common = Counter(imp_final).most_common()
    # Filtramos para que no salgan frecuencias tan pequeñas
    filtered_common = filter(lambda x: x[1] > cutter, most_common)
    frequencies = dict(filtered_common)
    return frequencies


#------------------- FILTER EQUILIBRIUM CASES -----------------------#


def get_lowess_smoothing(t, x, frac=0.01, ini=None, fin=None):
    z = lowess(x, t, frac=frac)
    # Extraer los valores suavizados
    t_smoothed = z[:, 0]
    x_smoothed = z[:, 1]

    return t_smoothed, x_smoothed


def normalize_timeseries(series_in, abs=False):
    """Normaliza la serie entre 0 y 1."""
    if abs:
        """Normaliza la serie entre 0 y 1."""
        series = np.abs(series_in)
    else:
        series = np.array(series_in)
    return (series - series.min()) / (series.max() - series.min())

def get_derivative(t, x, win=10, abs=True):
    dx = []
    tt = []
    n_data = len(t)

    for i in range(0, n_data, win):
        try:
            diff = x[i+win] - x[i]

        except IndexError:
            diff = x[-1] - x[i]

        dx.append(diff)
        tt.append(t[i])

    dx = np.array(dx)#normalize_timeseries(dx, abs)
    tt = np.array(tt)

    return tt, dx


def get_lowess_derivative(t, x, frac = 0.1, win=30, ini=None, fin=None):
    z = lowess(t, x, frac=frac)
    tt, derivative = get_derivative(t[ini:fin], z[:,0][ini:fin], win=win)
    derivative = derivative/derivative.max()

    return tt, derivative


def find_convergence_time(t, x, criterion=0.15, t_max=45000):
    for i, val in enumerate(t):
        if val > t_max:
            idx_max = i
            break

    inverse_x = x[:idx_max][::-1]
    inverse_t = t[:idx_max][::-1]

    if inverse_x[0] > criterion:
        return None
    else:
        for i, val in enumerate(inverse_x):
            if val > criterion:
                return inverse_t[i]


def get_converged_cases(df_imp, col="rmsd_prot", exceptions=[], reps=100,
                        t_max=48000, criterion=0.15):
    convergence = []
    for rep in range(1, reps+1):
        df = df_imp.xs(rep, level='R')
        time_column = df.index.get_level_values('Time')
        x = df[col]/df[col].max()
        t = time_column
        t_sm, x_sm = get_lowess_derivative(t, x, ini=20, fin=None)
        convergence.append([rep, find_convergence_time(t_sm, x_sm, t_max=t_max, criterion=criterion)])

    chosen_reps = list(filter(lambda x: (x[1] is not None) and (x[1] < 40000) and (x[0] not in exceptions), convergence))

    return chosen_reps

def create_chosen_list(chosen_list, filename):

    # Crear y escribir en el archivo
    with open(filename, "w") as file:
        # Escribir el encabezado
        file.write("REP\tTIME\n")
        # Escribir los datos
        for row in chosen_list:
            file.write(f"{row[0]}\t{row[1]}\n")











