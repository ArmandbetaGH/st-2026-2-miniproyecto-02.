"""Miniproyecto 2: aprendizaje automático y profundo contra el mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y las tablas en resultados/. El informe,
docs/informe.md, cita las cifras y las figuras que produce este guion.
"""
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from torch import nn  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, rmse)

# Después de importar statsmodels, que al cargarse fija sus propios avisos.
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

SEMILLA = 2026
N_PRUEBA = 24          # los meses del backtest, los mismos de comun.backtest

# El último modelo que ajustó cada función modelo(entrenamiento, h). Sirve
# para contar sus parámetros después del backtest.
ULTIMO = {}

# Error absoluto medio por mes del año de cada modelo en el backtest, y su
# sesgo: pronóstico menos real, promediado en todos los meses del backtest.
ERRORES_POR_MES = {}
SESGOS = {}

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

# ============================================================
# 1. La serie y el modelo a superar
# ============================================================
# La serie del miniproyecto 1, con su ficha en datos/README.md.
ARCHIVO = "datos/ventas_calzado_mensual_V2.csv"
FECHA = "mes"
VALOR = "pares_netos"
PERIODO = 12


def preparar(serie):
    """La preparación del miniproyecto 1, sin cambios: valida la serie mensual
    y conserva todos los registros observados.

    La serie no tiene huecos, así que no se imputa nada (si los tuviera, el
    guion se detiene: no se imputan automáticamente). El criterio IQR (1,5
    veces el rango intercuartílico) no marca ningún mes; los meses bajos de
    2025 se conservan, porque no hay soporte comercial para corregirlos.
    """
    serie = serie.sort_index().astype(float)
    if serie.index.has_duplicates:
        raise ValueError("La serie tiene meses duplicados: revisar el CSV.")
    faltantes = int(serie.isna().sum())
    if faltantes:
        raise ValueError(f"Hay {faltantes} meses sin dato: requieren decisión documentada "
                         "antes de ejecutar modelos. No se imputan automáticamente.")
    if not np.isfinite(serie.to_numpy()).all():
        raise ValueError("La serie contiene valores no finitos.")
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    limite_inf, limite_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    marcados = serie[(serie < limite_inf) | (serie > limite_sup)]
    print(f"  preparación: {faltantes} meses sin dato; IQR con límites "
          f"[{limite_inf:.1f}, {limite_sup:.1f}] marca {len(marcados)} meses; "
          f"se conservan todos los valores")
    return serie


# El SARIMA del miniproyecto 1: los órdenes salieron de la rejilla por AIC
# con los 24 meses anteriores al primer origen; d = D = 0 con constante.
ORDEN = (0, 0, 0)
ORDEN_ESTACIONAL = (1, 0, 0, PERIODO)
# Escala de trabajo del miniproyecto 1: se estima en decenas de pares y se
# vuelve a pares antes de medir; no cambia el patrón ni las métricas.
ESCALA = 10.0


def _ajustar_arima(entrenamiento, orden, orden_estacional):
    """El mismo ajuste del miniproyecto 1 (statsmodels ARIMA, máxima
    verosimilitud, constante cuando d = D = 0)."""
    from statsmodels.tsa.arima.model import ARIMA
    tendencia = "c" if orden[1] == 0 and orden_estacional[1] == 0 else "n"
    return ARIMA(entrenamiento.astype(float) / ESCALA, order=orden,
                 seasonal_order=orden_estacional, trend=tendencia,
                 enforce_stationarity=True, enforce_invertibility=True,
                 ).fit(method_kwargs={"maxiter": 120})


def modelo_clasico(serie):
    """El mejor modelo clásico del miniproyecto 1: SARIMA(0,0,0)(1,0,0)_12 con
    constante, y_t = c + Φ y_{t-12} + ε_t, reestimado en cada origen. Sin
    transformación y sin exógenas (el SARIMAX con festivos quedó peor)."""
    def modelo(entrenamiento, h):
        ajuste = _ajustar_arima(entrenamiento, ORDEN, ORDEN_ESTACIONAL)
        ULTIMO["modelo"] = ajuste
        return np.asarray(ajuste.forecast(steps=h), dtype=float) * ESCALA
    return modelo


NOMBRE_CLASICO = "SARIMA(0,0,0)(1,0,0,12) con constante"


def graficar_serie(serie):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy())
    eje.axvline(serie.index[-N_PRUEBA], color="gray", linestyle="--")
    eje.set_title(f"{VALOR}: serie mensual; a la derecha de la línea, los meses del backtest")
    eje.set_ylabel(VALOR)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. La tabla supervisada
# ============================================================
# Lo que aprende cada modelo: "nivel" es la serie tal cual; "dif12" es la
# diferencia con el mismo mes del año anterior. Se elige "nivel": con 48
# meses, la diferencia estacional se come los primeros 12 y los 12 rezagos
# otros 12, de modo que en el primer origen (24 meses) la tabla de "dif12"
# quedaría sin filas. justificar_objetivo imprime esas cifras.
OBJETIVO = "nivel"
REZAGOS = 12


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t) si z es la diferencia."""
    return y.iloc[-PERIODO] + z if OBJETIVO == "dif12" else z


def tabla_supervisada(z):
    """Una fila por mes t con:

    - rezago_k = z(t-k), k = 1..12;
    - mes = el mes del año de t (1 a 12), que se conoce de antemano;
    - media_3, media_12 y desv_12: media de z(t-1)..z(t-3), media y
      desviación de z(t-1)..z(t-12).

    Ninguna columna de la fila t usa z(t): los rezagos se corren con shift(k)
    y las ventanas se calculan sobre la serie ya corrida un mes.
    """
    X = pd.DataFrame(index=z.index)
    for k in range(1, REZAGOS + 1):
        X[f"rezago_{k}"] = z.shift(k)
    X["mes"] = z.index.month
    pasado = z.shift(1)
    X["media_3"] = pasado.rolling(3).mean()
    X["media_12"] = pasado.rolling(12).mean()
    X["desv_12"] = pasado.rolling(12).std()
    return X


def comprobar_sin_fuga(serie):
    """Prueba en el código que ninguna fila ve el mes que predice.

    Para cada mes t se altera y(t) (se multiplica por 10 y se le suma 1000),
    se rehace la tabla y se exige que la fila t quede idéntica. Como control,
    se cuenta en cuántos casos la alteración sí llega a filas posteriores: si
    nunca llegara, la prueba no estaría probando nada.
    """
    base = tabla_supervisada(objetivo(serie))
    llega_despues = 0
    for t in range(len(serie)):
        alterada = serie.copy()
        alterada.iloc[t] = 10 * alterada.iloc[t] + 1000
        tabla = tabla_supervisada(objetivo(alterada))
        if not np.allclose(base.iloc[t].to_numpy(float), tabla.iloc[t].to_numpy(float),
                           equal_nan=True):
            raise AssertionError(f"la fila de {serie.index[t]:%Y-%m} ve su propio mes")
        if not np.allclose(base.iloc[t + 1:].to_numpy(float), tabla.iloc[t + 1:].to_numpy(float),
                           equal_nan=True):
            llega_despues += 1
    print(f"  sin fuga: se alteró cada uno de los {len(serie)} meses y ninguna fila "
          f"cambió en su propio mes; la alteración sí cambió filas posteriores en "
          f"{llega_despues} casos")


def filas_completas(y):
    z = objetivo(y)
    X = tabla_supervisada(z)
    completas = X.notna().all(axis=1) & z.notna()
    return X[completas], z[completas]


def recursivo(ajustado, y, h):
    """Pronóstico recursivo a h meses.

    Para cada mes siguiente: se agrega a y como NaN, se arma su fila (que
    solo usa meses anteriores), se pronostica el objetivo, se lleva al nivel
    con a_nivel y se escribe como si fuera dato, de modo que entra como
    rezago_1 del mes siguiente. Basta con la cola de la serie: la fila de un
    mes usa a lo sumo 2 * PERIODO meses hacia atrás.
    """
    y = y.iloc[-(REZAGOS + 3 * PERIODO):].copy()
    pronosticos = []
    for fecha in fechas_siguientes(y, h):
        y.loc[fecha] = np.nan
        X = tabla_supervisada(objetivo(y)).iloc[[-1]]
        y_hat = float(a_nivel(y, float(np.asarray(ajustado.predict(X))[0])))
        y.loc[fecha] = y_hat
        pronosticos.append(y_hat)
    return np.array(pronosticos)


def con_tabla(fabrica):
    """Modelo para el backtest: arma la tabla con el entrenamiento del origen,
    ajusta un modelo nuevo y pronostica h meses en modo recursivo. El
    escalador vive dentro del modelo, así que se ajusta en cada origen."""
    def modelo(entrenamiento, h):
        X, z = filas_completas(entrenamiento)
        ajustado = fabrica()
        ajustado.fit(X, z)
        ULTIMO["modelo"] = ajustado
        return recursivo(ajustado, entrenamiento, h)
    return modelo


def justificar_objetivo(serie):
    """Las cifras detrás de OBJETIVO, con los meses anteriores al backtest."""
    global OBJETIVO
    elegido = OBJETIVO
    for opcion in ("nivel", "dif12"):
        OBJETIVO = opcion
        primero = len(filas_completas(serie.iloc[:-N_PRUEBA])[0])
        ultimo = len(filas_completas(serie.iloc[:-12])[0])
        print(f"  OBJETIVO = {opcion}: {primero} filas en el primer origen y {ultimo} en el último")
    OBJETIVO = elegido
    previo = serie.iloc[:-N_PRUEBA]
    debajo = int((serie.iloc[-N_PRUEBA:] < previo.min()).sum())
    anual = serie.groupby(serie.index.year).sum()
    print(f"  {debajo} de los {N_PRUEBA} meses del backtest quedan por debajo del mínimo "
          f"anterior ({previo.min():.0f}), que es el piso de un árbol que aprende el nivel")
    print("  pares por año: " + ", ".join(f"{a}: {v:.0f}" for a, v in anual.items()))


# ============================================================
# 3. Aprendizaje automático
# ============================================================
# Las cuatro fábricas devuelven un modelo nuevo sin entrenar; con_tabla
# ajusta uno por origen. KNN y SVR llevan el escalador dentro del pipeline;
# el SVR escala además el objetivo, porque su epsilon está en las unidades
# del objetivo. Los dos escaladores se ajustan en cada fit.

def knn(n_neighbors=5):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))


def svr(C=1.0, epsilon=0.1):
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=epsilon)),
        transformer=StandardScaler())


def bosque(max_depth=None, min_samples_leaf=1):
    return RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, random_state=SEMILLA)


def xgboost(max_depth=3, learning_rate=0.1):
    return XGBRegressor(n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                        random_state=SEMILLA)


# Rejillas pequeñas y con valores pensados para pocas filas: KNN con 1 a 5
# vecinos (el primer corte entrena con 6 filas), árboles poco profundos.
REJILLAS = {
    "KNN": (knn, {"n_neighbors": [1, 2, 3, 5]}),
    "SVR": (svr, {"C": [0.3, 1.0, 3.0, 10.0], "epsilon": [0.05, 0.1, 0.3]}),
    "Random Forest": (bosque, {"max_depth": [2, 3, None], "min_samples_leaf": [1, 2, 4]}),
    "XGBoost": (xgboost, {"max_depth": [1, 2, 3], "learning_rate": [0.03, 0.1, 0.3]}),
}

# Validación temporal solo con los meses anteriores al primer origen del
# backtest: 24 meses, que dan 12 filas (2023-01 a 2023-12). Tres cortes que
# entrenan con 6, 8 y 10 filas y validan cada uno los 2 meses siguientes.
VALIDACION = TimeSeriesSplit(n_splits=3, test_size=2)


def sin_prefijo(parametros):
    return {nombre.split("__")[-1]: valor for nombre, valor in parametros.items()}


def prefijo(estimador):
    """El prefijo de los hiperparámetros del último paso para GridSearchCV."""
    if isinstance(estimador, TransformedTargetRegressor):
        return "regressor__" + prefijo(estimador.regressor)
    if hasattr(estimador, "steps"):
        return estimador.steps[-1][0] + "__"
    return ""


def ajustar_hiperparametros(serie):
    """Para cada modelo de REJILLAS, la combinación con menor RMSE en
    validación temporal, con la tabla de serie.iloc[:-N_PRUEBA]: ninguno de
    los meses del backtest decide."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    print(f"  rejilla con {len(X)} filas, de {X.index[0]:%Y-%m} a {X.index[-1]:%Y-%m}; "
          f"TimeSeriesSplit de {VALIDACION.n_splits} cortes de {VALIDACION.test_size} meses")
    elegidos, filas = {}, []
    for nombre, (fabrica, rejilla) in REJILLAS.items():
        p = prefijo(fabrica())
        busqueda = GridSearchCV(fabrica(), {p + k: v for k, v in rejilla.items()},
                                cv=VALIDACION, scoring="neg_root_mean_squared_error")
        busqueda.fit(X, z)
        elegidos[nombre] = sin_prefijo(busqueda.best_params_)
        rmse_val = -busqueda.best_score_
        print(f"  {nombre}: {elegidos[nombre]}, RMSE de validación {rmse_val:.2f} "
              f"({len(busqueda.cv_results_['params'])} combinaciones)")
        filas.append({"modelo": nombre, "elegido": elegidos[nombre],
                      "RMSE_validacion": round(rmse_val, 2)})
    pd.DataFrame(filas).to_csv(RESULTADOS / "hiperparametros.csv", index=False)
    return elegidos


def graficar_importancia(serie, elegidos):
    """Importancia de las variables de Random Forest y XGBoost ajustados con
    lo elegido sobre los meses anteriores al backtest."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    figura, ejes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    importancias = {}
    for eje, (nombre, fabrica) in zip(ejes, [("Random Forest", bosque), ("XGBoost", xgboost)]):
        ajustado = fabrica(**elegidos[nombre]).fit(X, z)
        imp = pd.Series(ajustado.feature_importances_, index=X.columns)
        importancias[nombre] = imp
        eje.barh(imp.index[::-1], imp.to_numpy()[::-1])
        eje.set_title(nombre)
        eje.set_xlabel("importancia (fracción de la reducción del error)")
        mayores = imp.sort_values(ascending=False).head(3)
        print(f"  {nombre}: las tres variables más importantes son "
              + ", ".join(f"{v} ({i:.2f})" for v, i in mayores.items()))
    figura.suptitle("Importancia de las variables, ajustados con los meses anteriores al backtest")
    figura.tight_layout()
    figura.savefig(FIGURAS / "importancia.png", dpi=120)
    plt.close(figura)
    pd.DataFrame(importancias).round(4).to_csv(RESULTADOS / "importancia.csv")


# ============================================================
# 4. Aprendizaje profundo en PyTorch
# ============================================================
# Redes pequeñas y pocas épocas: el guion completo tiene que correr en CPU
# en pocos minutos.
EPOCAS_MAXIMAS = 300
TASA = 0.01
# En Colab con GPU la red y sus tensores van a cuda; sin GPU, a la CPU. Red se
# encarga de los dos: las clases de cada red no tienen que mover nada.
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


class Red:
    """Una red de PyTorch con fit y predict, para entrar a la tabla como un
    modelo de scikit-learn. Escala X y z con lo que ve en fit, que es el
    entrenamiento del origen, y devuelve el pronóstico en la escala original.
    Con secuencia=True la red recibe los rezagos en orden, con forma
    (lote, 12, 1), del más viejo al más reciente."""

    def __init__(self, construir, secuencia=False, epocas=EPOCAS_MAXIMAS, semilla=SEMILLA):
        self.construir = construir
        self.secuencia = secuencia
        self.epocas = epocas
        self.semilla = semilla

    def entradas(self, X):
        if self.secuencia:
            columnas = [f"rezago_{k}" for k in range(REZAGOS, 0, -1)]
            x = (X[columnas].to_numpy() - self.media) / self.desv
            return torch.tensor(x[:, :, None], dtype=torch.float32, device=DISPOSITIVO)
        return torch.tensor(self.escala_x.transform(X), dtype=torch.float32, device=DISPOSITIVO)

    def fit(self, X, z, X_val=None, z_val=None):
        torch.manual_seed(self.semilla)
        self.media, self.desv = z.mean(), z.std()
        self.escala_x = StandardScaler().fit(X)
        x = self.entradas(X)
        objetivo_t = torch.tensor(((z - self.media) / self.desv).to_numpy()[:, None],
                                  dtype=torch.float32, device=DISPOSITIVO)
        self.red = self.construir(x.shape[-1]).to(DISPOSITIVO)
        optimizador = torch.optim.Adam(self.red.parameters(), lr=TASA)
        perdida = nn.MSELoss()
        self.perdidas = []
        for _ in range(self.epocas):
            self.red.train()
            optimizador.zero_grad()
            error = perdida(self.red(x), objetivo_t)
            error.backward()
            optimizador.step()
            fila = [error.item()]
            if X_val is not None:
                fila.append(float(np.mean(((self.predict(X_val) - z_val) / self.desv) ** 2)))
            self.perdidas.append(fila)
        return self

    def predict(self, X):
        self.red.eval()
        with torch.no_grad():
            salida = self.red(self.entradas(X)).cpu().numpy()[:, 0]
        return salida * self.desv + self.media


class MLP(nn.Module):
    def __init__(self, entradas, ocultas=16):
        super().__init__()
        self.capas = nn.Sequential(nn.Linear(entradas, ocultas), nn.ReLU(),
                                   nn.Linear(ocultas, 1))

    def forward(self, x):
        return self.capas(x)


class LSTM(nn.Module):
    """nn.LSTM de una capa sobre la secuencia (lote, 12, 1) y una capa lineal
    que lee el último estado oculto."""

    def __init__(self, ocultas=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=ocultas, batch_first=True)
        self.salida = nn.Linear(ocultas, 1)

    def forward(self, x):
        estados, _ = self.lstm(x)
        return self.salida(estados[:, -1, :])


class Conv1D(nn.Module):
    """La tercera red: una convolución 1D sobre la secuencia (lote, 12, 1).

    Cada filtro de ancho 3 recorre los 12 rezagos con los mismos pesos y
    detecta un patrón local (una subida o una caída de tres meses); una capa
    lineal combina lo que cada filtro encontró en cada posición."""

    def __init__(self, filtros=8, ancho=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=filtros, kernel_size=ancho)
        self.salida = nn.Linear(filtros * (REZAGOS - ancho + 1), 1)

    def forward(self, x):
        mapas = torch.relu(self.conv(x.transpose(1, 2)))   # (lote, filtros, 10)
        return self.salida(mapas.flatten(1))


# Las candidatas: nombre, cómo se construye y si recibe la secuencia.
# Tamaños pequeños a propósito: en el último origen la tabla tiene 24 filas.
CANDIDATAS = {
    "MLP (4 ocultas)": (lambda n: MLP(n, 4), False),
    "MLP (16 ocultas)": (lambda n: MLP(n, 16), False),
    "LSTM (8 ocultas)": (lambda n: LSTM(8), True),
    "Conv1D (4 filtros)": (lambda n: Conv1D(4, 3), True),
}

# Meses de validación de las redes, al final de la tabla anterior al backtest.
MESES_VALIDACION_RED = 4


def validar_redes(serie):
    """Las épocas de cada red con validación temporal anterior al backtest.

    La tabla de serie.iloc[:-N_PRUEBA] (12 filas) se parte en entrenamiento
    y sus últimos MESES_VALIDACION_RED meses de validación. Cada candidata se entrena EPOCAS_MAXIMAS
    épocas y se guarda la época de menor pérdida de validación. Entre los
    MLP queda el de menor pérdida de validación.
    """
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    v = MESES_VALIDACION_RED
    X_ent, z_ent, X_val, z_val = X.iloc[:-v], z.iloc[:-v], X.iloc[-v:], z.iloc[-v:]
    print(f"  validación: entrena con {len(X_ent)} filas ({X_ent.index[0]:%Y-%m} a "
          f"{X_ent.index[-1]:%Y-%m}) y valida con {X_val.index[0]:%Y-%m} a {X_val.index[-1]:%Y-%m}")
    curvas, mejores = {}, {}
    for nombre, (construir, secuencia) in CANDIDATAS.items():
        ajustada = Red(construir, secuencia).fit(X_ent, z_ent, X_val, z_val)
        perdidas = np.array(ajustada.perdidas)
        epocas = int(np.argmin(perdidas[:, 1])) + 1
        curvas[nombre] = perdidas
        mejores[nombre] = (epocas, perdidas[epocas - 1, 1])
        parametros = sum(p.numel() for p in ajustada.red.parameters())
        print(f"  {nombre}: {parametros} parámetros; menor pérdida de validación "
              f"{perdidas[epocas - 1, 1]:.3f} en la época {epocas} "
              f"(entrenamiento {perdidas[epocas - 1, 0]:.3f}; en la época {EPOCAS_MAXIMAS}: "
              f"entrenamiento {perdidas[-1, 0]:.3f}, validación {perdidas[-1, 1]:.3f})")

    figura, ejes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for eje, (nombre, perdidas) in zip(ejes.ravel(), curvas.items()):
        eje.plot(range(1, len(perdidas) + 1), perdidas[:, 0], label="entrenamiento")
        eje.plot(range(1, len(perdidas) + 1), perdidas[:, 1], label="validación")
        eje.axvline(mejores[nombre][0], color="gray", linestyle="--",
                    label=f"época elegida ({mejores[nombre][0]})")
        eje.set_yscale("log")
        eje.set_title(nombre)
        eje.set_ylabel("MSE (escala estandarizada)")
        eje.legend(fontsize=8)
    for eje in ejes[-1]:
        eje.set_xlabel("época")
    figura.suptitle(f"Curvas de pérdida: validación con los {v} meses anteriores al backtest")
    figura.tight_layout()
    figura.savefig(FIGURAS / "perdida.png", dpi=120)
    plt.close(figura)

    mlps = [nombre for nombre in CANDIDATAS if nombre.startswith("MLP")]
    mlp = min(mlps, key=lambda nombre: mejores[nombre][1])
    print(f"  MLP elegido: {mlp}")
    return {nombre: mejores[nombre][0] for nombre in CANDIDATAS
            if not nombre.startswith("MLP") or nombre == mlp}


def red(nombre, epocas, semilla=SEMILLA):
    """La fábrica de una red para con_tabla."""
    construir, secuencia = CANDIDATAS[nombre]
    return lambda: Red(construir, secuencia, epocas, semilla)


SEMILLAS = [SEMILLA + i for i in range(5)]


def repetir_con_semillas(serie, nombre, epocas):
    """La mejor red con cinco semillas, cada una por el backtest."""
    filas = []
    for semilla in SEMILLAS:
        _, promedio = backtest(serie, con_tabla(red(nombre, epocas, semilla)))
        filas.append({"semilla": semilla, "RMSE": round(promedio["RMSE"], 2),
                      "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2)})
        print(f"  {nombre}, semilla {semilla}: RMSE {promedio['RMSE']:.2f}")
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTADOS / "semillas.csv", index=False)
    print(f"  cinco semillas: RMSE medio {tabla['RMSE'].mean():.2f}, "
          f"desviación {tabla['RMSE'].std():.2f} (mín {tabla['RMSE'].min():.2f}, "
          f"máx {tabla['RMSE'].max():.2f})")
    return tabla


# ============================================================
# 5. La tabla y la elección
# ============================================================

def contar_parametros(ajustado):
    """Cuántos números guarda el modelo ajustado. En los clásicos y las redes
    son los parámetros; en KNN, los valores que memoriza; en SVR, los vectores
    de soporte y sus coeficientes; en los árboles, los nodos."""
    if hasattr(ajustado, "params"):
        return len(ajustado.params)
    if isinstance(ajustado, Red):
        return sum(p.numel() for p in ajustado.red.parameters())
    if isinstance(ajustado, TransformedTargetRegressor):
        ajustado = ajustado.regressor_
    final = ajustado.steps[-1][1] if hasattr(ajustado, "steps") else ajustado
    if isinstance(final, KNeighborsRegressor):
        return final.n_samples_fit_ * final.n_features_in_
    if isinstance(final, SVR):
        return final.support_vectors_.size + final.dual_coef_.size + 1
    if isinstance(final, RandomForestRegressor):
        return sum(arbol.tree_.node_count for arbol in final.estimators_)
    if isinstance(final, XGBRegressor):
        return len(final.get_booster().trees_to_dataframe())
    return 0


def fila(nombre, serie, modelo):
    """Pasa el modelo por el backtest común y guarda, además, su error por mes
    del año (sin tocar comun.backtest: solo se anotan los pronósticos)."""
    ULTIMO.clear()
    anotados = []

    def anotado(entrenamiento, h):
        pronostico = np.asarray(modelo(entrenamiento, h))
        anotados.append((len(entrenamiento), pronostico))
        return pronostico

    inicio = time.perf_counter()
    _, promedio = backtest(serie, anotado)
    segundos = time.perf_counter() - inicio
    parametros = contar_parametros(ULTIMO["modelo"]) if "modelo" in ULTIMO else 0

    errores = pd.concat([p - serie.iloc[corte:corte + len(p)] for corte, p in anotados])
    por_mes = errores.abs().groupby(errores.index.month).mean()
    ERRORES_POR_MES[nombre] = por_mes
    SESGOS[nombre] = errores.mean()
    peor = int(por_mes.idxmax())
    print(f"  {nombre}: RMSE {promedio['RMSE']:.2f} en {segundos:.1f} s, {parametros} parámetros; "
          f"peor mes {MESES[peor - 1]} (error absoluto medio {por_mes.max():.1f}); "
          f"sesgo {SESGOS[nombre]:+.1f}")
    return {"modelo": nombre, "RMSE": round(promedio["RMSE"], 2),
            "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2),
            "parametros": parametros, "segundos": round(segundos, 1)}


def graficar_pronosticos(serie, modelos):
    """Pronóstico desde el último origen del backtest de cada modelo elegido."""
    entrenamiento, real = serie.iloc[:-12], serie.iloc[-12:]
    figura, eje = plt.subplots(figsize=(9, 3.6))
    eje.plot(serie.index[-36:], serie.iloc[-36:].to_numpy(), color="black", label="real")
    for nombre, modelo in modelos.items():
        pronostico = modelo(entrenamiento, 12)
        eje.plot(real.index, pronostico, marker="o", markersize=3,
                 label=f"{nombre} (RMSE {rmse(real, pronostico):.1f})")
        print(f"  desde el último origen, {nombre}: RMSE {rmse(real, pronostico):.1f}")
    eje.set_ylabel(VALOR)
    eje.set_title("Pronóstico a 12 meses desde el último origen")
    eje.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(FIGURAS / "pronosticos.png", dpi=120)
    plt.close(figura)


def graficar_errores_por_mes(orden):
    """Error absoluto medio de cada modelo por mes del año en el backtest."""
    tabla = pd.DataFrame({nombre: ERRORES_POR_MES[nombre] for nombre in orden}).T
    tabla.columns = MESES
    guardada = tabla.copy()
    guardada["sesgo"] = [SESGOS[nombre] for nombre in orden]
    guardada.round(2).to_csv(RESULTADOS / "errores_por_mes.csv")
    figura, eje = plt.subplots(figsize=(10, 0.45 * len(tabla) + 1.6))
    imagen = eje.imshow(tabla.to_numpy(), aspect="auto", cmap="Blues")
    eje.set_xticks(range(12), MESES)
    eje.set_yticks(range(len(tabla)), tabla.index, fontsize=8)
    for i in range(tabla.shape[0]):
        for j in range(12):
            valor = tabla.iat[i, j]
            eje.text(j, i, f"{valor:.0f}", ha="center", va="center", fontsize=7,
                     color="white" if valor > 0.6 * tabla.to_numpy().max() else "black")
    figura.colorbar(imagen, ax=eje, label="error absoluto medio")
    eje.set_title("Error absoluto medio por mes del año en el backtest (ordenado por RMSE)")
    figura.tight_layout()
    figura.savefig(FIGURAS / "errores_por_mes.png", dpi=120)
    plt.close(figura)


def main():
    inicio = time.perf_counter()
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    serie = preparar(cruda)
    graficar_serie(serie)

    print("1. Línea base y modelo a superar")
    filas = [fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]
    elegidos_para_figura = {}
    clasico = modelo_clasico(serie)
    filas.append(fila(NOMBRE_CLASICO, serie, clasico))
    fila_clasico = filas[-1]
    elegidos_para_figura[NOMBRE_CLASICO] = clasico

    print(f"2. Tabla supervisada (OBJETIVO = {OBJETIVO})")
    comprobar_sin_fuga(serie)
    justificar_objetivo(serie)
    X_ult, _ = filas_completas(serie.iloc[:-12])
    print(f"  columnas: {', '.join(X_ult.columns)}")
    print(f"  filas de la tabla: {len(filas_completas(serie.iloc[:-N_PRUEBA])[0])} en el primer "
          f"origen y {len(X_ult)} en el último")

    print("3. Aprendizaje automático")
    elegidos = ajustar_hiperparametros(serie)
    graficar_importancia(serie, elegidos)
    modelos_aa = {nombre: con_tabla(lambda f=fabrica, p=elegidos[nombre]: f(**p))
                  for nombre, (fabrica, _) in REJILLAS.items()}
    filas_aa = [fila(nombre, serie, modelo) for nombre, modelo in modelos_aa.items()]
    filas += filas_aa
    mejor_aa = min(filas_aa, key=lambda f: f["RMSE"])["modelo"]
    elegidos_para_figura[mejor_aa] = modelos_aa[mejor_aa]

    print("4. Aprendizaje profundo")
    epocas = validar_redes(serie)
    filas_red = [fila(f"{nombre}, {e} épocas", serie, con_tabla(red(nombre, e)))
                 for nombre, e in epocas.items()]
    filas += filas_red
    mejor = min(zip(epocas, filas_red), key=lambda par: par[1]["RMSE"])[0]
    elegidos_para_figura[mejor] = con_tabla(red(mejor, epocas[mejor]))
    semillas = repetir_con_semillas(serie, mejor, epocas[mejor])
    diferencia = semillas["RMSE"].mean() - fila_clasico["RMSE"]
    print(f"  la mejor red ({mejor}) queda {diferencia:+.2f} de RMSE frente al clásico: "
          f"{abs(diferencia) / semillas['RMSE'].std():.1f} veces la desviación entre semillas")

    print("5. La tabla")
    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])
    print(f"Datos frente a parámetros en el último origen: {len(serie) - 12} meses, "
          f"{len(X_ult)} filas en la tabla")
    graficar_errores_por_mes(tabla["modelo"])
    graficar_pronosticos(serie, elegidos_para_figura)
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
