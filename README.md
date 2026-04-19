# 🚢 Titanic - Machine Learning from Disaster

Solución a la [competición del Titanic de Kaggle](https://www.kaggle.com/competitions/titanic): predicción de supervivencia de los pasajeros usando machine learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Tabla de contenidos

- [Descripción del problema](#descripción-del-problema)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Resultados](#resultados)
- [Metodología](#metodología)
- [Instalación y uso](#instalación-y-uso)
- [Conclusiones](#conclusiones)
- [Nota sobre el leaderboard](#nota-sobre-el-leaderboard)

---

## 🎯 Descripción del problema

El RMS Titanic se hundió el 15 de abril de 1912 tras colisionar con un iceberg, dejando 1502 víctimas de los 2224 pasajeros y tripulantes. El reto consiste en construir un modelo que prediga qué pasajeros sobrevivieron basándose en sus datos (edad, sexo, clase, tarifa, etc.).

- **Tipo de problema:** clasificación binaria
- **Métrica:** accuracy
- **Datos:** 891 pasajeros en train, 418 en test

---

## 📂 Estructura del proyecto

```
titanic-kaggle/
├── data/                    # Datasets originales de Kaggle
├── notebooks/               # Notebook con el análisis completo
├── submissions/             # Archivos CSV para subir a Kaggle
├── images/                  # Gráficos generados
├── requirements.txt         # Dependencias del proyecto
└── README.md
```

---

## 📊 Resultados

| Versión | Modelo | CV Accuracy | Kaggle Public |
|---------|--------|-------------|---------------|
| v0 | Baseline (todas mujeres=1) | 0.7868 | 0.7656 |
| v1 | Gradient Boosting (default) | 0.8417 | ~0.77 |
| v2 | CatBoost | 0.8552 | ~0.78 |
| v3 | CatBoost + Optuna | ~0.86 | ~0.79 |
| **v4** | **Ensemble (Voting Soft)** | **~0.86** | **~0.80** |

**Mejora total sobre el baseline: +3-4 puntos porcentuales en el leaderboard público.**

---

## 🔬 Metodología

El proyecto sigue el pipeline estándar de un proyecto de ML:

### 1. Análisis exploratorio (EDA)

Hallazgos clave:
- **Sexo:** las mujeres sobrevivieron al 74%, los hombres solo al 19%.
- **Clase:** supervivencia del 63% en 1ª, 47% en 2ª y 24% en 3ª.
- **Tamaño de familia:** forma de "U invertida" — familias de 2-4 miembros sobrevivieron más (55-72%) que los pasajeros solos (30%) o las familias grandes (16%).
- **Cabina:** tener cabina registrada correlaciona con 67% de supervivencia vs 30% sin cabina.

### 2. Feature Engineering

Se crearon 10 variables nuevas a partir de las originales:

| Feature | Descripción |
|---------|-------------|
| `Title` | Título extraído del nombre (Mr, Mrs, Miss, Master, Rare) |
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | 1 si viaja solo, 0 si no |
| `FamilyGroup` | Bins categóricos (Solo, Small, Large) |
| `HasCabin` | 1 si tiene cabina registrada |
| `Deck` | Primera letra de la cabina (cubierta del barco) |
| `AgeBin` | Categorías de edad (Child, Teen, Young, Adult, Senior) |
| `FareBin` | Cuartiles de la tarifa |
| `TicketGroupSize` | Número de pasajeros con el mismo ticket |
| `FarePerPerson` | Tarifa dividida entre el tamaño del grupo del ticket |
| `IsChild` | 1 si Age < 16 |

**Estrategia de imputación:** la edad se imputó con la mediana agrupada por título (Master → 4 años, Mr → 29 años, Mrs → 35 años), capturando sexo + edad + estatus a la vez.

### 3. Modelado

Se probaron múltiples modelos de forma incremental:

- **Baseline:** Logistic Regression, Decision Tree
- **Avanzados:** Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Tuning:** Optuna para optimización bayesiana de hiperparámetros
- **Ensembling:** Voting Classifier (soft voting) con 5 modelos diversos

### 4. Validación

Se usó **Stratified K-Fold Cross-Validation (5 folds)** para estimar el rendimiento sin overfitting al train set. Todas las transformaciones se encapsulan en `Pipelines` de scikit-learn para evitar data leakage.

---

## 💻 Instalación y uso

### Requisitos
- Python 3.10+
- Las dependencias listadas en `requirements.txt`

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/TU-USUARIO/titanic-kaggle.git
cd titanic-kaggle

# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución

```bash
# Lanzar Jupyter
jupyter notebook notebooks/titanic_analysis.ipynb
```

El notebook se puede ejecutar completo con `Kernel → Restart & Run All`.

---

## 🎓 Conclusiones

1. **El feature engineering importa más que el modelo.** Pasar de 0.787 (baseline) a 0.855 se consiguió sobre todo con las variables creadas, no con la sofisticación del algoritmo.
2. **Imputar con criterio mejora mucho.** Imputar `Age` por título (no por media global) fue una de las decisiones de más impacto.
3. **CatBoost fue el mejor modelo individual** para este dataset, probablemente por su manejo nativo de variables categóricas y su regularización robusta.
4. **El ensembling da ganancias marginales** pero consistentes. Voting soft superó al mejor modelo individual.
5. **El tuning tiene rendimientos decrecientes.** Optuna encontró hiperparámetros ~1 punto mejores, pero el esfuerzo de horas de cómputo solo compensa cerca del techo del leaderboard.

---

## ⚠️ Nota sobre el leaderboard

El dataset del Titanic es **público** y los datos reales de supervivencia están documentados históricamente. Es técnicamente posible obtener 1.000 de accuracy buscando los pasajeros en bases de datos externas, pero eso no representa un logro de machine learning.

**El objetivo de este proyecto es demostrar un pipeline de ML riguroso y honesto**, no maximizar la puntuación en el leaderboard. Un score de 0.80 obtenido con una metodología sólida vale más que un 1.00 con trampa.

---

## 📚 Referencias

- [Competición Titanic en Kaggle](https://www.kaggle.com/competitions/titanic)
- [Documentación de scikit-learn](https://scikit-learn.org/)
- [Documentación de CatBoost](https://catboost.ai/)
- [Documentación de Optuna](https://optuna.org/)

---

## 👤 Autor

**Tu Nombre** - [@TU-USUARIO](https://github.com/TU-USUARIO) - tu-email@ejemplo.com

Hecho con ❤️ como proyecto de aprendizaje en machine learning.
