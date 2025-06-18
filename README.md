![AutoInsight Logo](https://github.com/ProyectoIABD/ProyectoFinalMaster/raw/main/LOGO.png)

# 🚘 AutoInsight

**AutoInsight** es una solución de análisis inteligente que predice el precio de vehículos de segunda mano utilizando modelos de Machine Learning y Deep Learning. El sistema combina datos estructurados, texto y procesamiento de imágenes para generar predicciones precisas del valor del vehículo.

## 🧠 Funcionalidades principales

* 🔍 Scraping de datos desde múltiples plataformas de compraventa de coches (Milanuncios, Flexicar...).
* 🧼 Limpieza y transformación de datos para modelado.
* 🎨 Clasificación del color del coche mediante una CNN entrenada con imágenes (VCoR dataset).
* 📝 Clasificación del estado del vehículo ("nuevo", "neutral", "malo".) a partir de la descripción utilizando NLP.
* 💰 Predicción del precio final del coche con modelos como Random Forest y XGBoost.
* 🌐 Interfaz web para probar el modelo con subida de imagen, texto descriptivo y características del vehículo.

---

## ⚙️ Instalación

### 1. Clona el repositorio

```bash
git clone https://github.com/ProyectoIABD/ProyectoFinalMaster.git
cd ProyectoFinalMaster
```

### 2. Crea el entorno Conda

```bash
conda env create -f tesla_env.yml --no-default-packages
conda activate proyecto_final
```

---

## 📂 Estructura del proyecto

```plaintext
AutoInsight/
│   .gitignore
│   fuentes.txt
│   LOGO.png
│   README.md
│   tesla_env.yml              # Entorno con dependencias necesarias
│
├───data/
│   ├───raw/                   # Datos sin procesar
│   │   ├───descripciones_nlp/     # Descripciones limpias para NLP
│   │   └───VCoR/                  # Dataset de imágenes para detección de color
│   ├───transformed/           # Datos transformados por scraping
│   └───processed/             # Datos listos para modelar
│
├───doc/
│   Final Master.pdf           # Presentacion del proyecto
│
├───src/
│   ├───data_transformation/   # Scripts para limpieza y transformación
│   ├───modeling/
│   │   ├───Notebooks/             # Modelado de precio, color y texto
│   │   └───trained/               # Modelos entrenados (.pkl, .pt)
│   ├───scrapper/              # Scripts y notebooks de scraping
│   └───website/               # Web app (Flask)
│       ├───static/uploads/        # Subidas de imagen
│       └───templates/             # HTML templates (index, resultados)
```

---

## 📚 Créditos

Proyecto desarrollado por [Marcos De Castro Muñoz](https://www.linkedin.com/in/marcos-de-castro-mu%C3%B1oz-5baa84196/) y [Víctor Angulo de Castro](https://www.linkedin.com/in/victor-angulo/) en equipo como parte de un trabajo final para [Master IABD](https://github.com/ProyectoIABD/ProyectoFinalMaster).

---