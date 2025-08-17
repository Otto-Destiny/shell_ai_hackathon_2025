# 🦅 Eagle Blend Optimizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg)](https://streamlit.io)
[![Made with ❤️ by eagle-team](https://img.shields.io/badge/Made%20with-❤️-red)](https://github.com/eagle-team)
[![Deploy on Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/eagle-team/eagle-blend-optimizer)

---


**An AI-Powered Web Application for Fuel Blend Property Prediction and Optimization**
*A project by eagle-team for the Shell.ai 2025 Hackathon*

## 📜 Project Overview

The **Eagle Blend Optimizer** is a sophisticated, data-driven tool designed to revolutionize the fuel blending process. By leveraging a powerful machine learning model, this application provides engineers and researchers with the ability to accurately predict the final properties of a fuel blend based on its constituent components. More than just a predictive tool, it features an advanced optimization engine that can determine the ideal component recipes required to meet specific, user-defined performance and cost targets.

This platform streamlines the R&D workflow, reduces reliance on costly and time-consuming physical lab tests, and empowers users to make smarter, faster, and more economical decisions in fuel formulation.



---

## ✨ Key Features

The application is organized into a clean, intuitive tab-based interface, with each section dedicated to a specific task in the blend management lifecycle.

### 📊 Dashboard
The central landing page provides a high-level, at-a-glance summary of the entire application's status.
-   **Performance KPIs:** Real-time metrics on model accuracy (R²), total predictions and optimizations performed, and the highest cost savings achieved.
-   **Activity Chart:** A visual representation of blend creation activity over time, helping to identify usage patterns.
-   **Recent Blends Feed:** A live feed showing the most recently created blends and their key properties.

### 🎛️ Blend Designer
A powerful sandbox for creating and analyzing new blend recipes.
-   **Manual Mode:** Interactively define a blend using up to five components. Users can either enter component properties manually or load them directly from the integrated **Fuel Registry**. The app provides instant predictions for 10 distinct blend properties and calculates the estimated cost.
-   **Batch Mode:** For high-throughput analysis, users can upload a CSV file containing hundreds or thousands of blend recipes and receive a complete prediction for all of them, which can be downloaded as a new CSV.

### ⚙️ Optimization Engine
This is the core intelligent feature of the application. Instead of just predicting outcomes, the engine works backward to find optimal solutions.
-   **Target-Driven Optimization:** Users specify their desired target values for any of the 10 blend properties.
-   **Constraints:** Targets can be set as "fixed" (must be met exactly) or "flexible" (goals to be aimed for). Cost can also be included as an optimization constraint.
-   **Multi-Solution Output:** The optimizer provides multiple optimal solutions (e.g., the cheapest blend, the highest quality blend), allowing users to analyze trade-offs via a Pareto front visualization.

### 📤 Blend Comparison
An analytical workspace for performing detailed side-by-side comparisons of up to three saved blends.
-   **Visual Analysis:** Generates a suite of comparative charts, including lollipop plots for cost, radar charts for property profiles, a cost-vs-quality scatter plot, and stacked bar charts for component composition.
-   **PDF Reporting:** All comparison data and visualizations can be exported as a clean, professional PDF report with a single click.

### 📚 Fuel Registry
The central database powering the entire application.
-   **Data Management:** Provides a complete view of all saved fuel **Components** and final **Blends**.
-   **CRUD Functionality:** Users can add new components (manually or via CSV), view all records, search for specific entries, and delete obsolete data.

### 🧠 Model Insights
A transparent look "under the hood" of the AI model.
-   **Performance Metrics:** Displays key model health indicators like overall R², Mean Squared Error (MSE), and Mean Absolute Percentage Error (MAPE).
-   **Per-Property Accuracy:** A detailed bar chart shows the model's R² score for each of the 10 properties, giving users confidence in the predictions by highlighting which properties the model predicts most accurately.

---

## 🛠️ Technical Stack

-   **Frontend:** Streamlit
-   **Data Manipulation:** Pandas & NumPy
-   **Visualization:** Plotly Express & Plotly Graph Objects
-   **Database:** SQLite (for local data persistence)
-   **Machine Learning Model:** The backend prediction is powered by a pre-trained model (e.g., TabPFN, XGBoost) loaded via the `predictor.py` script.
-   **Deployment:** Hosted on Hugging Face Spaces.

---

## 🚀 Getting Started

1.  **Explore the Registry:** Navigate to the **Fuel Registry** tab to see the available base components you can work with. You can add your own via the forms or by uploading a CSV.
2.  **Make a Prediction:** Go to the **Blend Designer**, load a few components from the registry, define their fractions (ensuring they sum to 1.0), and click "Predict".
3.  **Find an Optimal Blend:** Move to the **Optimization Engine**, set a few target properties, select your base components, and run the optimizer to discover new, high-performing recipes.
4.  **Compare Results:** Save your best results from the Designer and Optimizer, then head to the **Blend Comparison** tab to analyze them side-by-side.

---

## 👥 About the Team

This application was designed and built by **eagle-team**:
-   Destiny Otto
-   Williams Alabi
-   Godswill Otto
-   Alexander Ifenaike