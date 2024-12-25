Name: Anushka Pachaury

---

Project Overview
This project is focused on implementing and evaluating various temporal models, including Temporal Kolmogorov-Arnold Networks (TKAN) and efficient Kolmogorov-Arnold Networks (KAN) with different activation functions such as Jacobi, Chebyshev, B-Spline, and Bessel. The main objectives include:

1. Preprocessing S&P500 stock price data, specifically focusing on the `Adj Close` feature.
2. Running temporal models (TKAN) with Jacobi, Chebyshev, and B-Spline activations.
3. Running standard efficient KANs with various activation functions.
4. Plotting results and analyzing model performance.

The project utilizes a combination of custom implementations and open-source libraries for these tasks.

---

Code Structure
The project is organized into the following files and directories:

1. `tkan_Jacobi/` and `tkan_Chebyshev/`
   - These directories contain code that implements and supports the temporal models.
   - Purpose:
     - `tkan_Jacobi` focuses on implementing the Jacobi temporal model.
     - `tkan_Chebyshev` focuses on implementing the Chebyshev temporal model.
   - How to Install:
     - Navigate to each directory and run:
       pip install .

2. `efficient_kanJacobi.py` and `efficient_kanChebyshev.py`
   - Contain code for adding a recurrent layer to the standard efficient KAN, creating the Temporal KAN (tKAN) models.
   - Supports the Jacobi (`efficient_kanJacobi.py`) and Chebyshev (`efficient_kanChebyshev.py`) temporal activations.

3. `tkan.py`
   - Contains the main implementation of the tKAN model, integrating recurrent layers with KAN.

4. `TKAN.ipynb`
   - Jupyter Notebook for running all temporal models and generating performance plots for Jacobi, Chebyshev, and B-Spline activations.
   - Also includes preprocessing of S&P500 stock prices.

5. `KANModel.ipynb`
   - Jupyter Notebook for running standard efficient KANs with activation functions such as Jacobi, Chebyshev, B-Spline, and Bessel.
   - Includes preprocessing of the `Adj Close` feature of S&P500 stock prices.

6. Raw Data
   - The raw S&P500 stock price data (`Adj Close`) is included in a zip file for convenience.
   - Data source: Downloaded using the Yahoo Finance API.

---

How to Run
1. Set Up Python Virtual Environment
   - It is recommended to create a Python virtual environment to manage dependencies for this project:
     python3 -m venv tkan_env
     source tkan_env/bin/activate    # On macOS/Linux
     tkan_env\Scripts\activate       # On Windows

2. Install Dependencies
   - Navigate to the `tkan_Jacobi/` and `tkan_Chebyshev/` directories and install dependencies:
     cd tkan_Jacobi
     pip install .
     cd ../tkan_Chebyshev
     pip install .

3. Run Temporal Models
   - Open `TKAN.ipynb` in Jupyter Notebook or an equivalent environment.
   - Run the notebook to preprocess the S&P500 stock price data, train the temporal models (Jacobi, Chebyshev, B-Spline), and generate performance plots.

4. Run Standard Efficient KANs
   - Open `KANModel.ipynb` in Jupyter Notebook or an equivalent environment.
   - Run the notebook to preprocess the data and train standard efficient KANs with various activation functions (Jacobi, Chebyshev, B-Spline, Bessel).

5. Data Preprocessing
   - The preprocessing steps in both notebooks include:
     - Downloading stock price data using Yahoo Finance (`yfinance` library).
     - Focusing on the `Adj Close` column.
     - Normalizing the data using `MinMaxScaler`.

---

References
This project builds upon the following open-source projects:

1. Efficient-KAN by Blealtan
   - Provides the base implementation for Kolmogorov-Arnold Networks (KAN).
   - Link: https://github.com/Blealtan/efficient-kan

2. TKAN by Remigenet
   - Implements Temporal Kolmogorov-Arnold Networks (TKAN) with various activation functions.
   - Link: https://github.com/remigenet/TKAN

3. Yahoo Finance API
   - Data for S&P500 stock prices was obtained using the `yfinance` Python library.
   - Link: https://pypi.org/project/yfinance/

---

Output and Results
1. Temporal Models (`TKAN.ipynb`)
   - Produces performance plots for Jacobi, Chebyshev, and B-Spline temporal models.
   - Includes metrics such as loss and accuracy for each model.

2. Efficient KANs (`KANModel.ipynb`)
   - Generates results for standard KANs with Jacobi, Chebyshev, B-Spline, and Bessel activations.
   - Outputs include activation-specific plots and model performance metrics.
