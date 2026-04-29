# GNN-Predict-Delay

This repository implements a solution that uses an optimizer to find the best GNN architecture and parameters for predicting delay in RNP link pairs.

## Project Structure

- `test_gnn/`: Contains scripts for training and testing the GNN model and predicting delays.
- `test_gnn/test_gnn/dataset_organization/`: Contains scripts for organizing datasets for prediction tasks and training.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/janaina-ribeiro/GNN-Predict-Delay.git
    ```
2. Navigate to the project directory:
    ```bash
    cd GNN-Predict-Delay
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

4. Prepare the delay-traceroute dataset (note: update the paths inside the script as needed):
    ```bash
    python test_gnn/test_gnn/dataset_organization/dataset_organization_train.py
    ```
5. Train the model to find the best hyperparameters using `run_optimization.py` (note: update the paths inside the script as needed):
    ```bash
    # Using full mode 
    python test_gnn/run_optimization.py --mode full

    # Using quick mode
    python test_gnn/run_optimization.py --mode quick
    ```

6. After training, use the best hyperparameters to train the final model (update the parameter values if you find better hyperparameters):
    ```bash
    python test_gnn/train_params_best.py
    ```

## Predicting

7. Follow these steps to predict delay using the trained model:

- First, create the delay-traceroute dataset (note: update the paths inside the script as needed):
    ```bash   
    python test_gnn/test_gnn/dataset_organization/dataset_organization_predict.py
    ```
- Then, run the script to create the dataset graph (note: update the paths inside the script as needed):
    ```bash
    python test_gnn/test_gnn/dataset_organization/build_prediction_dataset_graph.py
    ```
- Finally, run the prediction script (note: update the paths inside the script as needed):
    ```bash
    python test_gnn/predict_delay.py
    ```
The predicted delays will be saved in the specified output directory. The output CSV files from the prediction include the runtime in CSV and JSON formats, as well as a file comparing the actual target with the target predicted by the model.

## IEEE Dataport Dataset
This section provides access to a specialized dataset containing historical traceroute information and delay measurements collected from RNP (Brazilian National Research and Education Network). The data collected for the experiment is from 2023:
https://ieee-dataport.org/documents/datasets-delay-traceroute-2023

## Contact

For any questions or suggestions:
- Janaina Ribeiro - janainaribeiro780@gmail.com / janaina.ribeiro@aluno.uece.br
