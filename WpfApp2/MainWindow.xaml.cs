using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using MySql.Data.MySqlClient;

namespace WpfApp2
{
    public partial class MainWindow : Window
    {
        private const int InputSize = 6;
        private const int HiddenSize = 20;
        private const int OutputSize = 1;
        private double learningRate = 0.0001;
        private const int Epochs = 80000;

        private double[,] weightsIH;
        private double[,] weightsHO;

        private bool AreWeightsLoaded = false;

        public MainWindow()
        {
            InitializeComponent();
            InitializeWeights();
        }

        private void InitializeWeights()
        {
            Random rand = new Random();
            double stdIH = Math.Sqrt(2.0 / (InputSize + HiddenSize));
            double stdHO = Math.Sqrt(2.0 / (HiddenSize + OutputSize));

            weightsIH = new double[InputSize, HiddenSize];
            weightsHO = new double[HiddenSize, OutputSize];

            for (int i = 0; i < InputSize; i++)
            {
                for (int h = 0; h < HiddenSize; h++)
                {
                    weightsIH[i, h] = rand.NextDouble() * stdIH * 2 - stdIH;
                }
            }

            for (int h = 0; h < HiddenSize; h++)
            {
                for (int o = 0; o < OutputSize; o++)
                {
                    weightsHO[h, o] = rand.NextDouble() * stdHO * 2 - stdHO;
                }
            }
        }

        private async Task<(List<double[]>, List<double>)> ReadDataFromDatabaseAsync()
        {
            string connectionString = "Server=mysql-yohann.alwaysdata.net; database=yohann_dojotrainerbdd; UID=yohann; password=Yoh@bdd;";
            string query = "SELECT `X`, `Y`, `Up`, `Down`, `Right`, `Left`, `Tag` FROM `Donnees`";
            List<double[]> inputsList = new List<double[]>();
            List<double> outputsList = new List<double>();

            using (MySqlConnection connection = new MySqlConnection(connectionString))
            {
                await connection.OpenAsync();
                MySqlCommand command = new MySqlCommand(query, connection);
                using (var reader = await command.ExecuteReaderAsync())
                {
                    while (await reader.ReadAsync())
                    {
                        double[] inputs = {
                            reader.GetDouble("X"),
                            reader.GetDouble("Y"),
                            reader.GetDouble("Up"),
                            reader.GetDouble("Down"),
                            reader.GetDouble("Right"),
                            reader.GetDouble("Left")
                        };
                        double output = reader.GetDouble("Tag");
                        inputsList.Add(inputs);
                        outputsList.Add(output);
                    }
                }
            }

            return (inputsList, outputsList);
        }

        private async Task TrainNetworkAsync(List<double[]> inputsList, List<double> outputsList)
        {
            Random rand = new Random();
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                for (int i = 0; i < inputsList.Count; i++)
                {
                    double[] inputs = inputsList[i];
                    double[] hiddenOutputs = new double[HiddenSize];
                    for (int j = 0; j < HiddenSize; j++)
                    {
                        hiddenOutputs[j] = 0;
                        for (int k = 0; k < InputSize; k++)
                        {
                            hiddenOutputs[j] += inputs[k] * weightsIH[k, j];
                        }
                        hiddenOutputs[j] = Sigmoid(hiddenOutputs[j]);
                    }

                    double output = 0;
                    for (int j = 0; j < HiddenSize; j++)
                    {
                        output += hiddenOutputs[j] * weightsHO[j, 0];
                    }
                    output = Sigmoid(output);

                    double error = outputsList[i] - output;
                    double[] outputDeltas = new double[HiddenSize];
                    for (int j = 0; j < HiddenSize; j++)
                    {
                        outputDeltas[j] = error * SigmoidDerivative(output) * weightsHO[j, 0];
                    }

                    for (int j = 0; j < HiddenSize; j++)
                    {
                        double deltaHO = error * SigmoidDerivative(output) * weightsHO[j, 0];
                        for (int k = 0; k < InputSize; k++)
                        {
                            weightsIH[k, j] += learningRate * deltaHO * SigmoidDerivative(hiddenOutputs[j]) * inputs[k];
                        }
                        weightsHO[j, 0] += learningRate * outputDeltas[j];
                    }
                }
                if (epoch % 1000 == 0)
                {
                    Dispatcher.Invoke(() => trainingProgressBar.Value = (epoch / (double)Epochs) * 100);
                }
            }
        }

        private double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

        private double SigmoidDerivative(double x)
        {
            double sigmoid = Sigmoid(x);
            return sigmoid * (1 - sigmoid);
        }

        private async Task SaveWeightsToDatabaseAsync()
        {
            string connectionString = "Server=mysql-yohann.alwaysdata.net; database=yohann_dojotrainerbdd; UID=yohann; password=Yoh@bdd;";
            using (MySqlConnection connection = new MySqlConnection(connectionString))
            {
                await connection.OpenAsync();

                // Enregistrez les poids de la couche d'entrée vers la couche cachée
                await SaveWeightsAsync(connection, "modelweights_ih", weightsIH, InputSize, HiddenSize);

                // Enregistrez les poids de la couche cachée vers la couche de sortie
                await SaveWeightsAsync(connection, "modelweights_ho", weightsHO, HiddenSize, OutputSize);
            }
        }

        private async Task SaveWeightsAsync(MySqlConnection connection, string tableName, double[,] weights, int rows, int cols)
        {
            // Supprimer les anciennes données
            string deleteQuery = $"DELETE FROM {tableName}";
            await new MySqlCommand(deleteQuery, connection).ExecuteNonQueryAsync();

            // Insérer les nouvelles données
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double weight = double.IsNaN(weights[i, j]) ? 0 : weights[i, j];
                    string weightStr = weight.ToString(System.Globalization.CultureInfo.InvariantCulture);

                    string column1 = tableName == "modelweights_ih" ? "input_index" : "hidden_index";
                    string column2 = tableName == "modelweights_ih" ? "hidden_index" : "output_index";

                    string insertQuery = $@"INSERT INTO {tableName} ({column1}, {column2}, weight)
                                     VALUES ({i}, {j}, {weightStr})";
                    await new MySqlCommand(insertQuery, connection).ExecuteNonQueryAsync();
                }
            }
        }



        private async Task LoadWeightsFromDatabaseAsync()
        {
            string connectionString = "Server=mysql-yohann.alwaysdata.net; database=yohann_dojotrainerbdd; UID=yohann; password=Yoh@bdd;";
            using (MySqlConnection connection = new MySqlConnection(connectionString))
            {
                await connection.OpenAsync();

                MySqlCommand commandIH = new MySqlCommand("SELECT input_index, hidden_index, weight FROM modelweights_ih", connection);
                using (var reader = await commandIH.ExecuteReaderAsync())
                {
                    while (await reader.ReadAsync())
                    {
                        int i = reader.GetInt32("input_index");
                        int j = reader.GetInt32("hidden_index");
                        double weight = reader.GetDouble("weight");
                        weightsIH[i, j] = weight;
                    }
                }

                MySqlCommand commandHO = new MySqlCommand("SELECT hidden_index, output_index, weight FROM modelweights_ho", connection);
                using (var reader = await commandHO.ExecuteReaderAsync())
                {
                    while (await reader.ReadAsync())
                    {
                        int j = reader.GetInt32("hidden_index");
                        int o = reader.GetInt32("output_index");
                        double weight = reader.GetDouble("weight");
                        weightsHO[j, o] = weight;
                    }
                }
            }
            AreWeightsLoaded = true;
        }

        private double[] Predict(double[] input)
        {
            double[] hiddenOutputs = new double[HiddenSize];
            for (int i = 0; i < HiddenSize; i++)
            {
                hiddenOutputs[i] = 0;
                for (int j = 0; j < InputSize; j++)
                {
                    hiddenOutputs[i] += input[j] * weightsIH[j, i];
                }
                hiddenOutputs[i] = Sigmoid(hiddenOutputs[i]);
            }

            double[] finalOutputs = new double[OutputSize];
            for (int o = 0; o < OutputSize; o++)
            {
                finalOutputs[o] = 0;
                for (int j = 0; j < HiddenSize; j++)
                {
                    finalOutputs[o] += hiddenOutputs[j] * weightsHO[j, o];
                }
                finalOutputs[o] = Sigmoid(finalOutputs[o]);
            }

            for (int i = 0; i < finalOutputs.Length; i++)
            {
                finalOutputs[i] = RoundToNearest(finalOutputs[i]);
            }

            return finalOutputs;
        }

        private double RoundToNearest(double value)
        {
            double[] targets = { 0.25, 0.5, 0.75, 1.0 };
            return targets.OrderBy(t => Math.Abs(value - t)).First();
        }

        private async void AutomateTraining()
        {
            bool trainingSucceeded = false;
            while (!trainingSucceeded)
            {
                var (inputsList, outputsList) = await ReadDataFromDatabaseAsync();
                await TrainNetworkAsync(inputsList, outputsList);

                // Sauvegarder les poids dans la base de données après l'entraînement
                await SaveWeightsToDatabaseAsync();

                // Charger les poids après l'entraînement
                await LoadWeightsFromDatabaseAsync();

                bool predictionsMatch = true;

                for (int i = 0; i < inputsList.Count; i++)
                {
                    var input = inputsList[i];
                    var expectedOutput = outputsList[i];
                    var prediction = Predict(input);

                    if (prediction[0] != expectedOutput)
                    {
                        predictionsMatch = false;
                        break;
                    }
                }

                if (!predictionsMatch)
                {
                    await ClearWeightsInDatabaseAsync();
                }
                else
                {
                    trainingSucceeded = true;
                    Dispatcher.Invoke(() => MessageBox.Show("Training succeeded and predictions match the expected outputs!"));
                }
            }
        }

        private async Task ClearWeightsInDatabaseAsync()
        {
            string connectionString = "Server=mysql-yohann.alwaysdata.net; database=yohann_dojotrainerbdd; UID=yohann; password=Yoh@bdd;";
            using (MySqlConnection connection = new MySqlConnection(connectionString))
            {
                await connection.OpenAsync();
                var commands = new List<string> { "DELETE FROM modelweights_ih", "DELETE FROM modelweights_ho" };
                foreach (var cmd in commands)
                {
                    MySqlCommand command = new MySqlCommand(cmd, connection);
                    await command.ExecuteNonQueryAsync();
                }
            }
        }

        private void AutomateTrainingButton_Click(object sender, RoutedEventArgs e)
        {
            AutomateTraining();
        }

        private void Load_Click(object sender, RoutedEventArgs e)
        {
            LoadWeightsFromDatabaseAsync();
        }

        private void Save_Click(object sender, RoutedEventArgs e)
        {
            SaveWeightsToDatabaseAsync();
        }
        private async Task<List<double[]>> ReadPredictionDataFromDatabase()
        {
            string connectionString = "Server=mysql-yohann.alwaysdata.net; database=yohann_dojotrainerbdd; UID=yohann; password=Yoh@bdd;";
            string query = "SELECT `X`, `Y`, `Up`, `Down`, `Right`, `Left` FROM `Donnees`";
            List<double[]> dataWithPredictions = new List<double[]>();

            using (MySqlConnection connection = new MySqlConnection(connectionString))
            {
                await connection.OpenAsync();
                MySqlCommand command = new MySqlCommand(query, connection);

                using (var reader = await command.ExecuteReaderAsync())
                {
                    while (await reader.ReadAsync())
                    {
                        double[] inputs = {
                    reader.GetDouble(0), // X
                    reader.GetDouble(1), // Y
                    reader.GetDouble(2), // Up
                    reader.GetDouble(3), // Down
                    reader.GetDouble(4), // Right
                    reader.GetDouble(5)  // Left
                };
                        dataWithPredictions.Add(inputs);
                    }
                }
            }
            return dataWithPredictions;
        }
        private async void Predict_Click(object sender, RoutedEventArgs e)
        {
            if (!AreWeightsLoaded)
            {
                await LoadWeightsFromDatabaseAsync(); // Attendez que les poids soient chargés
            }

            var predictionData = await ReadPredictionDataFromDatabase(); // Attendez que les données de prédiction soient lues
            trainingDataGrid.Items.Clear(); // Effacez les données existantes dans le DataGrid

            foreach (var data in predictionData)
            {
                var input = data;
                var prediction = Predict(input); // Faites la prédiction

                if (prediction != null && prediction.Length > 0)
                {
                    // Créez un objet anonyme pour inclure à la fois les données d'entrée et la prédiction brute
                    var displayData = new
                    {
                        X = input[0],
                        Y = input[1],
                        Up = input[2],
                        Down = input[3],
                        Right = input[4],
                        Left = input[5],
                        Prediction = prediction[0] // Utilisez la première sortie de la prédiction brute
                    };

                    trainingDataGrid.Items.Add(displayData); // Ajoutez les données au DataGrid pour affichage
                }
            }
        }
    }
}
