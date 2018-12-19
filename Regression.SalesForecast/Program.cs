using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Transforms;
using ML.Common;

namespace Regression.SalesForecast
{
    class Program
    {
        private static string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "products.stats.csv");
        static void Main(string[] args)
        {
            //创建上下文
            MLContext mlContext = new MLContext();

            //创建数据加载器
            var textLoader = mlContext.Data.TextReader(new TextLoader.Arguments
            {
                Column = new[] {
                    new TextLoader.Column("Next", DataKind.R4, 0 ),
                    new TextLoader.Column("ProductId", DataKind.Text, 1 ),
                    new TextLoader.Column("Year", DataKind.R4, 2 ),
                    new TextLoader.Column("Month", DataKind.R4, 3 ),
                    new TextLoader.Column("Units", DataKind.R4, 4 ),
                    new TextLoader.Column("Avg", DataKind.R4, 5 ),
                    new TextLoader.Column("Count", DataKind.R4, 6 ),
                    new TextLoader.Column("Max", DataKind.R4, 7 ),
                    new TextLoader.Column("Min", DataKind.R4, 8 ),
                    new TextLoader.Column("Prev", DataKind.R4, 9 )
                },
                HasHeader = true,
                Separator = ","
            });

            var trainingDataView = textLoader.Read(DataPath);

            var trainer = mlContext.Regression.Trainers.FastTreeTweedie();

            var trainingPipeline = 
                mlContext.Transforms.Concatenate("NumFeatures", "Year", "Month", "Units", "Avg",
                    "Count", "Max", "Min", "Prev")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(inputColumn: "ProductId",
                    outputColumn: "CatFeatures"))
                .Append(mlContext.Transforms.Concatenate(outputColumn: "Features",
                    inputColumns: new[] { "NumFeatures", "CatFeatures" }))
                .Append(mlContext.Transforms.CopyColumns("Next", "Label"))
                .Append(trainer);

            var crossValidationResults = mlContext.Regression.CrossValidate(trainingDataView, trainingPipeline, 6);
            ConsoleHelper.PrintRegressionFoldsAverageMetrics(trainer.ToString(), crossValidationResults);

            var model = trainingPipeline.Fit(trainingDataView);

            var predictionFunc = model.MakePredictionFunction<ProductData, ProductUnitPrediction>(mlContext);

            ProductData dataSample = new ProductData()
            {
                ProductId = "263",
                Month = 10,
                Year = 2017,
                Avg = 91,
                Max = 370,
                Min = 1,
                Count = 10,
                Prev = 1675,
                Units = 910
            };

            ProductUnitPrediction predictionResult = predictionFunc.Predict(dataSample);
            Console.WriteLine($"Product: {dataSample.ProductId}, month: {dataSample.Month + 1}, year: {dataSample.Year} - Real value (units): 551, Forecast Prediction (units): {predictionResult.Score}");

            dataSample = new ProductData()
            {
                ProductId = "988",
                Month = 11,
                Year = 2017,
                Avg = 41,
                Max = 225,
                Min = 4,
                Count = 26,
                Prev = 1094,
                Units = 1076
            };

            predictionResult = predictionFunc.Predict(dataSample);
            Console.WriteLine($"Product: {dataSample.ProductId}, month: {dataSample.Month + 1}, year: {dataSample.Year} - Forecasting (units): {predictionResult.Score}");

        }
    }
}
