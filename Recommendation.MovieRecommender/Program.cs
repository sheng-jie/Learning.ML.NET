using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using ML.Common;

namespace Recommendation.MovieRecommender
{
    class Program
    {
        private static readonly string TrainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "movies-rating-train.csv");
        private static readonly string TestDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "movies-rating-test.csv");
        static void Main(string[] args)
        {
            // 创建上下文
            var mlContext = new MLContext();

            // 创建数据加载器
            var dataLoader = TextLoader.CreateReader(mlContext, ctx => (
                UserId: ctx.LoadFloat(0),
                MovieId: ctx.LoadFloat(1),
                Label: ctx.LoadFloat(2)
            ), hasHeader: true, separator: ',');

            // 加载数据
            var trainingDataView = dataLoader.Read(TrainingDataPath);

            // 选取算法
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("UserId", "userIdEncoded")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("MovieId", "movieIdEncoded")
                    .Append(new MatrixFactorizationTrainer(mlContext, "userIdEncoded", "movieIdEncoded", labelColumn: "Label",
                        advancedSettings: s => { s.NumIterations = 100; s.K = 100; })));
            // 训练
            var model = pipeline.Fit(trainingDataView.AsDynamic);

            // 评估
            var testDataView = dataLoader.Read(TestDataPath);
            var prediction = model.Transform(testDataView.AsDynamic);
            var metrics = mlContext.Regression.Evaluate(prediction);
            ConsoleHelper.PrintRegressionMetrics(model.ToString(), metrics);

            // 测试
            var predictionFunc = model.MakePredictionFunction<MovieRating, MovieRatingPrediction>(mlContext);

            var testMovieRating = new MovieRating()
            {
                UserId = 6,
                MovieId = 10,
            };
            var movieRatingPrediction = predictionFunc.Predict(testMovieRating);

            Movie movieService = new Movie();
            Console.WriteLine("For userId:" + testMovieRating.UserId + " movie rating prediction (1 - 5 stars) for movie:" + movieService.Get(10).MovieTitle + " is:" + Math.Round(movieRatingPrediction.Score, 1));

        }
    }
}
