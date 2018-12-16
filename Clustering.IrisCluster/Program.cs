using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;

namespace Clustering.IrisCluster
{
    /// <summary>
    /// 山鸢尾、变色鸢尾和维吉尼亚鸢尾
    /// </summary>
    class Program
    {
        static readonly string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris-full.txt");
        static readonly string ModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

        private static void Main(string[] args)
        {
            //创建上下文
            MLContext mlContext = new MLContext(seed: 1);

            //创建文本数据加载器
            TextLoader textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = "\t",
                HasHeader = true,
                Column = new[] {
                        new TextLoader.Column ("Label", DataKind.R4, 0),
                            new TextLoader.Column ("SepalLength", DataKind.R4, 1),
                            new TextLoader.Column ("SepalWidth", DataKind.R4, 2),
                            new TextLoader.Column ("PetalLength", DataKind.R4, 3),
                            new TextLoader.Column ("PetalWidth", DataKind.R4, 4)
                    }
            });
            //加载数据集
            IDataView fullData = textLoader.Read(DataPath);

            //数据集划分
            (IDataView trainingDataView, IDataView testingDataView) =
            mlContext.Clustering.TrainTestSplit(fullData, testFraction: 0.2);

            // 创建数据处理管道
            var features = new string[] {
                "SepalLength",
                "SepalWidth",
                "PetalLength",
                "PetalWidth"
            };
            //特征工程:特征列绑定
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", features);

            //挑选算法
            var trainer = mlContext.Clustering.Trainers.KMeans(features: "Features", clustersCount: 3);
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            //评估模型
            IDataView predictions = trainedModel.Transform(testingDataView);

            var evaluateResult = mlContext.Clustering.Evaluate(predictions, score: "Score", features: "Features");
            PrintClusteringMetrics(evaluateResult.ToString(), evaluateResult);

            using (var fs = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainedModel, fs);


            Predict();
        }

        private static void Predict()
        {
            var mlContext = new MLContext();
            var setosa = new IrisData()
            {
                SepalLength = 5.1f,
                SepalWidth = 3.5f,
                PetalLength = 1.4f,
                PetalWidth = 0.2f
            };

            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                //加载模型
                var model = mlContext.Model.Load(stream);
                var predictionFunc = model.MakePredictionFunction<IrisData, IrisPrediction>(mlContext);

                var testResult = predictionFunc.Predict(setosa);

                Console.WriteLine($"Cluster assigned for setosa flower:{testResult.PredictedClusterId}");
                Console.ReadKey();


            }
        }

        public static void PrintClusteringMetrics(string name, ClusteringEvaluator.Result metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name} clustering model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       AvgMinScore: {metrics.AvgMinScore}");
            Console.WriteLine($"*       DBI is: {metrics.Dbi}");
            Console.WriteLine($"*************************************************");
        }


    }
}