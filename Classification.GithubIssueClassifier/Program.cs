using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using ML.Common;

namespace Classification.GithubIssueClassifier
{
    class Program
    {
        static readonly string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "corefx-issues-train.tsv");
        static readonly string ModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "GithubIssueClassifier.zip");

        static void Main(string[] args)
        {
            TrainAndTestModel();
        }

        private static void TrainAndTestModel()
        {
            var mlContext = new MLContext(seed: 0);

            // 创建文本加载器用于数据集读取
            TextLoader textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("ID", DataKind.Text, 0),
                    new TextLoader.Column("Area", DataKind.Text, 1),
                    new TextLoader.Column("Title", DataKind.Text, 2),
                    new TextLoader.Column("Description", DataKind.Text, 3)
                }
            });
            // 读取数据集
            var trainingDataView = textLoader.Read(DataPath);

            var dataProcessPipeLine = mlContext.Transforms.Conversion.MapValueToKey("Area", "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText("Title", "TitleFeaturized")) //特征向量
                .Append(mlContext.Transforms.Text.FeaturizeText("Description", "DescriptionFeaturized")) //特征向量
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(mlContext); //添加缓存检查点，避免后续评估模型时再次读取造成性能浪费

            // 挑选算法
            var trainer = mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent();
            //var averagePerceptronBinaryTranier = mlContext.BinaryClassification.Trainers.AveragedPerceptron(numIterations:10);
            //var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(averagePerceptronBinaryTranier);

            var trainingPipeline = dataProcessPipeLine.Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            // 使用交叉验证（交叉验证是在机器学习建立模型和验证模型参数时常用的办法）
            var crossValidationResults =
                mlContext.MulticlassClassification.CrossValidate(trainingDataView, trainingPipeline, numFolds: 6);
            ConsoleHelper.PrintMulticlassClassificationFoldsAverageMetrics(trainer.ToString(), crossValidationResults);
            //训练模型
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // 预测数据
            GithubIssue issue = new GithubIssue()
            {
                ID = "485241",
                Title = "WebSockets communication is slow in my machine",
                Description =
                    "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };
            // 基于训练模型创建预测引擎
            var predFunction = trainedModel.MakePredictionFunction<GithubIssue, GithubIssuePrediction>(mlContext);
            // 预测
            var prediction = predFunction.Predict(issue);
            Console.WriteLine(
                $"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

            // 保存模型
            Console.WriteLine("=============== Saving the model to a file ===============");
            using (var fs = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainedModel, fs);
        }
    }
}