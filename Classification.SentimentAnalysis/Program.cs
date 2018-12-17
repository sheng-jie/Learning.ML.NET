using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using ML.Common;

namespace Classification.SentimentAnalysis
{
    class Program
    {
        static readonly string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-all.tsv");
        static readonly string ModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "SentimentClassifyingModel.zip");

        static void Main(string[] args)
        {
            // 创建上下文
            var mlContext = new MLContext();
            // 创建文本加载器
            var textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = true,
                Column = new TextLoader.Column[]
                {
                    new TextLoader.Column("Label",DataKind.Bool,0), 
                    new TextLoader.Column("Text",DataKind.Text,1)
                }
            });
            // 读取数据集
            var fullDataView = textLoader.Read(DataPath);
            // 数据集一拆为二
            (IDataView trainingData, IDataView testingData) = mlContext.Clustering.TrainTestSplit(fullDataView);
            // 特征工程：数值化
            var dataProcessPipeline= mlContext.Transforms.Text.FeaturizeText("Text", "Features");
            ConsoleHelper.PeekDataViewInConsole<SentimentData>(mlContext, trainingData, dataProcessPipeline, 2);
            ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "Features", testingData, dataProcessPipeline, 1);
            // 使用决策树算法
            var trainer = mlContext.BinaryClassification.Trainers.FastTree();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // 训练
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);
            // 测试评估
            var predictions = trainedModel.Transform(testingData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions);

            ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);

            //保存模型
            using (var fs = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainedModel, fs);

            Console.WriteLine("The model is saved to {0}", ModelPath);
            
        }
    }
}
