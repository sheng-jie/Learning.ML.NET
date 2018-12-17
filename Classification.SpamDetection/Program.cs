using System;
using System.ComponentModel.Composition;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using Microsoft.ML;
using Microsoft.ML.Core;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Classification.SpamDetection
{
    class Program
    {
        static readonly string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "SMSSpamCollection");
        static readonly string DataDictPath = Path.Combine(Environment.CurrentDirectory, "Data", "");

        static void DownloadTrainingData()
        {
            if (!File.Exists(DataPath))
            {
                using (var client = new WebClient())
                {
                    client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip", "spam.zip");
                }

                ZipFile.ExtractToDirectory("spam.zip", DataDictPath);
            }

        }
        static void Main(string[] args)
        {
            DownloadTrainingData();
            // 创建上下文
            MLContext mlContext = new MLContext();
            // 创建文本数据加载器
            TextLoader textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Text, 0),
                    new TextLoader.Column("Message", DataKind.Text, 1)
                }
            });

            // 读取数据集
            var fullData = textLoader.Read(DataPath);
            // 特征工程和指定训练算法
            var estimator = mlContext.Transforms.CustomMapping<MyInput, MyOutput>(MyLambda.MyAction, "MyLambda")

                .Append(mlContext.Transforms.Text.FeaturizeText("Message", "Features"))
                .Append(mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent());
            // 使用交叉验证进行模型评估
            var cvResults = mlContext.BinaryClassification.CrossValidate(fullData, estimator, numFolds: 5);
            var aucs = cvResults.Select(r => r.metrics.Auc);
            Console.WriteLine($"The AUC is {aucs.Average()}");

            // 训练
            var model = estimator.Fit(fullData);


            var inPipe = new TransformerChain<ITransformer>(model.Take(model.Count() - 1).ToArray());
            var lastTransFormer = new BinaryPredictionTransformer<IPredictorProducing<float>>(mlContext,
                model.LastTransformer.Model,
                inPipe.GetOutputSchema(fullData.Schema), model.LastTransformer.FeatureColumn, threshold: 0.15f);
            var parts = model.ToArray();
            parts[parts.Length - 1] = lastTransFormer;
            var newModel = new TransformerChain<ITransformer>(parts);

            var predictor = newModel.MakePredictionFunction<SpamData, SpamPrediction>(mlContext);

            var testMsgs = new string[]
            {
                "That's a great idea. It should work.",
                "free medicine winner! congratulations",
                "Yes we should meet over the weekend!",
                "you win pills and free entry vouchers"
            };

            foreach (var message in testMsgs)
            {
                var input = new SpamData { Message = message };
                var prediction = predictor.Predict(input);

                Console.WriteLine("The message '{0}' is spam? {1}!", input.Message, prediction.IsSpam.ToString());
            }

            Console.WriteLine("Hello World!");
        }

        public class MyInput
        {
            public string Label { get; set; }
        }

        public class MyOutput
        {
            public bool Label { get; set; }
        }

        public class MyLambda
        {
            [Export("MyLambda")]
            public ITransformer MyTransformer => ML.Transforms.CustomMappingTransformer<MyInput, MyOutput>(MyAction, "MyLambda");

            [Import]
            public MLContext ML { get; set; }

            public static void MyAction(MyInput input, MyOutput output)
            {
                output.Label = input.Label == "spam" ? true : false;
            }
        }

    }

}