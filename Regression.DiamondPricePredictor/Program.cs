using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Transforms;

namespace Regression.DiamondPricePredictor
{
    class Program
    {
        private static string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "diamond-price.csv");

        static void Main(string[] args)
        {
            // 创建上下文
            var mlContext = new MLContext();

            // 创建数据加载器
            var reader = TextLoader.CreateReader(mlContext, ctx =>
            (
                Carat: ctx.LoadFloat(0),
                Price: ctx.LoadFloat(1)
            ), hasHeader: true, separator: ',');

            // 读取数据
            var data = reader.Read(DataPath);

            // 创建训练管道
            // 使用SDCA算法
            var pipeLine = reader.MakeNewEstimator()
                .Append(r => (r.Carat,
                    Target: mlContext.Regression.Trainers.Sdca(label: r.Price, features: r.Carat.AsVector())));

            // 训练
            var model = pipeLine.Fit(data).AsDynamic;

            // 预测
            var predictionFunc = model.MakePredictionFunction<DiamondData, DiamondPricePrediction>(mlContext);

            var prediction = predictionFunc.Predict(new DiamondData() {Carat = 1.35f});

            Console.WriteLine($"Predicted price - {string.Format("{0:C}", prediction.PredictedPrice)}");

            Console.Read();
        }
    }
}
