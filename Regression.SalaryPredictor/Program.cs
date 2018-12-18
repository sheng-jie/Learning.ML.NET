using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.Transforms;

namespace Regression.SalaryPredictor
{
    class Program
    {
        private static string DataPath = Path.Combine(Environment.CurrentDirectory, "Data", "SalaryData.csv");
        static void Main(string[] args)
        {
            // 创建上下文
            var mlContext = new MLContext();
            var reader = TextLoader.CreateReader(mlContext, ctx =>
            (
                YearsExperience: ctx.LoadFloat(0),
                Salary: ctx.LoadFloat(1)
            ), hasHeader: true, separator: ',');

            // 读取数据
            var data = reader.Read(DataPath);

            // 构建训练管道
            // 使用SDCA算法
            var pipeline = reader.MakeNewEstimator()
                .Append(r => (r.Salary,
                        Target: mlContext.Regression.Trainers.Sdca(label: r.Salary,
                            features: r.YearsExperience.AsVector())
                    ));

            // 训练
            var model = pipeline.Fit(data).AsDynamic;

            // 预测
            var predictionFunc = model.MakePredictionFunction<SalaryData, SalaryPrediction>(mlContext);

            var prediction = predictionFunc.Predict(new SalaryData() { YearsExperience = 8 });

            Console.WriteLine($"Predicted salary - {string.Format("{0:C}", prediction.PredictedSalary)}");

            Console.Read();


        }
    }
}
