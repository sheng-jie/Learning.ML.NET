using System;
using Microsoft.ML;
using Microsoft.ML.Runtime.Data;

namespace Classification.IrisClassifier {
    class Program {
        static void Main (string[] args) {
            Console.WriteLine ("Hello ML.NET!");
            //1.创建机器学习上下文
            var mlContext = new MLContext ();

            string dataPath = "iris-data.txt";
            //2.创建机器学习文本读取器
            var reader = mlContext.Data.TextReader (new TextLoader.Arguments () {
                Separator = ",",//指定分隔符
                    HasHeader = false,//文档是否有表头
                    Column = new [] {
                        new TextLoader.Column ("SepalLength", DataKind.R4, 0),
                            new TextLoader.Column ("SepalWidth", DataKind.R4, 1),
                            new TextLoader.Column ("PetalLength", DataKind.R4, 2),
                            new TextLoader.Column ("PetalWidth", DataKind.R4, 3),
                            new TextLoader.Column ("Label", DataKind.Text, 4),
                    }//进行文档列对应
            });

            //3. 将数据读取到dataview
            IDataView trainingDataView = reader.Read (new MultiFileSource (dataPath));

            //4. 创建训练管道，指定训练算法
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey ("Label")
                .Append (mlContext.Transforms.Concatenate ("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append (mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent ())//使用SDCA算法训练线性多类别分类模型
                .Append (mlContext.Transforms.Conversion.MapKeyToValue ("PredictedLabel"));

            //5. 训练
            var model = pipeline.Fit (trainingDataView);

            //6. 预测
            var prediction = model.MakePredictionFunction<IrisData, IrisPrediction> (mlContext)
                .Predict (new IrisData () {
                    SepalLength = 3.3f,
                        SepalWidth = 1.6f,
                        PetalLength = 0.2f,
                        PetalWidth = 5.1f
                });
            
            Console.WriteLine ($"Predicted flower type is :{prediction.PredictedLabels}");
        }
    }
}