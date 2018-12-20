using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Transforms;

namespace Classification.Tensorflow.ImageClassifier
{
    class Program
    {
        private static string DataPath = Path.Combine(Environment.CurrentDirectory, "Data");
        private static string ImageFolder = Path.Combine(Environment.CurrentDirectory, "Data");
        private static string TFModelPath = Path.Combine(Environment.CurrentDirectory, "Data");
        private static string TestDataPath = Path.Combine(Environment.CurrentDirectory, "Data");
        
        static void Main(string[] args)
        {
            // 创建上下文
            var mlContext = new MLContext();
            
            var loader = new TextLoader(mlContext, new TextLoader.Arguments()
            {
                Column = new[]
                {
                    new TextLoader.Column("ImagePath", DataKind.Text, 0),
                },
                HasHeader = false
            });

            var data = loader.Read(new MultiFileSource(DataPath));

            var pipeline =
                mlContext.Transforms.LoadImages(imageFolder: ImageFolder, columns: ("ImagePath", "ImageReal"))
                    .Append(mlContext.Transforms.Resize("ImageReal", "ImageReal", ImageNetSettings.ImageWidth,
                        ImageNetSettings.ImageHeight))
                    .Append(mlContext.Transforms.ExtractPixels(new[]
                    {
                        new ImagePixelExtractorTransform.ColumnInfo("ImageReal", "input",
                            interleave: ImageNetSettings.ChannelsLast, offset: ImageNetSettings.Mean),
                    }))
                    .Append(mlContext.Transforms.ScoreTensorFlowModel(TFModelPath, new[] {"input"},
                        new[] {"softmax2"}));
            var model = pipeline.Fit(data);

            var predictionFunc = model.MakePredictionFunction<ImageNetData, ImageNetPrediction>(mlContext);

            var testData = ImageNetData.ReadFromCsv(TestDataPath, ImageFolder);

            foreach (var sample in testData)
            {
                var probs = predictionFunc.Predict(sample).PredictedLabels;
                var imageData = new ImageNetDataProbability()
                {
                    ImagePath = sample.ImagePath,
                    Label = sample.Label
                };
            }

            Console.WriteLine("Hello World!");
        }
    }
}
