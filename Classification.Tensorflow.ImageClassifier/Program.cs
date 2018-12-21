using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Trainers.FastTree.Internal;
using Microsoft.ML.Transforms;

namespace Classification.Tensorflow.ImageClassifier
{
    class Program
    {
        private static string RootPath = Path.GetFullPath(@"..\..\..\");
        private static string DataPath = Path.Combine(RootPath, "Data", "Images", "tags.tsv");
        private static string ImageFolder = Path.Combine(RootPath, "Data", "Images");
        private static string TFModelPath = Path.Combine(RootPath, "Data", "TFModel", "tensorflow_inception_graph.pb");
        private static string TFModelLabelPath = Path.Combine(RootPath, "Data", "TFModel",
            "imagenet_comp_graph_label_strings.txt");

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
                    .Append(mlContext.Transforms.ScoreTensorFlowModel(TFModelPath, new[] { "input" },
                        new[] { "softmax2" }));
            var model = pipeline.Fit(data);

            var predictionFunc = model.MakePredictionFunction<ImageNetData, ImageNetPrediction>(mlContext);

            var testData = ImageNetData.ReadFromCsv(DataPath, ImageFolder);
            var labels = File.ReadAllLines(TFModelLabelPath);

            foreach (var sample in testData)
            {
                var probs = predictionFunc.Predict(sample).PredictedLabels;
                var imageData = new ImageNetDataProbability()
                {
                    ImagePath = sample.ImagePath,
                    Label = sample.Label
                };

                (imageData.PredicateLabel, imageData.Probability) = GetBestLabel(labels, probs);
                PrintReuslt(imageData);
            }
        }

        public static (string, float) GetBestLabel(string[] labels, float[] probs)
        {
            var max = probs.Max();
            var index = probs.AsSpan().IndexOf(max);
            return (labels[index], max);
        }

        public static void PrintReuslt(ImageNetDataProbability predictionResult)
        {
            var defaultForeground = Console.ForegroundColor;
            var labelColor = ConsoleColor.Magenta;
            var probColor = ConsoleColor.Cyan;
            var exactLabel = ConsoleColor.Green;
            var failLabel = ConsoleColor.Red;

            Console.Write("ImagePath: ");
            Console.ForegroundColor = labelColor;
            Console.Write($"{Path.GetFileName(predictionResult.ImagePath)}");
            Console.ForegroundColor = defaultForeground;
            Console.Write(" labeled as ");
            Console.ForegroundColor = labelColor;
            Console.Write(predictionResult.Label);
            Console.ForegroundColor = defaultForeground;
            Console.Write(" predicted as ");
            if (predictionResult.Label.Equals(predictionResult.PredicateLabel))
            {
                Console.ForegroundColor = exactLabel;
                Console.Write($"{predictionResult.PredicateLabel}");
            }
            else
            {
                Console.ForegroundColor = failLabel;
                Console.Write($"{predictionResult.PredicateLabel}");
            }
            Console.ForegroundColor = defaultForeground;
            Console.Write(" with probability ");
            Console.ForegroundColor = probColor;
            Console.Write(predictionResult.Probability);
            Console.ForegroundColor = defaultForeground;
            Console.WriteLine("");
        }
    }
}
