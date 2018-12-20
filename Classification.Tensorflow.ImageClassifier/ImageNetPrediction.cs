using Microsoft.ML.Runtime.Api;

namespace Classification.Tensorflow.ImageClassifier
{
    public class ImageNetPrediction
    {
        [ColumnName("softmax2")]
        public float[] PredictedLabels { get; set; }
    }
}