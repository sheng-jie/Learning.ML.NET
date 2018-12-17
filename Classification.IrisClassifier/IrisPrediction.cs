using Microsoft.ML.Runtime.Api;

namespace Classification.IrisClassifier
{
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels { get; set; }
    }
}