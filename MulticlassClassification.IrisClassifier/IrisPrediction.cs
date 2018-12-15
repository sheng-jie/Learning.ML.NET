using Microsoft.ML.Runtime.Api;

namespace MulticlassClassification.IrisClassifier
{
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels { get; set; }
    }
}