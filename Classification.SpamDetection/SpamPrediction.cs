using Microsoft.ML.Runtime.Api;

namespace Classification.SpamDetection
{
    public class SpamPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsSpam { get; set; }

        public float Score { get; set; }
        public float Probability { get; set; }

    }
}