using Microsoft.ML.Runtime.Api;

namespace Clustering.IrisCluster
{
    public class IrisPrediction
    {

        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId { get; set; }

        [ColumnName("Score")]
        public float[] Distances { get; set; }
    }
}