using Microsoft.ML.Runtime.Api;

namespace Regression.DiamondPricePredictor
{
    public class DiamondPricePrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }
}