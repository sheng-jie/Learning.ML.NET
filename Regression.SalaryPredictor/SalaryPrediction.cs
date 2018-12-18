using Microsoft.ML.Runtime.Api;

namespace Regression.SalaryPredictor
{
    public class SalaryPrediction
    {
        [ColumnName("Score")]
        public float PredictedSalary{ get; set; }
    }
}