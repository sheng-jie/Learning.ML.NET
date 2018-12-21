using Microsoft.ML.Runtime.Api;

namespace Classification.GithubIssueClassifier
{
    public class GithubIssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area { get; set; }
    }
}