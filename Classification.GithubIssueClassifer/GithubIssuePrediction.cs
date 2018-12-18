using Microsoft.ML.Runtime.Api;

namespace Classification.GithubIssueClassifer
{
    public class GithubIssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area { get; set; }
    }
}