namespace Recommendation.MovieRecommender
{
    public class MovieRating
    {
        public float UserId { get; set; }
        public float MovieId { get; set; }
        public float Label { get; set; }
    }
}