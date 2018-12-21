using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Recommendation.MovieRecommender
{
    public class Movie
    {
        public int MovieId { get; set; }

        public string MovieTitle { get; set; }

        public string MovieType { get; set; }

        public Lazy<List<Movie>> Movies = new Lazy<List<Movie>>(
            LoadMovieData);

        public Movie Get(int id)
        {
            return Movies.Value.FirstOrDefault(m => m.MovieId == id);
        }
        private static List<Movie> LoadMovieData()
        {
            string filePath = Path.Combine(Environment.CurrentDirectory, "Data", "movies-data.csv");

            var result = from line in File.ReadAllLines(filePath).AsSpan(1).ToArray()
                select new Movie()
                {
                    MovieId = Convert.ToInt32(line.Split(",")[0]),
                    MovieTitle = Convert.ToString(line.Split(",")[1]),
                    MovieType = Convert.ToString(line.Split(",")[2])
                };

            return result.ToList();
        }
    }
}