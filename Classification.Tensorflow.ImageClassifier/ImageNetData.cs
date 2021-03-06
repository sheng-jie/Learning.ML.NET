﻿using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Classification.Tensorflow.ImageClassifier
{
    public class ImageNetData
    {
        public string ImagePath { get; set; }
        public string Label { get; set; }
        public static IEnumerable<ImageNetData> ReadFromCsv(string file,string folder)
        {
            return File.ReadAllLines(file)
                .Select(x => x.Split('\t'))
                .Select(x => new ImageNetData { ImagePath = Path.Combine(folder, x[0]), Label = x[1] });

        }
    }

    public class ImageNetDataProbability : ImageNetData
    {
        public string PredicateLabel { get; set; }
        public float Probability { get; set; }

    }
}