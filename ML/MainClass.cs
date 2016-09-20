using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using CsvHelper;

namespace ML
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            var data = LoadData.LoadCsv<Iris, IrisMap>("iris.csv");
            var labelname = data.Select(x => x.Species).Distinct().ToArray();

            var train = (from n in data select new double[] { n.PetalLength, n.PetalWidth, n.SepalLength, n.SepalWidth }).ToArray();
            var label = data.Select(x => Array.IndexOf(labelname, x.Species)).ToArray();

            ClassificationReport(new int[] { 1, 0, 1, 0, 1 }, new int[] { 1, 1, 1, 0, 0 });

            var svm = new SVM(train, label);
            svm.learn();
            svm.predict(train[0]);
        }
        /// <summary>
        /// Classifications the report.
        /// </summary>
        /// <param name="truelabel">正解ラベル</param>
        /// <param name="predict">予測結果</param>
        public static void ClassificationReport(int[] truelabel, int[] predict)
        {
            Debug.Assert(truelabel.Length != predict.Length);
            var labelset = truelabel.Distinct().ToArray();
            var tfarray = new bool[truelabel.Length];
            double precision;
            double recall;
            for (int i = 0; i < truelabel.Length; i++)
            {
                tfarray[i] = truelabel[i] == predict[i];
            }
            Console.WriteLine("label\tPre\tRec\tF-score");
            foreach (var item in labelset)
            {
                // ラベルがitemのもののindexを正解ラベルからとってくる
                var preIndex = truelabel.Select((x, i) => new { Content = x, Index = i })
                                        .Where(x => x.Content == item)
                                        .Select(x => x.Index).ToArray();
                var recIndex = predict.Select((x, i) => new { Content = x, Index = i })
                                      .Where(x => x.Content == item)
                                      .Select(x => x.Index).ToArray();
                precision = tfarray.Select((x, i) => new { Content = x, Index = i })
                                   .Count(x => 0 <= Array.IndexOf(preIndex, x.Index) && x.Content == true)
                                   / (double)preIndex.Length;
                recall = tfarray.Select((x, i) => new { Content = x, Index = i })
                                .Count(x => 0 <= Array.IndexOf(recIndex, x.Index) && x.Content == true)
                                / (double)recIndex.Length;
                Console.WriteLine("{0}\t{1:f2}\t{2:f2}\t{3:f2}", item, precision, recall,
                                  (2 * precision * recall) / (precision + recall));

            }
        }
    }
}