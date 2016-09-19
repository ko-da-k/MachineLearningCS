using System;
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
            var data = LoadData.LoadCsv<Iris,IrisMap>("iris.csv");
            var labelname = data.GroupBy(x => x.Species).Select(x => x.Key).ToList();
            var train = (from n in data select new double[]{n.PetalLength,n.PetalWidth,n.SepalLength,n.SepalWidth}).ToArray();
            var label = data.Select(x => labelname.IndexOf(x.Species)).ToArray();
            Console.WriteLine("{0}",data.GroupBy(x => x.Species).Count());
            var svm = new SVM(train, label);
            svm.learn();
            svm.predict(train[0]);
        }
    }
}