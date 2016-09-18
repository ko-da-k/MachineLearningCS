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
            var svm = new SVM();
            var nn = new Neuro();
            foreach (var item in data.Where(x => x.Species == "setosa"))
            {
                Console.WriteLine("{0}\t{1}\t{2}\t{3}\t{4}",item.PetalLength,item.PetalWidth,item.SepalLength,item.SepalWidth,item.Species);
            }
        }
    }
}