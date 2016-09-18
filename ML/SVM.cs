using System;
using System.Text;
using System.IO;
using System.Linq;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.IO;

namespace ML
{
    class SVMTrainData
    {
        public double[][] train;
        public int[] label;
        public SVMTrainData()
        {
            Random rnd = new Random();
            train = new double[][] 
                {
                    new double[] {rnd.Next(0,5),rnd.Next(0,5)},
                    new double[] {rnd.Next(0,5),rnd.Next(0,5)},
                    new double[] {rnd.Next(0,5),rnd.Next(0,5)},
                    new double[] {rnd.Next(0,5),rnd.Next(0,5)},
                    new double[] {rnd.Next(5,10),rnd.Next(5,10)},
                    new double[] {rnd.Next(5,10),rnd.Next(5,10)},
                    new double[] {rnd.Next(5,10),rnd.Next(5,10)},
                    new double[] {rnd.Next(5,10),rnd.Next(5,10)},
                };
            label = new int[] { -1,-1,-1,-1, 1, 1, 1, 1 };
        }
    }

    class SVM : SVMTrainData
    {
        public void Learn()
        {
            SupportVectorMachine svm = new SupportVectorMachine(train[0].Length);
            var teacher = new LinearCoordinateDescent(svm, train, label);

            double error = teacher.Run();

            var answers = svm.Compute(train[0]);
            Console.WriteLine("{0}",answers);
            Console.WriteLine("{0}",error);
        }
    }
}

