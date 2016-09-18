using System;
using System.Text;
using System.IO;
using System.Linq;
using Accord.Neuro;
using Accord.Neuro.Networks;
using Accord.Neuro.Learning;
using AForge.Neuro.Learning;
using Accord.Neuro.ActivationFunctions;
using Accord.Math;


namespace ML
{
    class NeuralTrainData
    {
        //トレーニングデータ.今回は6次元
        public double[][] train =
        {
            new double[] { 1, 1, 1, 0, 0, 0 },
            new double[] { 1, 0, 1, 0, 0, 0 },
            new double[] { 1, 1, 1, 0, 0, 0 },
            new double[] { 0, 0, 1, 1, 1, 0 },
            new double[] { 0, 0, 1, 1, 0, 0 },
            new double[] { 0, 0, 1, 1, 1, 0 },
        };

        //トレーニングデータに対応するラベル
        //NNの出力層が2次元のためこの形
        public double[][] label =
        {
            new double[] { 1, 0 },
            new double[] { 1, 0 },
            new double[] { 1, 0 },
            new double[] { 0, 1 },
            new double[] { 0, 1 },
            new double[] { 0, 1 },
        };
    }

    class NN : NeuralTrainData
    {
        public void learn()
        {
            //Deep Belief Networks(DBN)の生成
            DeepBeliefNetwork net = new DeepBeliefNetwork(
                inputsCount:train.Length, //入力層の次元
                hiddenNeurons:new int[] {4,4,2}); //隠れ層と出力層の次元
            
            //ネットワークの重みをガウス分布で初期化する
            new GaussianWeights(net).Randomize();
            net.UpdateVisibleWeights();

            //学習アルゴリズムの生成
            var teacher = new BackPropagationLearning(net);
            for (int i = 0; i < 5000; i++) //5000回学習
            {
                teacher.RunEpoch(train, label);
            }
            net.UpdateVisibleWeights();

            double[] input = { 1, 1, 1, 1, 0, 0 };
            var output = net.Compute(input);

            int imax;
            output.Max(out imax);

            Console.WriteLine("class : {0}", imax);
            foreach (var item in output)
            {
                Console.Write("{0}", item);    
            }
        }
    }
}
