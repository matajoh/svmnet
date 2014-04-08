using SVM;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVMTest
{
    static class Utilities
    {
        private const double SCALE = 100;
        public const int TRAINING_SEED = 20080524;
        public const int TESTING_SEED = 20140407;

        public static Problem CreateTwoClassProblem(int count, bool isTraining = true)
        {
            Problem prob = new Problem();
            prob.Count = count;
            prob.MaxIndex = 2;
            
            Random rand = new Random(isTraining ? TRAINING_SEED : TESTING_SEED);
            // create points on either side of the vertical axis
            int positive = count / 2;
            List<double> labels = new List<double>();
            List<Node[]> data = new List<Node[]>();
            for (int i = 0; i < count; i++)
            {
                double x = rand.NextDouble() * SCALE + 10;
                double y = rand.NextDouble() * SCALE - (SCALE * .5);
                x = i < positive ? x : -x;
                data.Add(new Node[] { new Node(1, x), new Node(2, y) });
                labels.Add(i < positive ? 1 : -1);
            }
            prob.X = data.ToArray();
            prob.Y = labels.ToArray();

            return prob;
        }
    }
}
