using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SVM;

namespace SVMTest
{
    [TestClass]
    public class BinaryClassificationTests
    {
        [TestMethod]
        public void TestCLinear()
        {
            Problem train = Utilities.CreateTwoClassProblem(100);
            Parameter param = new Parameter();
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            param.Gamma = .5;
            param.SvmType = SvmType.C_SVC;
            param.KernelType = KernelType.LINEAR;
            param.Weights[-1] = 1;
            param.Weights[1] = 1;

            Model model = Training.Train(scaled, param);

            Problem test = Utilities.CreateTwoClassProblem(100, false);
            scaled = transform.Scale(test);
            double score = Prediction.Predict(scaled, null, model, false);

            Assert.AreEqual(1, score);
        }

        [TestMethod]
        public void TestCPolynomial()
        {
            Problem train = Utilities.CreateTwoClassProblem(100);
            Parameter param = new Parameter();
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            param.Gamma = .5;
            param.SvmType = SvmType.C_SVC;
            param.KernelType = KernelType.POLY;
            param.Weights[-1] = 1;
            param.Weights[1] = 1;

            Model model = Training.Train(scaled, param);

            Problem test = Utilities.CreateTwoClassProblem(100, false);
            scaled = transform.Scale(test);
            double score = Prediction.Predict(scaled, null, model, false);

            Assert.AreEqual(1, score);
        }

        [TestMethod]
        public void TestCRadialBasisFunction()
        {
            Problem train = Utilities.CreateTwoClassProblem(100);
            Parameter param = new Parameter();
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            param.Gamma = .5;
            param.SvmType = SvmType.C_SVC;
            param.KernelType = KernelType.RBF;
            param.Weights[-1] = 1;
            param.Weights[1] = 1;

            Model model = Training.Train(scaled, param);

            Problem test = Utilities.CreateTwoClassProblem(100, false);
            scaled = transform.Scale(test);
            double score = Prediction.Predict(scaled, null, model, false);

            Assert.AreEqual(1, score);
        }

        [TestMethod]
        public void TestCSigmoid()
        {
            Problem train = Utilities.CreateTwoClassProblem(100);
            Parameter param = new Parameter();
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            param.Gamma = .5;
            param.SvmType = SvmType.C_SVC;
            param.KernelType = KernelType.SIGMOID;
            param.Weights[-1] = 1;
            param.Weights[1] = 1;

            Model model = Training.Train(scaled, param);

            Problem test = Utilities.CreateTwoClassProblem(100, false);
            scaled = transform.Scale(test);
            double score = Prediction.Predict(scaled, null, model, false);

            Assert.AreEqual(1, score);
        }

        [TestMethod]
        public void TestNuLinear()
        {
            Problem train = Utilities.CreateTwoClassProblem(100);
            Parameter param = new Parameter();
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            param.Gamma = .5;
            param.SvmType = SvmType.NU_SVC;
            param.KernelType = KernelType.LINEAR;

            Model model = Training.Train(scaled, param);

            Problem test = Utilities.CreateTwoClassProblem(100, false);
            scaled = transform.Scale(train);
            double score = Prediction.Predict(scaled, null, model, false);

            Assert.AreEqual(1, score);
        }

        [TestMethod]
        public void TestNuPolynomial()
        {
            Problem train = Utilities.CreateTwoClassProblem(100);
            Parameter param = new Parameter();
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            param.Gamma = .5;
            param.SvmType = SvmType.NU_SVC;
            param.KernelType = KernelType.POLY;

            Model model = Training.Train(scaled, param);

            Problem test = Utilities.CreateTwoClassProblem(100, false);
            scaled = transform.Scale(train);
            double score = Prediction.Predict(scaled, null, model, false);

            Assert.AreEqual(1, score);
        }

        [TestMethod]
        public void TestNuRadialBasisFunction()
        {
            Problem train = Utilities.CreateTwoClassProblem(100);
            Parameter param = new Parameter();
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            param.Gamma = .5;
            param.SvmType = SvmType.NU_SVC;
            param.KernelType = KernelType.RBF;

            Model model = Training.Train(scaled, param);

            Problem test = Utilities.CreateTwoClassProblem(100, false);
            scaled = transform.Scale(train);
            double score = Prediction.Predict(scaled, null, model, false);

            Assert.AreEqual(1, score);
        }

        [TestMethod]
        public void TestNuSigmoid()
        {
            Problem train = Utilities.CreateTwoClassProblem(100);
            Parameter param = new Parameter();
            RangeTransform transform = RangeTransform.Compute(train);
            Problem scaled = transform.Scale(train);
            param.Gamma = .5;
            param.SvmType = SvmType.NU_SVC;
            param.KernelType = KernelType.SIGMOID;

            Model model = Training.Train(scaled, param);

            Problem test = Utilities.CreateTwoClassProblem(100, false);
            scaled = transform.Scale(train);
            double score = Prediction.Predict(scaled, null, model, false);

            Assert.AreEqual(1, score);
        }
    }
}
