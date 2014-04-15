using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SVM;

namespace SVMTest
{
    [TestClass]
    public class RegressionTests
    {
        [TestMethod]
        public void TestRegression()
        {
            SvmType[] svmTypes = new SvmType[] { SvmType.NU_SVR, SvmType.EPSILON_SVR };
            // LINEAR kernel is pretty horrible for regression
            KernelType[] kernelTypes = new KernelType[] { KernelType.LINEAR, KernelType.RBF, KernelType.SIGMOID };

            foreach (SvmType svm in svmTypes)
            {
                foreach (KernelType kernel in kernelTypes)
                {
                    double error = Utilities.TestRegressionModel(100, svm, kernel);

                    Assert.AreEqual(0, error, 2, string.Format("SVM {0} with Kernel {1} did not train correctly", svm, kernel));
                }
            }
        }
    }
}
