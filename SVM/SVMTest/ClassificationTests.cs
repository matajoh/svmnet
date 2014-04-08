using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SVM;

namespace SVMTest
{
    [TestClass]
    public class ClassificationTests
    {
        [TestMethod]
        public void TestTwoClass()
        {
            SvmType[] svmTypes = new SvmType[]{SvmType.C_SVC, SvmType.NU_SVC};
            KernelType[] kernelTypes = new KernelType[]{KernelType.LINEAR, KernelType.POLY, KernelType.RBF, KernelType.SIGMOID};

            foreach (SvmType svm in svmTypes)
            {
                foreach (KernelType kernel in kernelTypes)
                {
                    double score = Utilities.TestTwoClassModel(100, svm, kernel);

                    Assert.AreEqual(1, score, .01, string.Format("SVM {0} with Kernel {1} did not train correctly", svm, kernel));
                }
            }
        }

        [TestMethod]
        public void TestTwoClassProbability()
        {
            SvmType[] svmTypes = new SvmType[] { SvmType.C_SVC, SvmType.NU_SVC };
            KernelType[] kernelTypes = new KernelType[] { KernelType.LINEAR, KernelType.POLY, KernelType.RBF, KernelType.SIGMOID };

            foreach (SvmType svm in svmTypes)
            {
                foreach (KernelType kernel in kernelTypes)
                {
                    double score = Utilities.TestTwoClassModel(100, svm, kernel, true);

                    Assert.AreEqual(1, score, .01, string.Format("SVM {0} with Kernel {1} did not train correctly", svm, kernel));
                }
            }
        }

        [TestMethod]
        public void TestMulticlass()
        {
            SvmType[] svmTypes = new SvmType[] { SvmType.C_SVC, SvmType.NU_SVC };
            KernelType[] kernelTypes = new KernelType[] { KernelType.LINEAR, KernelType.POLY, KernelType.RBF, KernelType.SIGMOID };

            foreach (SvmType svm in svmTypes)
            {
                foreach (KernelType kernel in kernelTypes)
                {
                    double score = Utilities.TestMulticlassModel(8, 100, svm, kernel);

                    Assert.AreEqual(1, score, .1, string.Format("SVM {0} with Kernel {1} did not train correctly", svm, kernel));
                }
            }
        }

        [TestMethod]
        public void TestMulticlassProbability()
        {
            SvmType[] svmTypes = new SvmType[] { SvmType.C_SVC, SvmType.NU_SVC };
            KernelType[] kernelTypes = new KernelType[] { KernelType.LINEAR, KernelType.POLY, KernelType.RBF, KernelType.SIGMOID };

            foreach (SvmType svm in svmTypes)
            {
                foreach (KernelType kernel in kernelTypes)
                {
                    double score = Utilities.TestMulticlassModel(8, 100, svm, kernel, true);

                    Assert.AreEqual(1, score, .1, string.Format("SVM {0} with Kernel {1} did not train correctly", svm, kernel));
                }
            }
        }
    }
}
