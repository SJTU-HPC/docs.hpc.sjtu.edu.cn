.. _eigen:

Eigen
==========

简介
----

Eigen库是一个开源的矩阵运算库，其利用C++模板编程的思想，构造所有矩阵通过传递模板参数形式完成。由于模板类不支持库链接方式编译，而且模板类要求全部写在头文件中，从而导致导致Eigen库只能通过开源的方式供大家使用，并且只需要包含Eigen头文件就能直接使用。



Eigen使用说明
-----------------------------

思源一号上的Eigen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 先创建一个目录eigentest并进入该目录：

.. code::
        
    mkdir eigentest
    cd eigentest

2. 在该目录下创建如下测试文件myeigen.cpp：

.. code::
        
    #include <vector>
    #include <Eigen/Sparse>
    #include <Eigen/SparseLU>
    #include <chrono>
    #include <random>
    #include <algorithm>
    #include <stdlib.h>
    #include <iostream>
    #include <ctime>

    using namespace std;
    using namespace Eigen;
    using namespace chrono;

    typedef SparseMatrix<double> SpMat;
    typedef Triplet<double> T;

    int wlsFilter_1d(vector<double> &vecInput, vector<double> &vecOutput, vector<double> &vecL, double lambda, double alpha, double smallNum)
    {
        int number = vecInput.size();
        if (vecL.size() != number || number < 2)
            return -1;

        vector<double> vecDx(number + 1);
        vecDx[0] = 0;
        vecDx[number] = 0;
        for (int i = 1; i < number; i++)
        {
            vecDx[i] = lambda * (pow(abs(vecL[i] - vecL[i - 1]) + smallNum, alpha));
        }

        vector<Triplet<double>> tripletList(3 * number - 2);
        for (int i = 0; i < number; i++)
        {
            tripletList[i] = T(i, i, 1 + vecDx[i] + vecDx[i + 1]);
        }
        for (int i = 1; i < number; i++)
        {
            tripletList[number + i - 1] = T(i, i - 1, -vecDx[i]);
        }
        for (int i = 1; i < number; i++)
        {
            tripletList[2 * number + i - 2] = T(i - 1, i, -vecDx[i]);
        }

        SpMat A(number, number);
        A.setFromTriplets(tripletList.begin(), tripletList.end());

        // cout << A << endl;

        VectorXd b(number);
        for (int i = 0; i < number; i++)
        {
            b(i) = vecInput[i];
        }

        auto solver_time1 = microseconds(0);
        auto solver_time2 = microseconds(0);
        auto solver_time3 = microseconds(0);
        auto solver_time4 = microseconds(0);
        auto solver_time5 = microseconds(0);
        auto solver_time6 = microseconds(0);
        auto solver_time7 = microseconds(0);

        cout << "=====================几种不同求解方法的计算结果=====================" << endl;

        auto start1 = high_resolution_clock::now();
        BiCGSTAB<SpMat> solver1(A);
        VectorXd x1 = solver1.solve(b);
        auto end1 = high_resolution_clock::now();
        solver_time1 = duration_cast<microseconds>(end1 - start1);
        cout << "x1 =" << endl;
        for (int i = 0; i < 10; i++)
        {
            cout << x1[i] << ",";
        }
        cout << endl;
        cout << "================================================================" << endl;

        auto start2 = high_resolution_clock::now();
        SparseLU<SpMat> solver2(A);
        VectorXd x2 = solver2.solve(b);
        auto end2 = high_resolution_clock::now();
        solver_time2 = duration_cast<microseconds>(end2 - start2);
        cout << "x2 =" << endl;
        for (int i = 0; i < 10; i++)
        {
            cout << x2[i] << ",";
        }
        cout << endl;
        cout << "================================================================" << endl;

        auto start3 = high_resolution_clock::now();
        SparseQR<SpMat, COLAMDOrdering<int>> solver3(A);
        VectorXd x3 = solver3.solve(b);
        auto end3 = high_resolution_clock::now();
        solver_time3 = duration_cast<microseconds>(end3 - start3);
        cout << "x3 =" << endl;
        for (int i = 0; i < 10; i++)
        {
            cout << x3[i] << ",";
        }
        cout << endl;
        cout << "================================================================" << endl;

        auto start4 = high_resolution_clock::now();
        SimplicialLLT<SpMat> solver4(A);
        VectorXd x4 = solver4.solve(b);
        auto end4 = high_resolution_clock::now();
        solver_time4 = duration_cast<microseconds>(end4 - start4);
        cout << "x4 =" << endl;
        for (int i = 0; i < 10; i++)
        {
            cout << x4[i] << ",";
        }
        cout << endl;
        cout << "================================================================" << endl;

        auto start5 = high_resolution_clock::now();
        SimplicialLDLT<SpMat> solver5(A);
        VectorXd x5 = solver5.solve(b);
        auto end5 = high_resolution_clock::now();
        solver_time5 = duration_cast<microseconds>(end5 - start5);
        cout << "x5 =" << endl;
        for (int i = 0; i < 10; i++)
        {
            cout << x5[i] << ",";
        }
        cout << endl;
        cout << "================================================================" << endl;

        auto start6 = high_resolution_clock::now();
        ConjugateGradient<SpMat> solver6(A);
        VectorXd x6 = solver6.solve(b);
        auto end6 = high_resolution_clock::now();
        solver_time6 = duration_cast<microseconds>(end6 - start6);
        cout << "x6 =" << endl;
        for (int i = 0; i < 10; i++)
        {
            cout << x6[i] << ",";
        }
        cout << endl;
        cout << "================================================================" << endl;

        auto start7 = high_resolution_clock::now();
        LeastSquaresConjugateGradient<SpMat> solver7(A);
        VectorXd x7 = solver7.solve(b);
        auto end7 = high_resolution_clock::now();
        solver_time7 = duration_cast<microseconds>(end7 - start7);
        cout << "x7 =" << endl;
        for (int i = 0; i < 10; i++)
        {
            cout << x7[i] << ",";
        }
        cout << endl;
        cout << endl;

        cout << "=====================几种不同求解方法消耗的时间=====================" << endl;
        cout << "  Time taken by solve1: " << 1.0 * solver_time1.count() / 1000 << " milliseconds" << endl;
        cout << "  Time taken by solve2: " << 1.0 * solver_time2.count() / 1000 << " milliseconds" << endl;
        cout << "  Time taken by solve3: " << 1.0 * solver_time3.count() / 1000 << " milliseconds" << endl;
        cout << "  Time taken by solve4: " << 1.0 * solver_time4.count() / 1000 << " milliseconds" << endl;
        cout << "  Time taken by solve5: " << 1.0 * solver_time5.count() / 1000 << " milliseconds" << endl;
        cout << "  Time taken by solve6: " << 1.0 * solver_time6.count() / 1000 << " milliseconds" << endl;
        cout << "  Time taken by solve7: " << 1.0 * solver_time7.count() / 1000 << " milliseconds" << endl;

        return 0;
    }

    int main()
    {

        std::default_random_engine e;
        std::uniform_real_distribution<double> u(-1.0, 1.0); // 左闭右闭区间
        e.seed(time(0));

        vector<double> x;
        double temp = 0;
        double step = 0.005;
        while (temp < 10)
        {
            x.push_back(temp);
            temp += step;
        }
        vector<double> y(x.size());
        int bgIdx = 30;
        int endIdx = 60;
        for (int i = 0; i < x.size(); i++)
        {
            y[i] = 0.2 * x[i] * x[i] + 2 * x[i];
        }
        for (int i = bgIdx; i < endIdx; i++)
        {
            y[i] += 3 * u(e);
        }
        double ymax = *max_element(y.begin(), y.end());
        vector<double> vecL(x.size());
        for (int i = 0; i < vecL.size(); i++)
        {
            vecL[i] = y[i] / ymax;
        }

        double alpha = 0.5;
        double lambda = 100;
        double smallNum = 0.1;
        vector<double> yfilt;

        int res = wlsFilter_1d(y, yfilt, vecL, lambda, alpha, smallNum);

        return 0;
    }
 

3. 在该目录下创建如下作业提交脚本eigentest.slurm:

.. code::

  #!/bin/bash
  
  #SBATCH --job-name=eigentest      
  #SBATCH --partition=64c512g      
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load eigen/3.4.0-gcc-11.2.0
  module load gcc/11.2.0

  g++  myeigen.cpp -o myeigen

  ./myeigen

4. 使用如下命令提交作业：

.. code::

  sbatch eigentest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

    =====================几种不同求解方法的计算结果=====================
    x1 =
    0.0582939,0.060135,0.0635594,0.0683589,0.0743685,0.081461,0.0895431,0.0985527,0.108456,0.119248,
    ================================================================
    x2 =
    0.0582939,0.060135,0.0635594,0.0683589,0.0743685,0.081461,0.0895431,0.0985527,0.108456,0.119248,
    ================================================================
    x3 =
    0.0582939,0.060135,0.0635594,0.0683589,0.0743685,0.081461,0.0895431,0.0985527,0.108456,0.119248,
    ================================================================
    x4 =
    0.0582939,0.060135,0.0635594,0.0683589,0.0743685,0.081461,0.0895431,0.0985527,0.108456,0.119248,
    ================================================================
    x5 =
    0.0582939,0.060135,0.0635594,0.0683589,0.0743685,0.081461,0.0895431,0.0985527,0.108456,0.119248,
    ================================================================
    x6 =
    0.0582939,0.060135,0.0635594,0.0683589,0.0743685,0.081461,0.0895431,0.0985527,0.108456,0.119248,
    ================================================================
    x7 =
    0.0582939,0.060135,0.0635594,0.0683589,0.0743685,0.081461,0.0895431,0.0985527,0.108456,0.119248,

    =====================几种不同求解方法消耗的时间=====================
    Time taken by solve1: 95.936 milliseconds
    Time taken by solve2: 32.905 milliseconds
    Time taken by solve3: 34.937 milliseconds
    Time taken by solve4: 6.422 milliseconds
    Time taken by solve5: 6.48 milliseconds
    Time taken by solve6: 120.568 milliseconds
    Time taken by solve7: 1356.32 milliseconds



pi2.0上的Eigen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 此步骤和上文完全相同；



2. 此步骤和上文完全相同；



3. 在该目录下创建如下作业提交脚本eigentest.slurm:

.. code::

  #!/bin/bash

  #SBATCH --job-name=eigentest    
  #SBATCH --partition=small     
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  module load eigen/3.4.0-gcc-11.2.0
  module load gcc/11.2.0

  g++  myeigen.cpp -o myeigen

  ./myeigen

4. 使用如下命令提交作业：

.. code::

  sbatch eigentest.slurm

5. 作业完成后在.out文件中可看到如下结果：

.. code::

    =====================几种不同求解方法的计算结果=====================
    x1 =
    0.0535321,0.0552228,0.0583417,0.0626708,0.0680304,0.0742728,0.0812777,0.0889491,0.0972111,0.106007,
    ================================================================
    x2 =
    0.0535321,0.0552228,0.0583417,0.0626708,0.0680304,0.0742728,0.0812777,0.0889491,0.0972111,0.106007,
    ================================================================
    x3 =
    0.0535321,0.0552228,0.0583417,0.0626708,0.0680304,0.0742728,0.0812777,0.0889491,0.0972111,0.106007,
    ================================================================
    x4 =
    0.0535321,0.0552228,0.0583417,0.0626708,0.0680304,0.0742728,0.0812777,0.0889491,0.0972111,0.106007,
    ================================================================
    x5 =
    0.0535321,0.0552228,0.0583417,0.0626708,0.0680304,0.0742728,0.0812777,0.0889491,0.0972111,0.106007,
    ================================================================
    x6 =
    0.0535321,0.0552228,0.0583417,0.0626708,0.0680304,0.0742728,0.0812777,0.0889491,0.0972111,0.106007,
    ================================================================
    x7 =
    0.0535321,0.0552228,0.0583417,0.0626708,0.0680304,0.0742728,0.0812777,0.0889491,0.0972111,0.106007,

    =====================几种不同求解方法消耗的时间=====================
    Time taken by solve1: 93.473 milliseconds
    Time taken by solve2: 32.919 milliseconds
    Time taken by solve3: 35.014 milliseconds
    Time taken by solve4: 6.436 milliseconds
    Time taken by solve5: 6.48 milliseconds
    Time taken by solve6: 120.658 milliseconds
    Time taken by solve7: 1356.61 milliseconds



  



参考资料
-----------

-  `Eigen 官网 <https://eigen.tuxfamily.org/index.php?title=Main_Page>`__
-  `Eigen 知乎 <https://zhuanlan.zhihu.com/p/462494086>`__

