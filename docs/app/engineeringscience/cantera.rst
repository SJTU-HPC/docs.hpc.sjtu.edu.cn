.. _cantera:

Cantera
==========

简介
----

Cantera是一个化学反应动力学开源求解套件，可以处理Chemkin格式的动力学机理、热力学和输运性质文件，通过适当编程即可实现常用反应器(激波管、快速压缩机、搅拌反应器等)的模拟及敏感性分析等功能。



Cantera使用说明
-----------------------------

在思源一号上自行安装并使用Cantera
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1. 使用conda创建虚拟环境并安装Cantera：

.. code::
        
  srun -p 64c512g -n 1 --pty /bin/bash
  module load miniconda3/4.10.3
  conda create --name cantera_test
  source  activate cantera_test
  conda install -c cantera/label/dev cantera

2. 创建一个目录canteratest并进入该目录：

.. code::
        
    mkdir canteratest
    cd canteratest

3. 在该目录下创建如下测试文件cantera_test.py：

.. code::
        
  """
  Compute the "equilibrium" and "frozen" sound speeds for a gas

  Requires: cantera >= 2.5.0
  Keywords: thermodynamics, equilibrium
  """

  import cantera as ct
  import math


  def equilSoundSpeeds(gas, rtol=1.0e-6, max_iter=5000):
      """
      Returns a tuple containing the equilibrium and frozen sound speeds for a
      gas with an equilibrium composition.  The gas is first set to an
      equilibrium state at the temperature and pressure of the gas, since
      otherwise the equilibrium sound speed is not defined.
      """

      # set the gas to equilibrium at its current T and P
      gas.equilibrate('TP', rtol=rtol, max_iter=max_iter)

      # save properties
      s0 = gas.s
      p0 = gas.P
      r0 = gas.density

      # perturb the pressure
      p1 = p0*1.0001

      # set the gas to a state with the same entropy and composition but
      # the perturbed pressure
      gas.SP = s0, p1

      # frozen sound speed
      afrozen = math.sqrt((p1 - p0)/(gas.density - r0))

      # now equilibrate the gas holding S and P constant
      gas.equilibrate('SP', rtol=rtol, max_iter=max_iter)

      # equilibrium sound speed
      aequil = math.sqrt((p1 - p0)/(gas.density - r0))

      # compute the frozen sound speed using the ideal gas expression as a check
      gamma = gas.cp/gas.cv
      afrozen2 = math.sqrt(gamma * ct.gas_constant * gas.T /
                           gas.mean_molecular_weight)

      return aequil, afrozen, afrozen2


  # test program
  if __name__ == "__main__":
      gas = ct.Solution('gri30.yaml')
      gas.X = 'CH4:1.00, O2:2.0, N2:7.52'
      for n in range(27):
          T = 300.0 + 100.0 * n
          gas.TP = T, ct.one_atm
          print(T, equilSoundSpeeds(gas))


4. 在该目录下创建如下作业提交脚本canteratest.slurm:

.. code::

  #!/bin/bash
  
  #SBATCH --job-name=canteratest      
  #SBATCH --partition=64c512g      
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  python3 cantera_test.py

5. 使用如下命令提交作业：

.. code::

  sbatch canteratest.slurm

6. 作业完成后在.out文件中可看到如下结果：

.. code::

  300.0 (351.81897215910766, 351.8225040320234, 351.8256504699811)
  400.0 (404.3283414405797, 404.60713567095115, 404.6106870323977)
  500.0 (450.60495699772054, 450.5397010505711, 450.54335869692443)
  600.0 (493.2014592526757, 491.4792382523635, 491.4832797222876)
  700.0 (531.9259905892586, 528.573790448385, 528.5780936855717)
  800.0 (565.8023687396801, 562.6519887313171, 562.6563319848512)
  900.0 (594.787246530831, 594.3886921418637, 594.3928468410445)
  1000.0 (623.1834570351056, 623.1967311658512, 624.387559050875)
  1100.0 (652.9331414338375, 653.0237878409358, 653.0280730486886)
  1200.0 (680.2903244154709, 680.3796031034527, 680.3839821434552)
  1300.0 (706.4795851760939, 706.6231852692649, 706.627654889073)
  1400.0 (731.5765086318436, 731.8946265500344, 731.8991788491217)
  1500.0 (755.5871134959623, 756.3126654597705, 756.3172844867763)
  1600.0 (778.4547221738709, 779.9838934521655, 779.9885516424242)
  1700.0 (800.0747386258183, 803.0113981678297, 803.0160523778995)
  1800.0 (820.3501225230362, 825.5029866841091, 825.5075790369601)
  1900.0 (839.2604068697004, 847.5787433425231, 847.5832049739269)
  2000.0 (856.9424757185266, 869.3775874590473, 869.3818488069585)
  2100.0 (873.7271469549471, 891.0618214354686, 891.0658256580159)
  2200.0 (889.9263467742846, 912.8186991059179, 912.8224139325766)
  2300.0 (906.658936254018, 934.8583105569138, 934.8617342865722)
  2400.0 (923.8715214357717, 957.4080987743781, 957.411256949987)
  2500.0 (942.1089113756211, 980.7060227583402, 980.7089610687678)
  2600.0 (961.7644659835721, 1004.9962598901973, 1004.9990315355183)
  2700.0 (982.983503202847, 1030.5311450749714, 1030.5338048850456)
  2800.0 (1005.8548751928753, 1057.5791215213358, 1057.5817227102573)
  2900.0 (1030.5547401650322, 1086.430611844347, 1086.433204007728)


在pi2.0上自行安装并使用Cantera
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 使用conda创建虚拟环境并安装Cantera：

.. code::
        
  srun -p small -n 1 --pty /bin/bash
  module load miniconda3/4.8.2
  conda create --name cantera_test
  source  activate cantera_test
  conda install -c cantera/label/dev cantera



2. 此步骤和上文完全相同；



3. 此步骤和上文完全相同；


4. 在该目录下创建如下作业提交脚本canteratest.slurm:

.. code::

  #!/bin/bash
  
  #SBATCH --job-name=canteratest      
  #SBATCH --partition=small    
  #SBATCH --ntasks-per-node=1     
  #SBATCH -n 1                     
  #SBATCH --output=%j.out
  #SBATCH --error=%j.err

  ulimit -s unlimited
  ulimit -l unlimited

  python3 cantera_test.py

5. 使用如下命令提交作业：

.. code::

  sbatch canteratest.slurm

6. 作业完成后在.out文件中可看到如下结果：

.. code::

  300.0 (351.81897215910766, 351.8225040320234, 351.8256504699811)
  400.0 (404.3283414405797, 404.60713567095115, 404.6106870323977)
  500.0 (450.60495699772054, 450.5397010505711, 450.54335869692443)
  600.0 (493.2014592526757, 491.4792382523635, 491.4832797222876)
  700.0 (531.9259905892586, 528.573790448385, 528.5780936855717)
  800.0 (565.8023687396801, 562.6519887313171, 562.6563319848512)
  900.0 (594.787246530831, 594.3886921418637, 594.3928468410445)
  1000.0 (623.1834570351056, 623.1967311658512, 624.387559050875)
  1100.0 (652.9331414338375, 653.0237878409358, 653.0280730486886)
  1200.0 (680.2903244154709, 680.3796031034527, 680.3839821434552)
  1300.0 (706.4795851760939, 706.6231852692649, 706.627654889073)
  1400.0 (731.5765086318436, 731.8946265500344, 731.8991788491217)
  1500.0 (755.5871134959623, 756.3126654597705, 756.3172844867763)
  1600.0 (778.4547221738709, 779.9838934521655, 779.9885516424242)
  1700.0 (800.0747386258183, 803.0113981678297, 803.0160523778995)
  1800.0 (820.3501225230362, 825.5029866841091, 825.5075790369601)
  1900.0 (839.2604068697004, 847.5787433425231, 847.5832049739269)
  2000.0 (856.9424757185266, 869.3775874590473, 869.3818488069585)
  2100.0 (873.7271469549471, 891.0618214354686, 891.0658256580159)
  2200.0 (889.9263467742846, 912.8186991059179, 912.8224139325766)
  2300.0 (906.658936254018, 934.8583105569138, 934.8617342865722)
  2400.0 (923.8715214357717, 957.4080987743781, 957.411256949987)
  2500.0 (942.1089113756211, 980.7060227583402, 980.7089610687678)
  2600.0 (961.7644659835721, 1004.9962598901973, 1004.9990315355183)
  2700.0 (982.983503202847, 1030.5311450749714, 1030.5338048850456)
  2800.0 (1005.8548751928753, 1057.5791215213358, 1057.5817227102573)
  2900.0 (1030.5547401650322, 1086.430611844347, 1086.433204007728)


  



参考资料
-----------

-  `Cantera 官网 <https://cantera.org/>`__
-  `安装 Cantera 知乎 <https://zhuanlan.zhihu.com/p/546180253>`__

