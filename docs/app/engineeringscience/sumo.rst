.. _sumo:

SUMO
=====================

简介
---------------

SUMO，全称Simulation of Urban Mobility，是开源、微观、多模态的交通仿真软件，发展始于2000年。
它纯粹是微观的，可以针对每辆车进行单独控制，因此非常适合交通控制模型的开发。


SUMO(GUI) 使用方法  
---------------------------

.. code:: bash

   1. 用pi集群帐号登录 https://studio.hpc.sjtu.edu.cn/ 平台
   2. 在网站通过  Interactive Apps -> Desktop -> Launch 进入桌面(注意使用GPU桌面)
   3. 打开终端，通过命令行来调用软件

      module purge
      module load sumo/1.10.0-sumo
      sumo-gui 

   4. 从下图可知，一次模拟我们需要rou.xml 和net.xml 两个文件。其中rou.xml用来表述交通需求，net.xml用来表述道路信息。
      而道路信息又由下面四个文件通过netconvert命令生成:

      nod.xml 用来描述节点信息
      edg.xml 用来描述边的信息
      typ.xml 用来描述预定义的边的类型（类似于做一个封装）
      con.xml 用来描述边到边的合并形式

|image1|

---------------------------

.. code:: bash

   Building network using xml

      编写nod.xml
      贴出官网的例子,exa.nod.xml :
      <nodes> <!-- The opening tag -->
   <node id="0" x="0.0" y="0.0" type="traffic_light"/> <!-- def. of node "0" -->
   <node id="1" x="-500.0" y="0.0" type="priority"/> <!-- def. of node "1" -->
   <node id="2" x="+500.0" y="0.0" type="priority"/> <!-- def. of node "2" -->
   <node id="3" x="0.0" y="-500.0" type="priority"/> <!-- def. of node "3" -->
   <node id="4" x="0.0" y="+500.0" type="priority"/> <!-- def. of node "4" -->
   <node id="m1" x="-250.0" y="0.0" type="priority"/> <!-- def. of node "m1" -->
   <node id="m2" x="+250.0" y="0.0" type="priority"/> <!-- def. of node "m2" -->
   <node id="m3" x="0.0" y="-250.0" type="priority"/> <!-- def. of node "m3" -->
   <node id="m4" x="0.0" y="+250.0" type="priority"/> <!-- def. of node "m4" -->
      </nodes> <!-- The closing tag -->

      编写 edg.xml
      与上面相似，exa.edg.xml :
      <edges>
   <edge id="1fi" from="1" to="m1" priority="2" numLanes="2" speed="11.11"/>
   <edge id="1si" from="m1" to="0" priority="3" numLanes="3" speed="13.89"/>
   <edge id="1o" from="0" to="1" priority="1" numLanes="1" speed="11.11"/>
   <edge id="2fi" from="2" to="m2" priority="2" numLanes="2" speed="11.11"/>
   <edge id="2si" from="m2" to="0" priority="3" numLanes="3" speed="13.89"/>
   <edge id="2o" from="0" to="2" priority="1" numLanes="1" speed="11.11"/>
   <edge id="3fi" from="3" to="m3" priority="2" numLanes="2" speed="11.11"/>
   <edge id="3si" from="m3" to="0" priority="3" numLanes="3" speed="13.89"/>
   <edge id="3o" from="0" to="3" priority="1" numLanes="1" speed="11.11"/>
   <edge id="4fi" from="4" to="m4" priority="2" numLanes="2" speed="11.11"/>
   <edge id="4si" from="m4" to="0" priority="3" numLanes="3" speed="13.89"/>
   <edge id="4o" from="0" to="4" priority="1" numLanes="1" speed="11.11"/>
      </edges>

      编写 typ.xml
      这里就不详细描述了，因为就类似于建立一个type类供edge使用。

      编写con.xml
      与上面类似，exa.con.xml :
      <connections>
   <connection from="1si" to="3o" fromLane="0" toLane="0"/>
   <connection from="1si" to="2o" fromLane="2" toLane="0"/>
   <connection from="2si" to="4o" fromLane="0" toLane="0"/>
   <connection from="2si" to="1o" fromLane="2" toLane="0"/>
      </connections>

      生成 net.xml
      使用 netconvert 生成 exa.net.xml
      netconvert --node-files=exa.nod.xml --edge-files=exa.edg.xml \  --connection-files=exa.con.xml --type-files=exa.typ.xml \  --output-file=exa.net.xml
      如果没有 con.xml 或者 typ.xml 则忽略对应的参数。
      使用sumo-gui查看net结果如下

|image2|

---------------------------

.. code:: bash

      Build Demand Model

      举个简单的例子exa.rou.xml:
      <routes>
    <vType accel="1.0" decel="5.0" id="ACar" length="2.0" maxSpeed="10.0" sigma="1.0" />
    <vType accel="0.8" decel="5.0" id="BCar" length="2.0" maxSpeed="15.0" sigma="1.0" />
    <route id="route_ns" edges="4fi 4si 3o"/>
    <route id="route_we" edges="1fi 1si 2o"/>
    <flow depart="1" id="flow_n_s" route="route_ns" type="ACar" begin="0" end="3600" probability="0.1" />
    <flow depart="1" id="flow_w_e" route="route_we" type="BCar" begin="0" end="3600" probability="0.3" />
      </routes>

      Simulation

      首先我们需要编写exa.sumocfg:
      <configuration>
    <input>
        <net-file value="exa.net.xml"/>
        <route-files value="exa.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="10000"/>
    </time>
      </configuration>

      然后在shell中run
      sumo-gui -c exa.sumocfg 

      或者直接打开 sumo-gui 选择 open simulation，打开 exa.sumocfg 文件即可。
      对于稍微复杂的情况，建议直接使用netedit软件以图形界面的方式生成net.xml道路信息文件。

参考资料
--------

-  `SUMO 官网 <https://sumo.dlr.de/docs/index.html>`__
-  `SUMO参考视频教程 <https://www.bilibili.com/video/BV1H7411F76Bfrom=search&seid=18074238600246103248>`__


.. |image1| image:: ../../img/SUMO1.png
.. |image2| image:: ../../img/SUMO2.png