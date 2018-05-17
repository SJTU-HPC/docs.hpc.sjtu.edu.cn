# How to table

## Do you really need a table?

1. Would this information be more clear if presented as a list?
2. How many dimensions (rows and columns) are really needed?
3. Would a paragraph be more clear?

## How complicated is the data you want to show?

The output of a datebase join operation is probably better displayed
as two seperate tables.

Example:
[Cori configuration](http://www.nersc.gov/users/computational-systems/cori/configuration/)

| Category            | Quantity | Description                                                                    |
|---------------------|:--------:|--------------------------------------------------------------------------------|
| HSW Cabinets        | 14       | Each cabinet has 3 chassis, each chassis has 16 blades, each blade has 4 nodes |
| HSW Nodes           | 2388     | Each node has X, Y, Z, Q, V, Q                                                 |
| KNL Nodes           | 9668     | Each node has X, Y, Z, A, B, C                                                 |
| DVS Nodes           | 32       |                                                                                |
| Lustre Router Nodes | 130      |                                                                                |
| RSIP Nodes          | 10       |                                                                                |

This table could be much more clear as multiple tables.

### Node types
| Type          | Quantity |
|---------------|:--------:|
| KNL           | 9668     |
| HSW           | 2388     |
| Login         | 12       |
| MOM           | None     |
| Share root    | 16       |
| Lustre Router | 130      |
| DVS Server    | 32       |
| RSIP          | 10       |

### System Overview
| Feature                      | Description                                          |
|------------------------------|------------------------------------------------------|
| Aries Dragonfly Interconnect | 46 TB/s global bisection bandwidth                   |
| HSW Cabinets                 | 14                                                   |
| KNL Cabinets                 | 54                                                   |
| HSW Peak Flops               | 2.29 PFlops                                          |
| KNL Peak Flops               | 29.1 PFlops                                          |
| HSW Agg. memory              | 203 TB                                               |
| KNL Agg. memory              | 1PB                                                  |
| Scratch storage              | Cray Sonexion 2000 Lustre: 30 PB, >700 GB/s          |
| Burst Buffer                 | 288 Cray DataWarp nodes: 1.9PB, >1.7 TB/s, 28M IOP/s |

### HSW node architecutre
| Feature | Quantiy  | Description                               |
|---------|:--------:|-------------------------------------------|
| sockets | 2        |                                           |
| cores   | 32       | 16 per socket                             |
| threads | 64       | 2 hyperthreads per core                   |
| DRAM    | 128 GB   | four 16 GB                                |
| L3      | 80 MB    | 40 MB per socket                          |
| L2      | 8.192 MB | 256 KB per core                           |
| L1      | 2.048 MB | 64 KB per core (32 KB iCache, 32 KB data) |

## How do make your your tables look nice in plain text?

Try an [online formatter](http://markdowntable.com) to cleanup when
you have finished entering the content.

## Inline HTML

Inline HTML is supported, but should be used sparingly as it harms the
plaintext legibility of the document.

<table>
  <tr>
    <th>Name</th>
    <th colspan="2">Telephone</th>
  </tr>
  <tr>
    <td>Bill Gates</td>
    <td>55577854</td>
    <td>55577855</td>
  </tr>
</table>

<table>
  <caption>Monthly savings</caption>
  <tr>
    <th>Month</th>
    <th>Savings</th>
  </tr>
  <tr>
    <td>January</td>
    <td>$100</td>
  </tr>
  <tr>
    <td>February</td>
    <td>$50</td>
  </tr>
</table>

## Line break in a table

| Category            | Quantity | Description                                                                            |
|---------------------|:--------:|----------------------------------------------------------------------------------------|
| HSW Cabinets        | 14       | Each cabinet has 3 chassis, <br>each chassis has 16 blades, <br>each blade has 4 nodes |
| HSW Nodes           | 2388     | Each node has X, Y, Z, Q, V, Q                                                         |
| KNL Nodes           | 9668     | Each node has X, Y, Z, A, B, C                                                         |
| DVS Nodes           | 32       |                                                                                        |
| Lustre Router Nodes | 130      |                                                                                        |
| RSIP Nodes          | 10       |                                                                                        |
