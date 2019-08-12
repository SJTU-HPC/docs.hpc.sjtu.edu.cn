## NIM用户指南                
                

NERSC信息管理系统（NERSC Information Management）简称NIM，是一个包含用户信息，登录账号，使用情况，使用额度等信息的网络入口。

### 首次登录
https
您在首次登录到NIM系统时，必须修改您的NIM密码。您的MIN密码同时也是您的NERSC密码，而且必须符合一定的条件。请看 [Passwords ](https://www.nersc.gov/users/accounts/user-accounts/passwords/)

### 登录

登录时请使用链接 https://nim.nersc.gov

请使用该链接作为收藏夹链接，否则您可能需要进行两次登录。
输入您的NERSC用户名和NIM密码。如果您遇到相关问题，请联系 账号帮助办公室，1-800-66NERSC，菜单选项 2。
同一台计算机上只能够保持一个NIM会话（session）。当您第二次登录NIM时，您将会看到以下信息：

```
Another NIM session is currently active on your computer. If you continue to log in, it will be terminated.

```

您可以选择 **继续**，这将退出之前的会话，或者 **取消** 这将保持之前的会话。

### 浏览器要求

为了使用NIM系统，您需要一个能够处理表格，框架和Javascript的浏览器。同时，您的浏览器需要使用cookies（作认证用途）。以下是免费且兼容NIM系统的浏览器：

* [Firefox](https://www.mozilla.com/)

* [Safari](https://www.apple.com/safari/)

* [Internet Explorer](https://www.microsoft.com/windows/internet-explorer/)

您必须把浏览器设置为启用Javascript和接受cookies

* Firefox用户请在菜单中点击“Preferences”
	1. 选择“Content”页面并确保启用JavaScript
	2. 选择“Privacy”按钮并确保启用"Accept cookies from sites"

* Safari用户请在菜单中点击“Preferences”
	1. 选择“Security”按钮并确保启用JavaScript
	2. 选择“Privacy”按钮并禁用来自第三方及广告的cookies

* Internet Explorer用户请选择在菜单“工具”中选择“Internet选项”
	1. 在General页面中点击浏览历史记录的设定按钮，然后找到“Internet临时文件”，选择“每次服务网页时”
	2. 在“Security”页面中选择“Medium (or less)” 作为安全等级。或者选择“自定义”并允许Java，Javascript和cookies

### 无法使用NIM？

如需初始化或重置NIM密码，请联系NERSC账号帮助办公室 (800-66-NERSC, 菜单选项 2; 或 510-486-8612; 或发电邮至 [accounts@nersc.gov](mailto:accounts@nersc.gov)。
如无法登录，请联系NERSC技术顾问 (800-66-NERSC, 菜单选项 3; 或 510-486-8611; 或发电邮至 [consult@nersc.gov](mailto:consult@nersc.gov)

## NIM菜单导航

### Quick Search/快速搜索

**Quick Search** 帮助您快速找到个人用户或项目的相关信息。

### NIM菜单栏

顶端蓝色菜单栏帮助您访问NIM系统的不同部分。**My Stuff** 下拉菜单连接到您的个人信息和其他常用功能

1. **NIM Home** 跳转到登录页面
2. **My Stuff** 连接到您的相关信息
3. **Search**下拉菜单帮助您搜索信息
4. **Reports** 下拉菜单提供使用报告
5. **Actions** 下拉菜单更改信息，如默认账户，默认shell等

**Quick Search**帮助您快速找到个人用户或项目的相关信息。

### Searching

**Search menu** 允许您选择筛选条件。筛选条件由多个行列组成。您可以点击**More** 或 **Fewer** 以增加或减少条件。使用**Submit Query** 按钮进行搜索。每一行的筛选条件由以下部分组成：

* **逻辑运算符** 下拉菜单
* **筛选项目**下拉菜单。 您可以点击**the select fields** 下拉菜单选择不同的筛选项目
* **输入框** 填入搜索目标的数值。如果您使用**List?** 请勿填入任何数值
* **List?** 按钮：用于选择筛选项目的值（并非所有筛选项目都存在预设值）
* **Sort Order** 下拉菜单
* **Hide?** 选择是否隐藏该项

对于部分搜索 **Subt?** 用于选择是否计算搜索结果的总和。**Sums only?** 用于选择是否仅显示搜索结果的总和

### 格式化显示结果

点击 **Submit Query** 按钮，搜索结果将在“显示区域”中显示。

#### Sorting

点击每一列上的箭头对搜索结果进行排序。向上的箭头代表升序，向下的箭头代表降序
第一次点击箭头，结果将以该列数值为第一排序标准。第二次点击其他列的箭头，结果将以该列数值为第二排序标准。如此类推。请注意：如果您在提交搜索条件时选择了排序方法，使用箭头可能会令您提交的排序方法失效

您无法取消已经选择的排序方法。如需改变第一排序标准，第二排序标准等，您必须重新搜索。

#### 格式选项

搜索结果默认以HTML表格显示。根据显示情况，其他格式或可使用：

* **Static HTML**或 **Plain text**：显示不含任何可点击链接的结果
* **Select Columns**：选择其他要显示的列（您可以在弹出窗口中选择添加或删除列）

## 用户账号信息

首次登录NIM，项目用户将处于 **Account Usage** 显示区域。该区域由六个标签页组成。项目用户可以看到 **Account Usage** 标签页；PI和项目管理者可以看到 **Roles** 标签页。您可以通过点击，在不同的标签页之间切换

### Account Usage 标签页

#### 如何找到：

1. 点击 **My Stuff** 菜单；选择 **My Account Usage**
2. 在NIM页面顶端的 **Quick Search** 中选择 **Login Name** 输入登录用户名，点击 **Go**

“Account Usage Summary” 显示用户的MPP和HPSS用量和设定，以及用户所属项目的使用情况。 **Current HPSS Usage** 显示在本年度中，您储存在HPSS中的文件数量，以及所占空间的大小（以gigabytes为单位）和上传下载的总量（以gigabytes为单位）

### Logins by Host 标签页

**Login Names by Host** 显示您在各个系统中的状态，默认shell，默认Unix组和默认结算仓库

### Unix Groups 标签页

**Unix Groups** 显示您在各个系统中的所属Unix组

### Roles 标签页

**Roles** 显示您在各个项目中的角色（如用户，PI，代理PI）

### Contact Info 标签页

#### 如何找到：

点击**My Stuff** 菜单；选择 **My Contact Info**显示您的电子邮件地址，工作电话，工作地点。如需更新相关联系方式，请点击左上方的 **Update** 链接。为了能够和您及时联系，保持更新您的联系方式对我们十分重要

### Grid Certificates 标签页

注册您的grid certificate的地方，详见 [Grid Certificates](https://www.nersc.gov/users/software/grid/certificates/)

## 使用NIM管理您的账号

本章节介绍如何个性化定制您的用户账号，以满足您的需求和保持更新

### 回答您的安全问题

#### 如何找到：

在NIM主菜单中点击 **Actions** 下拉菜单；选择 **Set Security Questions**
即使您忘记了密码，又或者密码过期（密码有效期为六个月），只要正确回答安全问题，您就可以重置NIM密码

### 自助修改密码步骤

如需使用自助修改密码服务，请在NIM登录页面点击"重置密码"，然后输入您的NERSC用户名，以及安全认证图片中的字母，并点击“请求发送验证码”。NERSC将把验证码发送到您登记的电子邮箱。输入验证码并点击“检查验证码”。系统将要求您回答部分安全问题。如果回答正确，您将可以输入一个新的NIM（NERSC）密码
#### 忘记了NERSC用户名？

如果您忘记了您的用户名，请点击NIM登录页面的"忘记用户名？"，然后输入您登记的电子邮箱（必须与我们记录中的电子邮箱一致），以及安全认证图片中的字母，并点击“获取用户名”。NERSC将把您的用户名发送到您的电子邮箱

### 更改您的NERSC密码

#### 如何找到：

1. 在NIM页面左上方处选择 **Change My Password**
2. 点击**Actions** 菜单；选择 **Change NIM Password**

NIM (或者 NERSC) 密码适用于所有需要密码验证的NERSC服务。NIM密码必须遵守能源部的相关政策。详见[Passwords](https://www.nersc.gov/users/accounts/user-accounts/passwords/)

### 更改您的默认登录shell

#### 如何找到：

1. 点击**Actions** 菜单；选择 **Change Shell**
2. 您将看到一个包含主机和对应默认登录shell的表格。点击相应的 **Change Shell** 则可以更改您的默认登录shell。点击 **Save** 以保存更改

### 更改您的默认仓库（使用正确的账号进行MPP结算）

NERSC的计算机时间被划分到不同的项目账号，也称为仓库（repos）。每个用户都有一个默认的仓库。所有非互动性的结算都在默认仓库进行。如果Batch jobs没有指定仓库，其结算也在默认仓库进行。
您可以在 Account Usage Summary 页面找到您的默认仓库。对大多数用户来说，Account Usage Summary 页面就是登录时的初始页面。您也可以从 My Stuff 菜单中选择 **My Account Usage** 找到该页面。请注意：

* **Dflt Now?** Y 表示默认仓库已生效
* **Base Repo?** Y 表示该仓库已经被选为您的默认仓库。除非您在"base default repo"中的余额用尽，否则该仓库将一直作为您的默认仓库。如果余额用尽，NIM会选择另一仓库（如果有的话）作为您的默认临时仓库。

#### 在哪里改变您的默认仓库：

* 点击 **Actions** 菜单，选择 **Change Default MPP Repo**
* 从 **Change Default Repo** 下拉菜单中选择您要的仓库，点击 **Save** 按钮
* 
### 更改您的"SRU project percents"（使用正确的账号结算HPSS使用量）

HPSS储存系统追踪每个用户的使用量，但HPSS并不知道用户所属的仓库。HPSS每天一次把用户的使用量信息传送到NIM。NIM依据"SRU project percents"，把使用量划分到用户的仓库。
您可以在NIM的Account Usage page页面找到您的project percents。对大多数用户来说，Account Usage Summary 页面就是登录时的初始页面。您也可以从 My Stuff 菜单中选择 **My Account Usage** 找到该页面。请注意：**Proj %** 指的是您的HPSS仓库

#### 在哪里更改您的SRU project percents:

点击 **Actions** 菜单，选择 **Change SRU Proj Pct**
输入每个项目的百分比，然后点**Save**按钮

### 更新您的联系信息

为了和用户更有效率地进行沟通，以及遵守能源部的相关计算机使用规定，我们要求用户提供以下信息：名字，电邮地址，工作电话，所属机构组织。请您保持这些个人信息有效，以便联系。

#### 怎么找到：

点击**My Stuff** 菜单，选择 **My Contact Info**
点击 **Update** 链接。输入新信息后，请点击页面底部的 **Submit** 按钮

### 项目与仓库信息

#### 怎么找到：

从**Quick Search**下拉菜单中选择 **Repository**，输入仓库名并点击 **Go**

**Project**显示区域包含六个页面。您可以通过点击它们来切换。NERSC的项目和其allocation request相联系。当一个项目获得资源，相关资源数量会被存入项目的“银行账号”。NERSC把这种“银行账号”称为仓库（repo）

### Project Information页面

该页面显示项目概况，如项目的PI和PI代理，标题，以及支持该项目的能源部科学办公室。项目的标题连接到该项目的allocation request。MPP和HPSS资源量和使用量概况也在此页面显示

### User Roles & Contact Info页面

该页面显示项目用户的联系方式。PI和项目管理者可以在此更改用户的角色。详见 [NIM Guide for PIs](https://www.nersc.gov/users/accounts/nim/nim-guide-for-pis/)

### User Status by Repo页面

该页面显示用户在不同资源的结算概况

### MPP Usage & Quotas页面

该页面显示用户在项目MPP仓库的使用量和账号设置。PI和项目管理者可以在此更改用户的限额。详见 [关于如何结算](https://www.nersc.gov/users/accounts/nim/nim-guide-for-pis/)。 更多关于MPP结算的详情请见[关于如何结算](https://www.nersc.gov/users/accounts/user-accounts/how-usage-is-charged/)

### HPSS Usage & Quotas页面

该页面显示用户在HPSS仓库的使用量和账号设置。PI和项目管理者可以在此更改用户的配额。详见 [Guide for PIs](https://www.nersc.gov/users/accounts/nim/nim-guide-for-pis/) 和 [HPSS结算相关信息](https://www.nersc.gov/users/storage-and-file-systems/hpss/hpss-charging-old/)

### Transfer History页面

该页面显示您项目仓库的所有存取记录
