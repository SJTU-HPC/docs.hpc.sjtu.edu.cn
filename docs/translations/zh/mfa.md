# 多重身份验证 Multi-Factor Authentication (MFA)

## 简介

2019年开始，NERSC用户必须使用多重身份验证，Multi-Factor Authentication (简称MFA)。比起通常的密码身份验证，MFA提供更好的保护，帮助用户防范钓鱼和其他现代网络安全威胁。通过NERSC的MFA，您可以使用NIM密码加上一个一次性密码，"one-time password" (简称OTP)进行身份验证。正如其名，该密码仅一次性有效。
您可以找到更多关于 [设置](https://www.nersc.gov/users/connecting-to-nersc/mfa/#configuringMFA")和[使用MFA](https://www.nersc.gov/users/connecting-to-nersc/mfa/#usingMFAwithSSH)的指引
目前NERSC系统的ssh登录和网页都支持MFA验证。在未来的数月，我们将应用新技术提升MFA的使用体验，并把MFA验证推广到大部分NERSC资源。
我们以对用户造成最小影响的方式推广MFA。请阅读以下内容以了解MFA的新特性，以及如何安装使用MFA。

### 新特性: MFA for NoMachine (NX)

MFA现已支持NX。更多详情见[MFA for NoMachine (NX)](https://www.nersc.gov/users/connecting-to-nersc/mfa/#nx)

### NERSC的MFA是怎么工作的

NERSC的MFA要求用户在其移动设备上安装一个验证应用（app）。您可以通过[NIM](https://nim.nersc.gov/)进行设置（如果您没有iOS或安卓移动设备，请参见[alternatives](https://www.nersc.gov/users/connecting-to-nersc/mfa/#authenticatorApps)。验证应用会显示一个每30秒改变一次的六位数密码。每个密码仅一次性有效，因此又被称为一次性密码（"one time password" 或 "OTP"）。当您登录到要求进行MFA验证的系统时，您需要输入您的NIM密码和您的一次性密码。

![MFA和密码登录](https://www.nersc.gov/assets/Uploads/_resampled/ResizedImage600219-Screen-Shot-2017-05-05-at-12.03.01-PM.png)

### 设置和使用MFA令牌（Token）

一次性密码入口在验证应用里面有时候被称为令牌，或者更具体来说，软令牌（soft token）。为了使用MFA，您需要创建一个NERSC的令牌并安装到验证应用上。

设置NERSC令牌的步骤：

* 安装验证应用
* 在NIM账号中启用MFA
* 用NIM生成NERSC的令牌
* 把令牌安装到验证应用上

### 安装验证应用

NERSC支持Google的验证应用（Google Authenticator），兼容安卓和iOS设备。 其他的TOTP (Time-based OTP) 验证应用也可以使用，您可以检索"TOTP"以寻找更多选项。如果您没有以上设备，NERSC也支持Authy，一款兼容Windows和Mac设备的桌面应用。本说明侧重于Google验证应用，Authy相关说明见 [further below](https://www.nersc.gov/users/connecting-to-nersc/mfa/#usingAuthy).
要安装Google验证应用，安卓用户 [下载地址](https://play.google.com/store/apps/details?id=com.google.android.apps.authenticator2&amp;hl=en_US"); 苹果用户 [下载地址](https://itunes.apple.com/us/app/google-authenticator/id388497605?mt=8)。
请注意：您 __不需要__ 移动信号或者WiFi信号以使用Google验证应用。 验证应用根据您的设备的内部时钟生成一次性密码。一旦设置完成，您无需电话或互联网服务也可使用验证功能。

### 在NIM中启用MFA

生成令牌之前，您需要在NIM账号中选择“MFA Opt In”，请使用您的账号和密码登录 
[nim](https://nim.nersc.gov)。
请注意，如果您选择了“MFA Opt In”，但没有生成令牌，您依然可以仅使用NIM密码登录到您的NIM账号（无需使用一次性密码）。
要启用MFA，请在黄色的横向菜单栏中点击 **MFA Token**。或者在 **Actions** 下拉菜单中选择 **MFA Enable/Disable**

![启用MFA](https://www.nersc.gov/assets/Uploads/nim-mfa-opt-in-3.png)

选择 **Enabled** 并点击 **Save All Rows** 按钮

![选择Enabled](https://www.nersc.gov/assets/Uploads/nim-mfa-enabling-5.png)

点击上面的 **Contact Information**，您可以看到页面显示MFA已经启用：

![显示MFA已经启用](https://www.nersc.gov/assets/Uploads/mfa-enabled-nim-contact-info.png)

**注意**：一旦启用MFA，如果没有MFA令牌，您将无法使用NERSC的资源和服务。如果您没有设置MFA令牌（见下文）或删除了您所有的MFA令牌，您将在NIM网页上看到以下警告信息：

![警告信息](https://www.nersc.gov/assets/Uploads/nim-warning-no-mfa-tokens.png)

### 生成和安装令牌

在这一步，您将把您移动设备上的验证应用和您的NIM账号连接到一起。首先在NIM系统中创建一个令牌，然后把它安装到应用上。
具体方法：回到前一页（MFA Opt In/Out），点击 **Enter and manage MFA Tokens** 按钮。您将看到 **MFA Tokens** 标签页

![MFA令牌标签页](https://www.nersc.gov/assets/Uploads/nim-mfa-enabling05.png)

要创建令牌，点击 **Add Token** 按钮，然后在 **Select User** 区域看到您的NERSC账号名字。您需要先确认您的用户名和显示的一致（部分用户拥有旧账号），然后您可以在 **Enter Token Description** 中增加描述信息。

![创建令牌](https://www.nersc.gov/assets/Uploads/nim-mfa-enabling06.png)

在上面的例子中（my phone），用户尝试生成一个可供手机Google验证应用使用的令牌。如果您拥有多个令牌并安装到不同的设备上，您可以添加不同的描述信息加以区别。
点击 **Submit** 按钮，然后网页将显示一个令牌和一个QR二维码。请注意下面例子中的令牌id：TOTP22856A68

![令牌id](https://www.nersc.gov/assets/Uploads/nim-mfa-enabling09.png)

#### 使用Google验证应用（Google Authenticator）

如果您需要在安卓设备上使用Google验证，请打开您设备上的Google验证应用并点击右下方的红色“+”按钮，然后选择“Scan a barcode”

![安卓设备](https://www.nersc.gov/assets/Uploads/googleauthenticator1bc.png)

在苹果设备上，寻找右上角的“+”：

![苹果设备](https://www.nersc.gov/assets/Uploads/googleauthenticator1bi.png)

如果需要，请允许应用使用您的摄像头并扫描二维码。
如果您不想用摄像头扫描二维码，选择“Enter a provided key”并输入MFA令牌中“secret”的对应值。在“Account name”中填入MFA令牌的TOTP数字（例如NERSC-TOTP22856A68），然后选择“Time based”。扫描二维码或手动输入令牌密码后，验证应用会显示新的令牌。这样您的移动设备就和您的NIM账号连接到一起了。
当您在您的设备上添加新的令牌时，令牌的名字将以“NERSC-__登录id__-__令牌id__“ （例如NERSC-wyang-TOTP22856A68）的形式，出现在您设备左端的令牌列表里面。

#### 使用Authy

[Authy](https://authy.com/) 是一个功能上类似于Google验证的桌面应用。如果您首次使用它，您需要注册一个Authy账号并提供电话号码和电邮地址，然后输入收到验证码完成注册。
点击“+”添加账号。您将看到提示"Enter Code given by the website"，输入在NIM生成令牌时“Authy Web Code”的值（请看上面被黑色块遮盖的二维码图片，“Authy Web Code”就在图片底部。给这个令牌选择一个名字（例如：NERSC-TOTP38776DC3），要在应用中显示的颜色（例如：Generic Black），以及令牌的长度（您必须选择“6-digit”）并点击“Save”。

![桌面应用Authy](https://www.nersc.gov/assets/Uploads/authy4-5.png)

### 测试您的新令牌

要测试新令牌是否设置正确，在NIM里点击该令牌。这将显示其详细信息，以及 **Test** 和 **Delete** 按钮 。您将看到如下信息：

![令牌详细信息](https://www.nersc.gov/assets/Uploads/nim-mfa-testing01.png)

在 **Enter One-Time-Password** 中输入您的验证应用生成的一次性密码（请勿输入您的NIM密码），然后点击 **Test Now**。如果一切设置妥当，NIM页面会显示成功：

![测试新令牌](https://www.nersc.gov/assets/Uploads/nim-mfa-testing02.png)

### 多个令牌

如果您有多台移动设备，您可以给每台设备添加一个令牌。例如，您可以给手机和平板电脑创建各自的令牌。如下图：

![添加多个令牌](https://www.nersc.gov/assets/Uploads/nim-mfa-tokens.png)

您可以拥有最多4个令牌。
当您登录到NERSC资源时，您可以使用__**任意**__令牌进行认证。您无需指定某一特定的令牌，因为服务器会检查您所有的令牌。

### 更改和管理您的令牌

您可以在**MFA Tokens**标签页里看到您所创建的所有令牌（如上图）。

### 删除令牌

如果您不再需要一个令牌，您可以删除它。在NIM中选择您想删除的令牌，点击 **Delete** 按钮

### 后备一次性密码

如果您丢失了您的设备，又或者它不在您的身边，您将无法登录。后备密码是一组应对该问题的一次性密码。
点击页面底部“Generate backup passwords”旁边的“Generate!”：

![后备一次性密码](https://www.nersc.gov/assets/Uploads/_resampled/ResizedImage600167-nim-backup-otps-2.png)

请把这些密码打印或保存到文档并保管在安全的地方。当您需要使用后备密码时，选择首个未使用密码并做好标记，以方便下一次使用。请注意：您必须按照NIM提供密码的顺序使用它们。
请注意：如果您生成一组新的后备密码，那么之前的后备密码将会失效。

### 使用MFA登录NIM

用户启用MFA后需要使用MFA登录NIM系统。登录页面见下图：

![登录页面](https://www.nersc.gov/assets/Uploads/_resampled/ResizedImage600273-nim-login-mfa-staff2.png)

### 丢失令牌（重新设置MFA）

如果您永久性丢失了您的MFA令牌（例如更换了手机）您可以删除现有的令牌，然后登陆到NIM创建新令牌。为此，请点击 **Lost your tokens?** “丢失令牌？”的链接（如上图）。我们将通过电子邮件给您发送一个链接。点击链接跳转到新页面后，您需要确认是否删除令牌。确认后，您将可以仅凭密码登录NIM，然后创建新的令牌。

### 使用MFA进行SSH登录

使用MFA进行SSH登录的最简单方法就是直接ssh到一个NERSC系统。当您ssh到一台NERSC主机时，您将看到提示输入“Password + OTP”：

```
$ ssh cori.nersc.gov
 *****************************************************************
 *                                                               *
 *                      NOTICE TO USERS                          *
 *                      ---------------                          *
...
Password + OTP:
```


打开您的验证应用，获取与您的令牌相对应的一次性密码（例如上面例子中的“NERSC-wyang-TOTP22856A68”）：

![获取一次性密码](https://www.nersc.gov/assets/Uploads/googleauthenticator4b.png)

马上在"Password + OTP: "的提示后输入您的**NIM密码以及一次性密码**, 所有密码都在同一行。例如您的NIM密码是`iL0ve_Burrit0$`，而您的验证应用显示的一次性密码是“015 691”，如下图所示，您必须输入`iL0ve_Burrit0$015691`（请忽略验证应用中显示的空格）。
**Authy用户请注意： Authy 显示的第一个一次性密码经常失效，因而导致登录失败。为了获取正确的一次性密码，请点击应用左上方的后退按钮，然后重新点击令牌。我们已经把这一情况反映给Authy。**
请注意：一旦启用MFA，您在NIM中注册的ssh密钥（ssh key）将会失效。您必须在每次登录时进行MFA验证。请阅读下一章关于ssh代理（sshproxy）的内容以了解如何使用MFA（限每天一次），获取密钥用于自动化工作流程。

### ssh代理（sshproxy）

NERSC开发了一项**ssh代理（sshproxy）**服务。它允许您使用MFA获取一个有时间限制的ssh密钥（默认24小时）。ssh代理提供一种单点登录(single-sign-on)NERSC系统的服务。一旦获取密钥，您可以使用它ssh到NERSC的各个系统（如Cori，DTN等) 直到密钥失效。
ssh代理服务使用RESTful API获取密钥。NERSC提供一个可以在类Unix系统命令行上运行的bash客户端脚本，以及一个python脚本。而支持PuTTY的Windows客户端也将快上线。

#### 安装客户端

在类Unix平台（包括macOS）上，您可以直接在项目目录（project directory）中直接下载bash客户端sshproxy.sh：

```
$ scp __myusername__@cori.nersc.gov:/project/projectdirs/mfa/NERSC-MFA/sshproxy.sh .
```

请把 __myusername__ 替换成您的NERSC用户名
或者您可以在您的计算机上运行以下命令，从NERSC的Github仓库中下载bash客户端脚本，并设置正确的文件权限：

```
$ curl -O https://raw.githubusercontent.com/NERSC/NERSC-MFA/master/sshproxy.sh 
$ chmod u+x sshproxy.sh
```
另外，您可以克隆NERSC的仓库到本地计算机：

```
$ git clone https://github.com/NERSC/NERSC-MFA.git
```

上面的Git命令会在您的当前工作目录下创建一个名为NERSC-MFA的文件夹。您可以在里面找到该脚本。您可以保留着这个Git仓库，然后不时使用“git pull”命令进行更新，以确保拥有NERSC修改过的最新版本。

#### 使用ssh代理

ssh代理客户端默认使用您的用户名，其获取的ssh密钥也只有24小时的有效期。私钥和公钥分别被命名为“nersc”和“nersc-cert.pub”，并保存在 ~/.ssh 目录下。
在您安装的目录下运行sshproxy.sh脚本。例如脚本被安装到当前目录，输入：

```
$ ./sshproxy.sh
```

和您ssh到NERSC系统时一样，该脚本会提示您输入NIM密码和一次性密码：

```
Enter your password+OTP: 
```
一并输入您的NIM密码和一次性密码。验证成功后，客户端会安装ssh密钥并显示在您__本地计算机__上安装的路径，以及有效期。密钥默认的路径和文件名分别是“~/.ssh/nersc”和“~/.ssh/nersc-cert.pub”。您可以使用命令行参数改变它们的名字。

```
$ ./sshproxy.sh
Enter your password+OTP: 
Successfully obtained ssh key /Users/wyang/.ssh/nersc
Key /Users/wyang/.ssh/nersc is valid: from 2018-08-30T12:24:00 to 2018-08-31T12:25:52
```
您可以看到下面的三个ssh密钥文件（包括私钥，公钥和包含公钥的证书文件）被安装到了 ~/.ssh 目录：

```
$ ls -l ~/.ssh/nersc*
-rw-------  1 wyang  wyang  3179 Aug 30 12:25 /Users/wyang/.ssh/nersc
-rw-------  1 wyang  wyang  1501 Aug 30 12:25 /Users/wyang/.ssh/nersc-cert.pub
-rw-------  1 wyang  wyang  1501 Aug 30 12:25 /Users/wyang/.ssh/nersc.pub
```

上面的例子显示，一对ssh密钥安装到本地计算机上。您可以利用它们ssh到NERSC的各系统，无需进行更进一步的身份验证，直到密钥有效期过期。

#### 检查证实有效期

您可以检查现有ssh密钥的过期时间。如果ssh密钥的证书文件是 ~/.ssh/nersc-cert.pub，请在您的__本地__计算机上运行下面的命令：

```
$ ssh-keygen -L -f ~/.ssh/nersc-cert.pub | grep Valid
        Valid: from 2018-08-30T12:24:00 to 2018-08-31T12:25:52
```

请注意以上显示的时间是您的当地时间，而不是NERSC所在的太平洋时间。

#### 使用ssh代理密钥

您可以使用从ssh代理获得的密钥，通过命令行指定密钥路径，登录NERSC系统。例如，使用名为“nersc”的密钥登录cori.nersc.gov时：

```
$ ssh -i ~/.ssh/nersc cori.nersc.gov
```

这将允许您无需再次验证身份就可以登录。

#### ssh代理命令行参数

您可以使用几个参数改变sshproxy.sh的默认行为。请运行“sshproxy.sh -h”获取帮助信息。

```
$ ./sshproxy.sh -h
Usage: sshproxy.sh [-u &lt;user&gt;] [-s &lt;scope&gt;] [-o &lt;filename&gt;] [-U &lt;server URL&gt;]
         -u &lt;user&gt;	Specify remote username (default: &lt;your_login_name&gt;)
         -o &lt;filename&gt;  Specify pathname for private key (default: /Users/&lt;your_login_name&gt;/.ssh/nersc)
         -s &lt;scope&gt;     Specify scope (default: 'default')
         -a             Add key to ssh-agent (with expiration)
         -U &lt;URL&gt;       Specify alternate URL for sshproxy server (generally only used for testing purposes)

```

如果您的NERSC用户名和您在本地计算机上的用户名不一样，您可以使用“-u”指定您的用户名：

```
$ ./sshproxy.sh -u __myusername__
```

如果您想用其他名字命名ssh密钥文件，您可以使用“-o”指定文件名：

```
$ ./sshproxy.sh -o mynersc
```

注意“-a”选项可以用作自动添加新的密钥到您的ssh-agent。它将给密钥设置一样的有效期，所以当密钥过期后，ssh-agent将不再尝试使用它。
如果您计算机上使用的是旧版本的ssh（例如OpenSSH_7.2），您可能可以使用“-a”选项。否则，ssh和scp命令会要求额外的选项。请见接下来的的例子。要查看版本信息，运行“ssh -V”。

#### 长期有效的SSH密钥

使用"-s"选项满足您工作中的特别需要。如何您的自动化工作需要长期使用密钥，您可以提出申请。点击 [这里](https://nersc.service-now.com/com.glideapp.servicecatalog_cat_item_view.do?v=1&amp;sysparm_id=85f7c6dfdb5463407cf774608c9619fa&amp;sysparm_link_parent=e15706fc0a0a0aa7007fc21e1ab70c2f&amp;sysparm_catalog=e0d08b13c3330100c8b837659bba8fb4&amp;sysparm_catalog_view=catalog_default) 提交申请。我们需要进行审核才可以发放长期有效的ssh密钥。一旦申请获批，我们将提示您如果设置。

#### SSH配置文件选项

我们建议您在ssh配置文件中加入部分选项。这些选项可以避免某些因ssh密钥过期而产生的问题，也可以自定义密钥的默认文件名从而省去输入的麻烦。注意：这些选项可以被命令行选项覆盖。
如果您通常仅仅使用ssh代理默认的“nersc”密钥，您可以在配置文件中指定该密钥，无需每次都在命令行里输入。为此，请编辑__本地计算机__上的 **~/.ssh/config** 文件，加入以下这行：

```
Host cori*.nersc.gov gpweb*.nersc.gov dtn*.nersc.gov
    IdentityFile ~/.ssh/nersc
```

有了这一行命令，无论您什么时候ssh到NERSC的系统，您的ssh客户端都会自动使用您的ssh代理密钥。
如果您的ssh密钥没有向ssh服务器提交正确的ssh密钥，服务器将依然提示您输入NIM密码和一次性密码。ssh服务器和客户端都不会提示您ssh密钥过期。

#### 登录到NERSC计算机

如果您按上述方法设置好ssh密钥，只要密钥没有过期，您登录到NERSC计算机时将无需进行进一步验证：

```
$ ssh cori.nersc.gov
 *****************************************************************
 *                                                               *
 *                      NOTICE TO USERS                          *
 *                      ---------------                          *
...
$                    # You're on cori
```

登录以后，您可以像在任何login node上一样，构建自己的代码，提交batch job，调试自己的代码等等。
您可以使用scp给NERSC系统上传或下载文件：

```
$ scp myfile cori.nersc.gov:~
 *****************************************************************
 *                                                               *
 *                      NOTICE TO USERS                          *
 *                      ---------------                          *
...
myfile                                        100%   13     0.5KB/s   00:00
```

您同样不会看到提示验证身份。

### 当您使用过期密钥登录

如果您使用过期密钥登录，服务器不会告诉您密钥已经过期，而是提示您使用MFA登录，就像您没有使用密钥的时候一样：

```
$ ssh cori.nersc.gov
 *****************************************************************
 *                                                               *
 *                      NOTICE TO USERS                          *
 *                      ---------------                          *
...
Password + OTP: 
```

您可以在任何时候运行sshproxy.sh脚本，生成新的ssh密钥。

### 主机认证（Host Based Authentication）

依照设置，NERSC高性能计算机在Cori和NX系统之间的ssh登录使用主机认证（Host Based Authentication）的方式进行身份验证。这意味着，一旦您远程登录到上面提到的任一主机，您就可以无需再次验证或使用ssh代理登录到其他NERSC主机。

#### 在NoMachine(NX)上使用MFA

当您登录到NX时，请一并输入您的NIM密码和六位数一次性密码。一旦成功登陆NX，再登录到Cori时就无需再进行验证。

#### 在MyProxy上使用MFA

NERSC的MyProxy服务将要求激活MFA的用户使用NIM密码和一次性密码进行身份验证。

#### 在网页服务（Web Services）上使用MFA

大部分NERSC网站用户使用以下其中一种服务进行身份验证：Shibboleth和NEWT。它们都提供单点登录(single-sign-on)服务。也就是说，只要您在其中一个网站通过身份验证，您就可以在24小时以内登录其他网站而无需再次验证。Shibboleth和NEWT都需要启用MFA的用户输入其NIM密码和一次性密码。
使用Shibboleth服务的网站会如下图一样，显示带有NERSC标志的登录界面。请使用您的NIM用户名和密码登录。

![使用NIM用户名和密码登录](https://www.nersc.gov/assets/Uploads/nersclogin.png)

然后将提示您输入一次新密码：

![输入一次密码](https://www.nersc.gov/assets/Uploads/shib-login02b.png)

使用NEWT服务的网站显示的登录界面会有所不同。请见下图：

![NEWT服务的登录界面](https://www.nersc.gov/assets/Uploads/mynersc-login-mfa.png)

部分NERSC网站因为技术原因并不使用Shibboleth和NEWT。对于那些网站，单点登录不会生效，您登录每个网站时都需要进行MFA身份验证。如下图，使用NIM密码和一次性密码进行登录：

![使用NIM用户名和密码登录](https://www.nersc.gov/assets/Uploads/jupyter-login01b.png)

NIM用户入口也将要求启用MFA的用户使用MFA进行登录

## MFA在各用户系统的状态

目前大多数可供用户通过ssh访问的系统都支持MFA，例如Cori。网站和其他服务从2018年9月开始支持MFA。下面的表格显示NERSC各系统和服务对MFA的支持情况。

### 现已支持

| 身份验证        | 主机           |
| ------------- |:-------------:|
| SSH | Cori |
| SSH |Data Transfer Nodes|
| SSH | gpweb |
| SSH | gpdb |
| Shibboleth |Online Help Desk[(https://help.nersc.gov)](https://help.nersc.gov)|
| Shibboleth |[Science gateways](https://portal.nersc.gov/) 带NERSC (Shibboleth)登录标志|
| NEWT | [My NERSC](https://my.nersc.gov) |
| NEWT | [Science gateways](https://portal.nersc.gov/)接受NIM密码而不显示NERSC (Shibboleth) 登录标志|
| MyProxy | - |
| Others | [NIM](https://nim.nersc.gov/) |
| Others | NX和[NX-cloud](https://nxcloud01.nersc.gov/) |
| Others | [Jupyter](https://jupyter.nersc.gov) |
| Others | [RStudio](https://rstudio.nersc.gov/) |
| Others | [HPSS token generation](../../../filesystems/archive/#automatic-token-generation)|

### 即将支持

| 身份验证        | 主机           |
| ------------- |:-------------:|
| TBD | [Shifter Registry](https://registry.services.nersc.gov/) |
| TBD | [Spin Registry](https://registry.spin.nersc.gov/)|

### 不支持

| 主机|
| ------------- |
| 其他运行在[portal.nersc.gov](https://portal.nersc.gov/)上的science gateways | 
| 	GRDC | 
| WeFold | 
| CRCNS | 
| The Materials Project | 
| QCD | 

### 常见问题（FAQ）

**（问）我没有智能手机或平板电脑，怎么办？**

您可以使用一个名叫 [Authy](https://authy.com/download) 的桌面应用。它兼容Windows和Mac的计算机。
您也可以使用Authy的Chrome浏览器插件。而Firefox浏览器的验证应用插件是[FoxAuth](https://addons.mozilla.org/en-US/firefox/addon/foxauth/)。其他浏览器的TOTP验证应用插件同样有效。
考虑到安全因素，我们建议您不要在日常用于连接NERSC系统的机器上安装MFA验证应用和浏览器插件，请选择在其他设备上安装。
我们正考虑提供一个使用硬件令牌登录的方案（用户需要自行购买硬件令牌）。我们将在方案可行时提供更多信息。

**（问）我设备上的时钟会出现偏差，特别是当我在海外旅行而手机又连接不上网络的时候。我依然可以使用它生成一次性密码吗？NERSC一次新密码服务器和客户端进行时间同步的要求是什么？**

只要时间差不超过180秒，NERSC服务器会根据您设备的时间偏差进行调整。如果时间差超过180秒，您进行MFA验证时就会出现问题。大多数的情况是，您的手机时间会出现短暂偏差，手机接上网络后，它会和信号塔同步，然后手机时间将大幅变动。在这种情况下，我们的解决办法是删除现有的令牌并创建新令牌。

**（问）我丢失了设备，怎么办？**

如果您在另一设备上设置好了MFA验证应用，您应该尽快登录到NIM并删除所有和丢失设备有关的MFA令牌。
如果您没有其他设置好MFA验证应用的设备，那么请到NIM的登录界面，[https://nim.nersc.gov](https://nim.nersc.gov/)，点击“丢失令牌？”的链接，获取验证码登录您的NIM账号。登录后，请删除所有令牌并创建新令牌。如果您还没有新设备，您可以生成备用一次性密码，使用它们登录。

**（问）我有两台设备。怎么可以把MFA令牌从一台复制或转移到另一台？**

您不能够复制或转移令牌。但当NIM生成一个QR二维码和“secret”码时（请见生成和安装令牌的章节），您可以使用同样的QR二维码和“secret”码在不同的设备上创建令牌。如果所有设备的时钟都同步运行且时间一致，那么所有设备上的验证应用都会显示一样的一次性密码。

**（问）我启用了MFA，但登录不断出错，怎么办？**

如果这仅仅发生在某一特定主机上（例如Cori），那么请通过该链接 [https://nim.nersc.gov](https://nim.nersc.gov/) 登录到您的NIM账户。这将清除累计的登录失败次数。然后再尝试登录到之前有问题的主机。
如果您多次输入错误的一次性密码，NERSC的MFA服务器将拒绝您的请求。在这种情况下，您需要等待15分钟才可以再次使用MFA服务。
如果您使用sshproxy.sh脚本生成的ssh密钥进行身份验证，请检查密钥是否过期。
使用ssh密钥进行身份验证的一个常用方法是使用ssh-agent（authentication agent）。把ssh私钥加入到ssh-agent，它会在远程服务器上找到相应的公钥进行身份验证。您可能在有意无意中使用这种方法登录（特别是当您在运行sshproxy.sh脚本时加上“-a”选项的时候）。Ssh-agent逐一检查保存的私钥，如果它尝试6次也找不到对应的密钥，ssh认证失败。当您在ssh-agent里储存太多密钥时，即便里面包含正确的密钥，只要它没有被用于最开始的6次验证中，验证都会失败。要知道ssh-agent中保存了多少密钥，请在本地计算机上运行“ssh-add -l”。如果您保存了很多密钥，您可以运行命令“ssh-add -D”删除它们，然后再次运行sshproxy.sh脚本。您也可以用“-d”选项选择性地删除密钥（更多说明请见ssh-add的man page）。
如果您忘记了密码，请依照 [https://www.nersc.gov/users/accounts/user-accounts/passwords/](https://www.nersc.gov/users/accounts/user-accounts/passwords) 中的“忘记密码”（Forgotten Passwords）章节进行操作。
如果您所有的MFA令牌都无效，请点击[NIM登录页面](https://nim.nersc.gov)的“丢失令牌？”链接获取用于登录NIM系统的验证码。如果您的账户启用了MFA，但还没有在设备上设置好验证应用就退出了NIM，您同样应该按照“丢失令牌？”操作。您会通过电子邮件获得一个验证码。请注意：用验证码登录NIM系统以后，所有MFA令牌将被自动删除。一旦登录，请创建新的令牌。

**（问）我登录到NERSC的一项服务时，使用了一个验证应用上显示的一次性密码。我需要再次登陆到另一NERSC服务，我可以在30秒以内使用同一个一次性密码吗？**

不可以，因为一次性密码仅一次性有效。您必须等待下一个30秒时间窗口以获取新的一次性密码。

**(问）我们正在NERSC的机器上运行一些自动化程序。我们要怎么样才能够使用MFA继续我们的工作？**

请使用ssh代理服务。目前它的ssh密钥只有24小时的有效期。如果需要更长的有效期，请填写申请表格。

**（问）哪些服务支持MFA？**

请查看上面“MFA在各用户系统的状态”一章中的表格。

**（问）我希望使用的NERSC资源还未支持MFA。这是否意味着如果我启用MFA，我将无法使用该资源？**

不。这仅仅意味着您仅凭NIM密码就可以使用该资源。

**（问）NERSC的所有主机是否都支持ssh主机认证（host based authentication）？**

目前ssh主机认证仅被Cori和NX支持。从NX上，您可以无需经过再次验证就通过ssh登陆到Cori。然而当您从Cori，ssh到其他的主机（例如dtn01），您需要进行身份验证。反之亦然。我们正努力把主机认证功能扩展到其他系统。目前，您可以通过ssh代理获取ssh密钥，在不同主机之间跳转。

**（问）有没有Windows平台的ssh代理客户端？**

我们正在开发和测试新工具，详情将很快公布。
另外，如果您拥有 [Cygwin](https://www.cygwin.com/) ，您可以在Cygwin终端上使用Linux版本的ssh代理。请确保您的Cygwin包含curl和openssh的依赖包。

**（问）我怎么样才可以使用类似BBEdit，FileZilla，WinSCP之类的工具？它们都需要通过NERSC主机的身份验证。**

如果这些工具支持ssh密钥验证，您可以使用ssh代理生成的ssh密钥。在这种情况下，请查看上面的“SSH配置文件选项”一章。另外请阅读该工具的使用说明和相关文档。BBEdit和FileZilla都支持ssh密钥验证，但FileZilla需要一些手动设置。
如果您使用WinSCP，请在“File protocol”中选择“SCP”，在登录窗口中输入您的用户名，把“password”一栏留空。然后点击“login”按钮，并在“Authentication Banner”窗口中点击“Continue”按钮。您会看到弹出窗口，请在里面输入NIM密码和一次性密码。

### 问题和建议

如有任何问题，请联系 [https://help.nersc.gov/](https://help.nersc.gov)
