# Support Vector Machine
## 模型推导
给定超平面 $$\bm{w}^T \bm{x} + b = 0 $$
和空间中得任意一点$\bm{x}_0$,则点到平面的距离可以表示为 $$d(\bm{x}_0,P) = \frac{|\bm{w}^T \bm{x}_0 + b|}{||\bm{w}||}$$

假设SVM可以把样本空间中的所有的点$(\bm{x}_i,y_i)$都正确分类，那么他们到平面的距离就可以表示为
$$d(\bm{x}_i,P) = \frac{|\bm{w}^T \bm{x}_i + b|}{||\bm{w}||}$$，我们取$$\hat{d} = min(d)$$
SVM要做的就是最大化间隔，也就是使得分类超平面到两个类的距离尽可能远。也就是求 $$ max\ \hat{d} = max\  min(d)$$
通过观察可以发现，式子的分母可以通过w控制，于是问题就可以转化为求
$$ max \  \frac{1}{||w||}$$ 等价于
$$ min \frac{1}{2}||w||^2$$
约束条件$$ \ y_i(\bm{w}^T\bm{x}_i+b)\geq 1$$
只要求出上述最优化问题的解可得$\bm{w}$，同时对于支持向量有
$$ \ y_i(\bm{w}^T\bm{x}_i+b) = 1$$ 可以求出 $b$
## 最优化求解
求解上文提到的带约束的最优化问题，一般可以用Lagrange 乘子法。
$$L(\bm{x},\lambda) = f(x) + \lambda g(x)$$

$$L(\bm{w},b,\bm{\lambda}) = \frac{1}{2}||w||^2 + \sum_{i=1}^m \lambda_i (1-\ y_i(\bm{w}^T\bm{x}_i+b))$$
令$$\frac{\partial L} {\partial \bm{w}}=0,\frac{\partial L} {\partial \bm{b}}=0,$$
得到
$$\bm{w} = \sum_{i=1}^m \lambda_i y_i \bm{x}_i$$
$$0 = \sum_{i=1}^m \lambda_i y_i$$
消去$\bm{w},b$得到一个仅关于$\bm{\lambda}$的问题：

$$max \ \sum_{i=1}^{m} \lambda_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m}
\lambda_i \lambda_j y_i y_j \bm{x}_i^T \bm{x}_j$$
约束条件：
$$0 = \sum_{i=1}^m \lambda_i y_i$$
$$\lambda_i \geq 0$$
求解上述的问题可以用SMO(Sequential Minimal Optimization)。
## 核方法

对于有些数据集，在低维空间内不是线性可分的，需要将它映射到高维度，从而达到线性可分的目的。
将$x \rightarrow \phi(x)$,希望可以在 $\phi(x)$所在的空间里可以找到一个超平面可以将数据集分隔开。
于是优化的函数变为：
$$max \ \sum_{i=1}^{m} \lambda_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m}
\lambda_i \lambda_j y_i y_j \phi(\bm{x}_i)^T \phi(\bm{x}_j)$$
约束条件：
$$0 = \sum_{i=1}^m \lambda_i y_i$$
$$\lambda_i \geq 0$$
核方法就是对于上面的$$\phi(\bm{x}_i)^T \phi(\bm{x}_j)$$可能存在计算困难得问题，于是我们有
$$k(\bm{x}_i, \bm{x}_j)=\phi(\bm{x}_i)^T \phi(\bm{x}_j)$$,即通过计算低维空间中的一个二元函数来代替高维空间中的向量乘积。