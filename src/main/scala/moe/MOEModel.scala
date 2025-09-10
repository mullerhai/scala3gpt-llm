package moe

import torch.{Float32,Int64,Int32, *}
import torch.nn.*
import torch.nn.functional as F
import torch.nn.modules.{HasParams, TensorModule}
import torch.optim.{Adam, Optimizer}

import scala.util.Random

// 设置随机种子
//Random.setSeed(1024)
//torch.manualSeed(1024)

// 基础专家模型
//class BasicExpert[ParamType <: FloatNN: Default](featureIn: Int, featureOut: Int) extends HasParams[ParamType]
//with TensorModule[ParamType]  {
//  val linear = register(nn.Linear(featureIn, featureOut))
//  def apply(x: Tensor[ParamType]): Tensor[ParamType] = {
//    linear.forward(x)
//  }
//  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
//    linear.forward(x)
//  }
//}

// 基础MOE模型
//class BasicMOE[ParamType <: FloatNN: Default](featureIn: Int, featureOut: Int, expertNumber: Int) extends HasParams[ParamType]
//  with TensorModule[ParamType]  {
//  val experts = nn.ModuleList((0 until expertNumber).map(num => new BasicExpert(featureIn, featureOut))*)
//  val gate = register(Linear(featureIn, expertNumber))
//  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
//  def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
//    // x 的形状是 (batch, featureIn)
//    val expertWeight = gate.forward(x)  // 形状是 (batch, expertNumber)
//    // 计算每个专家的输出并增加一个维度
//    val expertOutList = experts.map(expert => expert(x).unsqueeze(1))
//    // 拼接专家输出，形状变为 (batch, expertNumber, featureOut)
//    val expertOutput = torch.cat(expertOutList.toSeq, dim = 1)
//    // 调整权重形状以进行矩阵乘法
//    val reshapedWeight = expertWeight.unsqueeze(1)  // (batch, 1, expertNumber)
//    
//    // 矩阵乘法计算最终输出
//    val output = reshapedWeight.matmul(expertOutput)  // (batch, 1, featureOut)
//    
//    // 移除多余的维度
//    output.squeeze()
//  }
//}

// 测试基础MOE模型
object TestBasicMOE {
  def apply(): Unit = {
    val x = torch.rand(Seq(2, 4))
    val basicMoe = BasicMOE(4, 3, 2)
    val out = basicMoe.forward(x)
    println(out)
  }
}

// MOE路由器
//class MOERouter[ParamType <: FloatNN: Default](hiddenDim: Int, expertNumber: Int, topK: Int) extends HasParams[ParamType]
//  with TensorModule[ParamType]  {
//  val gate = register(nn.Linear(hiddenDim, expertNumber))
//  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
//  def forward(hiddenStates: Tensor[ParamType]): (Tensor[ParamType], Tensor[ParamType], Tensor[Int64], Tensor[ParamType]) = {
//    // 计算路由logits
//    val routerLogits = gate.forward(hiddenStates)  // 形状是 (b * s, expertNumber)
//    // 计算专家经过softmax之后的概率
//    val routingProbs = F.softmax(routerLogits, dim = -1, dtype = hiddenStates.dtype)
//    // 计算topk的专家的输出
//    val routerWeights_selectedExperts = torch.topk(routingProbs, topK, dim = -1) //, largest = true, sorted = true
//    val (routerWeights, selectedExperts) = (routerWeights_selectedExperts._1,routerWeights_selectedExperts._2)
//    // 专家权重归一化
//    val normalizedWeights = routerWeights / routerWeights.sum(dim = -1, keepdim = true)
//    val routerWeightsTyped = normalizedWeights.to(hiddenStates.dtype)
//    // 生成专家掩码
//    val expertMask = F.one_hot(selectedExperts, numClasses = expertNumber)
//    val permutedMask = expertMask.permute(2, 1, 0).to(hiddenStates.dtype)
//    (routerLogits, routerWeightsTyped, selectedExperts, permutedMask)
//  }
//}

// MOE配置类
//case class MOEConfig(
//    hiddenDim: Int,
//    expertNumber: Int,
//    topK: Int,
//    sharedExpertsNumber: Int = 2
//)

// 稀疏MOE模型
//class SparseMOE[ParamType <: FloatNN: Default](config: MOEConfig) extends HasParams[ParamType]
//  with TensorModule[ParamType] {
//  val hiddenDim: Int = config.hiddenDim
//  val expertNumber: Int = config.expertNumber
//  val topK: Int = config.topK
//  val experts = nn.ModuleList( (0 until expertNumber).map(num => new BasicExpert(hiddenDim, hiddenDim))* )
//  val router = MOERouter(hiddenDim, expertNumber, topK)
//  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
//  def forward(x: Tensor[ParamType]): (Tensor[ParamType], Tensor[ParamType]) = {
//    // x 形状是 (b, s, hiddenDim)
//    val batchSize = x.shape(0).toInt
//    val seqLen = x.shape(1).toInt
//    // 合并前两个维度，因为不是Sample维度了，而是token维度
//    val hiddenStates = x.view(-1, hiddenDim) // 形状是(b * s, hiddenDim)
//    val (routerLogits, routerWeights, selectedExpertsIndices, expertMask) = 
//      router.forward(hiddenStates)
//    // 创建输出张量
//    val finalHiddenStates = torch.zeros(
//      Seq(batchSize * seqLen, hiddenDim),
//      dtype = hiddenStates.dtype,
//      device = hiddenStates.device
//    )
//    // 对每个专家进行处理
////    hiddenStates.toArray
//    for (expertIdx <- 0 until expertNumber) {
//      val expertLayer = experts(expertIdx)
//      // 获取当前专家的掩码并找到需要处理的token
//      val idx_topx = torch.where(expertMask(expertIdx))
//      val idx = idx_topx(0)
//      val topX: Tensor[ParamType] = idx_topx(1)
////      topX.numpy().toArray
////      val (idx, topX) = torch.where(expertMask(expertIdx))
//      val hiddenStateUnsqueezed = hiddenStates.unsqueeze(0)
//      // 提取需要处理的token的隐藏状态
////      val currentState = hiddenStateUnsqueezed(::,topX.toArray.toSeq.asInstanceOf[Seq[Long]], ::).reshape(-1, hiddenDim)
//      val currentState = hiddenStateUnsqueezed(::, topX.to(DType.int64), ::).reshape(-1, hiddenDim)
//      // 应用专家层并加权
//      val weights = routerWeights(topX.to(DType.int64), idx.to(DType.int64)).unsqueeze(-1)
//      val currentHiddenStates = expertLayer(currentState) * weights
//
//      // 将当前专家的输出加到最终结果中
//      finalHiddenStates.index_add_(0, topX.to(DType.int64), currentHiddenStates.to(hiddenStates.dtype))
//    }
//    // 将结果还原到原始形状
//    val reshapedOutput = finalHiddenStates.reshape(batchSize, seqLen, hiddenDim)
//    
//    (reshapedOutput, routerLogits)
//  }
//}


//            current_state = hidden_states.unsqueeze(
//                0
//            )[:, top_x, :].reshape(-1, hidden_dim) # （selected_token_number, hidden_dim）
//
//            # router_weight 的 shape 是 (b * s, top_k)
//            current_hidden_states = expert_layer(
//                current_state
//            ) * router_weights[top_x, idx].unsqueeze(-1)  # （selected_token_number, 1） 这里有广播

// 测试稀疏MOE模型
object TestTokenLevelMOE {
  def apply(): Unit = {
    val x = torch.rand(Seq(2, 4, 16))
    val config = MOEConfig(16, 2, 2)
    val tokenLevelMoe = SparseMOE(config)
    val (out, logits) = tokenLevelMoe.forward(x)
    println(s"Output shape: ${out.shape}, Router logits shape: ${logits.shape}")
  }
}

// 共享专家的稀疏MOE模型
//class ShareExpertMOE[ParamType <: FloatNN: Default](config: MOEConfig) extends HasParams[ParamType]
//  with TensorModule[ParamType] {
//  val moeModel = SparseMOE(config)
//  val sharedExperts = nn.ModuleList(
//    (0 until config.sharedExpertsNumber).map(num => BasicExpert(config.hiddenDim, config.hiddenDim))*
//  )
//  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
//  def forward(x: Tensor[ParamType]): (Tensor[ParamType], Tensor[ParamType]) = {
//    // 首先通过moe模型
//    val (sparseMoeOut, routerLogits) = moeModel.forward(x)
//    
//    // 然后通过共享专家
//    val sharedExpertsOut = sharedExperts.map(expert => expert(x))
//    
//    // 堆叠共享专家的输出并求和
//    val sharedExpertsOutSum = torch.stack(sharedExpertsOut.toSeq, dim = 0).sum(dim = 0)
//    
//    // 将sparse_moe_out和shared_experts_out相加
//    (sparseMoeOut + sharedExpertsOutSum, routerLogits)
//  }
//}

// 测试共享专家的MOE模型
object TestShareExpertMOE {
  def apply(): Unit = {
    val x = torch.rand(Seq(2, 4, 16))
    val config = MOEConfig(16, 2, 2)
    val shareExpertMoe = ShareExpertMOE(config)
    val (out, logits) = shareExpertMoe.forward(x)
    println(s"Output shape: ${out.shape}, Router logits shape: ${logits.shape}")
  }
}

// 计算Switch Transformers的负载均衡损失
def switchLoadBalancingLoss[ParamType <: FloatNN: Default](routerLogits: Tensor[ParamType], numExperts: Int): Tensor[Float32] = {
  // 计算路由概率
  val routerProbs = torch.softmax(routerLogits, dim = -1)  // [b*s, numExperts]
  
  // 获取每个token的最优专家
  val anySelectedExperts = torch.topk(routerProbs, k = 2, dim = -1, largest = true, sorted = true)  // [b*s]
  
  // 创建one-hot矩阵表示选中的专家
  val mask = F.one_hot(anySelectedExperts._2.to(DType.int64), numExperts).float()  // [b*s, numExperts]
  
  // 计算每个专家的期望负载 (理想情况下应该是 1/numExperts)
  val expectedLoad = torch.onesLike(routerProbs) / numExperts
  
  // 计算实际负载 (每个专家处理的token数量除以总token数量)
  // 在batch维度上计算平均值
  val actualLoad = mask.mean(dim = 0)  // [numExperts]
  
  // 计算auxiliary loss
  // 这会惩罚负载分布与期望负载的差异
  val auxLoss = torch.sum(actualLoad * routerProbs.mean(dim = 0)) * numExperts
  
  // 计算z_loss (可选)
  // 这会惩罚过大的路由logits
  val zLoss = torch.mean(torch.square(routerLogits))
  val zLossWeight = 0.001f  // 可调整的超参数
  
  // 总损失
  val totalLoss = auxLoss + zLoss * zLossWeight
  
  totalLoss.to(torch.float32)
}

// 测试MOE训练
object TestMOETraining {
  def apply(): Unit = {
    // 创建简单的数据集参数
    val batchSize = 32
    val seqLen = 16
    val hiddenDim = 32
    val numBatches = 100
    
    // 初始化模型和优化器
    val config = MOEConfig(
      hiddenDim = hiddenDim,
      expertNumber = 4,
      topK = 2,
      sharedExpertsNumber = 2
    )
    
    val model = ShareExpertMOE(config)
    val optimizer = Adam(model.parameters(true), lr = 0.001f)
    
    // 训练循环
    model.train()
    for (batch <- 0 until numBatches) {
      // 生成随机输入数据
      val x = torch.randn(Seq(batchSize, seqLen, hiddenDim))
      val target = torch.randn(Seq(batchSize, seqLen, hiddenDim))
      
      // 前向传播
      val (output, routerLogits) = model.forward(x)
      
      // 计算损失
      // 预测的MSE损失
      val mseLoss = F.mse_loss(output, target)
      
      val auxLoss = switchLoadBalancingLoss(routerLogits, config.expertNumber)
      // 组合损失
      val totalLoss = mseLoss + 0.01f * auxLoss
      
      // 反向传播和优化
      optimizer.zeroGrad()
      totalLoss.backward()
      optimizer.step()
      
      if (batch % 10 == 0) {
        println(f"Batch $batch, Loss: ${totalLoss.item} " +
                f"(MSE: ${mseLoss.item}, Aux: ${auxLoss.item})")
      }
    }
  }
}

// 运行所有测试
object MOETests {
  def main(args: Array[String]): Unit = {
    println("Testing Basic MOE:")
    TestBasicMOE()
    
    println("\nTesting Token Level MOE:")
    TestTokenLevelMOE()
    
    println("\nTesting Share Expert MOE:")
    TestShareExpertMOE()
    
    println("\nTesting MOE Training:")
    TestMOETraining()
  }
}
