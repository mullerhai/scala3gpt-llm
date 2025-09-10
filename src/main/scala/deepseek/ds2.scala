//package deepseek
//
//import torch.Tensor
//import torch.*
//import torch.nn.functional as F
//import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
//import torch.nn.modules.{HasParams, TensorModule}
//import torch.nn.functional.{dropout, softmax}
//import torch.optim.Optimizer
//import torch.utils.data.{DataLoader, Dataset}
//import scala.util.Random
//import scala.math.sqrt
//
//// RMS 归一化实现
//class DeepseekV2RMSNorm[ParamType <: FloatNN: Default](hiddenSize: Int, eps: Double = 1e-6)
//    extends HasParams[ParamType] with TensorModule[ParamType]:
//  // 创建可训练参数
//  val weight: Tensor[ParamType] = registerParameter("weight", Tensor.ones[ParamType](hiddenSize))
//  val varianceEpsilon: Double = eps
//
//  def forward(input: Tensor[ParamType]): Tensor[ParamType] =
//    // 保存输入类型，用于之后转换回来
//    val inputDtype = input.dtype
//    // 转换为 Float32 进行计算
//    val hiddenStates = input.toFloat32
//    // 计算方差
//    val variance = hiddenStates.pow(2).mean(-1, keepdim = true)
//    // 归一化
//    val normalized = hiddenStates * variance.add(varianceEpsilon).rsqrt()
//    // 应用权重并转回原始类型
//    (weight * normalized).toType(inputDtype)
//
//  // 实现 apply 方法，默认调用 forward
//  def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)
//
//// 旋转位置编码实现
//class DeepseekV2RotaryEmbedding[ParamType <: FloatNN: Default](
//    dim: Int,
//    maxPositionEmbeddings: Int = 2048,
//    base: Double = 10000.0
//) extends HasParams[ParamType]:
//  // 计算逆频率
//  private val invFreq: Tensor[ParamType] = {
//    val arange = Tensor.arange[ParamType](0, dim, 2).toFloat32
//    val freq = Tensor.full[ParamType](1.0) / (base ** (arange / dim.toFloat32))
//    registerBuffer("invFreq", freq)
//  }
//
//  // 初始化缓存
//  private var cosCached: Tensor[ParamType] = _
//  private var sinCached: Tensor[ParamType] = _
//  private var maxSeqLenCached: Int = maxPositionEmbeddings
//
//  // 设置余弦和正弦缓存
//  private def setCosSinCache(seqLen: Int, device: torch.Device, dtype: torch.DType): Unit =
//    if maxSeqLenCached != seqLen then
//      maxSeqLenCached = seqLen
//      val t = Tensor.arange[ParamType](seqLen, device = device, dtype = invFreq.dtype)
//      val freqs = torch.outer(t, invFreq.to(t.device))
//      val emb = torch.cat(Seq(freqs, freqs), dim = -1)
//      cosCached = registerBuffer("cosCached", emb.cos().toType(dtype))
//      sinCached = registerBuffer("sinCached", emb.sin().toType(dtype))
//
//  // 初始化缓存
//  setCosSinCache(maxPositionEmbeddings, invFreq.device, invFreq.dtype)
//
//  def forward(x: Tensor[ParamType], seqLen: Option[Int] = None): (Tensor[ParamType], Tensor[ParamType]) =
//    val seqLenValue = seqLen.getOrElse(x.shape(2))
//    if seqLen.isDefined && seqLenValue > maxSeqLenCached then
//      println(s"seq_len: $seqLenValue, $maxSeqLenCached")
//      setCosSinCache(seqLenValue, x.device, x.dtype)
//
//    (cosCached.slice(Seq(0, seqLenValue)), sinCached.slice(Seq(0, seqLenValue)))
//
//  // 实现 apply 方法，默认调用 forward
//  def apply(x: Tensor[ParamType], seqLen: Option[Int] = None): (Tensor[ParamType], Tensor[ParamType]) =
//    forward(x, seqLen)
//
//// 旋转一半隐藏维度
//object RotaryEmbeddingUtils {
//  def rotateHalf[ParamType <: FloatNN: Default](x: Tensor[ParamType]): Tensor[ParamType] =
//    val dim = x.shape(-1)
//    val x1 = x.slice(Seq(0, dim / 2), dim = -1)
//    val x2 = x.slice(Seq(dim / 2, dim), dim = -1)
//    torch.cat(Seq(-x2, x1), dim = -1)
//
//  // 应用旋转位置编码（标准版本）
//  def applyRotaryPosEmb[ParamType <: FloatNN: Default](
//      q: Tensor[ParamType],
//      k: Tensor[ParamType],
//      cos: Tensor[ParamType],
//      sin: Tensor[ParamType],
//      positionIds: Tensor[Int],
//      unsqueezeDim: Int = 1
//  ): (Tensor[ParamType], Tensor[ParamType]) =
//    val cosExpanded = cos.indexSelect(0, positionIds).unsqueeze(unsqueezeDim)
//    val sinExpanded = sin.indexSelect(0, positionIds).unsqueeze(unsqueezeDim)
//
//    // 重塑 q 和 k 以应用旋转
//    val qShape = q.shape
//    val qReshaped = q.view(qShape(0), qShape(1), qShape(2), qShape(3) / 2, 2)
//      .transpose(3, 4)
//      .reshape(qShape)
//
//    val kShape = k.shape
//    val kReshaped = k.view(kShape(0), kShape(1), kShape(2), kShape(3) / 2, 2)
//      .transpose(3, 4)
//      .reshape(kShape)
//
//    // 应用旋转编码
//    val qEmbed = qReshaped * cosExpanded + rotateHalf(qReshaped) * sinExpanded
//    val kEmbed = kReshaped * cosExpanded + rotateHalf(kReshaped) * sinExpanded
//    (qEmbed, kEmbed)
//
//  // 应用旋转位置编码（v2 版本，仅处理 q）
//  def applyRotaryPosEmbV2[ParamType <: FloatNN: Default](
//      q: Tensor[ParamType],
//      cos: Tensor[ParamType],
//      sin: Tensor[ParamType],
//      positionIds: Tensor[Int],
//      unsqueezeDim: Int = 1
//  ): Tensor[ParamType] =
//    val cosExpanded = cos.indexSelect(0, positionIds).unsqueeze(unsqueezeDim)
//    val sinExpanded = sin.indexSelect(0, positionIds).unsqueeze(unsqueezeDim)
//
//    // 重塑 q 以应用旋转
//    val qShape = q.shape
//    val qReshaped = q.view(qShape(0), qShape(1), qShape(2), qShape(3) / 2, 2)
//      .transpose(3, 4)
//      .reshape(qShape)
//
//    // 应用旋转编码
//    qReshaped * cosExpanded + rotateHalf(qReshaped) * sinExpanded
//}
//
//// 配置类
//case class DeepseekConfig(
//    hiddenSize: Int,
//    numHeads: Int,
//    maxPositionEmbeddings: Int,
//    ropeTheta: Double,
//    attentionDropout: Double,
//    qLoraRank: Int,
//    qkRopeHeadDim: Int,
//    kvLoraRank: Int,
//    vHeadDim: Int,
//    qkNopeHeadDim: Int,
//    attentionBias: Boolean
//)
//
//// 为了完整性，添加一个简单的 Linear 层实现
//class Linear[ParamType <: FloatNN: Default](
//    inFeatures: Int,
//    outFeatures: Int,
//    bias: Boolean = true
//) extends HasParams[ParamType] with TensorModule[ParamType]:
//  val weight: Tensor[ParamType] = registerParameter(
//    "weight",
//    Tensor.randn[ParamType](outFeatures, inFeatures) * sqrt(1.0 / inFeatures)
//  )
//  val biasParam: Option[Tensor[ParamType]] = 
//    if bias then
//      Some(registerParameter("bias", Tensor.zeros[ParamType](outFeatures)))
//    else None
//
//  def forward(input: Tensor[ParamType]): Tensor[ParamType] =
//    val output = torch.matmul(input, weight.t)
//    biasParam.map(b => output + b).getOrElse(output)
//
//  def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)
//
//// MLA v2 核心实现（带有矩阵吸收）
//class MLAV2[ParamType <: FloatNN: Default](config: DeepseekConfig)
//    extends HasParams[ParamType] with TensorModule[ParamType]:
//  import RotaryEmbeddingUtils._
//
//  // 基本参数
//  val attentionDropout: Double = config.attentionDropout
//  val hiddenSize: Int = config.hiddenSize
//  val numHeads: Int = config.numHeads
//  val maxPostionEmbeddings: Int = config.maxPositionEmbeddings
//  val ropeTheta: Double = config.ropeTheta
//
//  // 压缩相关参数
//  val qLoraRank: Int = config.qLoraRank
//  val qkRopeHeadDim: Int = config.qkRopeHeadDim
//  val kvLoraRank: Int = config.kvLoraRank
//  val vHeadDim: Int = config.vHeadDim
//  val qkNopeHeadDim: Int = config.qkNopeHeadDim
//  val qHeadDim: Int = config.qkNopeHeadDim + config.qkRopeHeadDim
//
//  // Query 处理相关层
//  val qDownProj: Linear[ParamType] = registerModule(
//    "qDownProj",
//    Linear[ParamType](hiddenSize, qLoraRank, bias = config.attentionBias)
//  )
//  val qDownLayernorm: DeepseekV2RMSNorm[ParamType] = registerModule(
//    "qDownLayernorm",
//    DeepseekV2RMSNorm[ParamType](qLoraRank)
//  )
//  val qUpProj: Linear[ParamType] = registerModule(
//    "qUpProj",
//    Linear[ParamType](qLoraRank, numHeads * qHeadDim, bias = false)
//  )
//
//  // Key/Value 处理相关层
//  val kvDownProj: Linear[ParamType] = registerModule(
//    "kvDownProj",
//    Linear[ParamType](hiddenSize, kvLoraRank + qkRopeHeadDim, bias = config.attentionBias)
//  )
//  val kvDownLayernorm: DeepseekV2RMSNorm[ParamType] = registerModule(
//    "kvDownLayernorm",
//    DeepseekV2RMSNorm[ParamType](kvLoraRank)
//  )
//  val kvUpProj: Linear[ParamType] = registerModule(
//    "kvUpProj",
//    Linear[ParamType](
//      kvLoraRank,
//      numHeads * (qHeadDim - qkRopeHeadDim + vHeadDim),
//      bias = false
//    )
//  )
//
//  // 输出投影层
//  val oProj: Linear[ParamType] = registerModule(
//    "oProj",
//    Linear[ParamType](numHeads * vHeadDim, hiddenSize, bias = config.attentionBias)
//  )
//
//  // 旋转位置编码
//  val rotaryEmb: DeepseekV2RotaryEmbedding[ParamType] = registerModule(
//    "rotaryEmb",
//    DeepseekV2RotaryEmbedding[ParamType](
//      qkRopeHeadDim,
//      maxPostionEmbeddings,
//      ropeTheta
//    )
//  )
//
//  def forward(
//      hiddenStates: Tensor[ParamType],
//      attentionMask: Option[Tensor[Int]] = None,
//      positionIds: Option[Tensor[Int]] = None,
//      compressedKv: Option[Tensor[ParamType]] = None
//  ): (Tensor[ParamType], Tensor[ParamType]) =
//    val (bsz, qLen, _) = (hiddenStates.shape(0), hiddenStates.shape(1), hiddenStates.shape(2))
//
//    // 确保有 position_ids
//    val posIds = positionIds.getOrElse(
//      Tensor.arange[Int](qLen).unsqueeze(0).expand(bsz, -1)
//    )
//
//    // 1. Query 投影和分割
//    var q = qUpProj(qDownLayernorm(qDownProj(hiddenStates)))
//    q = q.view(bsz, qLen, numHeads, qHeadDim).transpose(1, 2)
//    val (qNope, qPe) = q.split(Seq(qkNopeHeadDim, qkRopeHeadDim), dim = -1)
//
//    // 2. Key/Value 投影和分割
//    val compressedKvValue = compressedKv.getOrElse(
//      // 如果没有提供 compressed_kv，生成一个随机的
//      Tensor.randn[ParamType](bsz, qLen, kvLoraRank + qkRopeHeadDim)
//    )
//    val kvSeqLen = compressedKvValue.shape(1)
//    val (compressedKvSplit, kPe) = compressedKvValue.split(
//      Seq(kvLoraRank, qkRopeHeadDim), 
//      dim = -1
//    )
//
//    // 重塑 kPe 用于广播
//    val kPeReshaped = kPe.view(bsz, kvSeqLen, 1, qkRopeHeadDim).transpose(1, 2)
//
//    // 3. 矩阵吸收 - 重塑 kv_up_proj 权重
//    val kvUpProjWeights = kvUpProj.weight.view(numHeads, -1, kvLoraRank)
//    val qAbsorb = kvUpProjWeights.slice(Seq(0, numHeads), Seq(0, qkNopeHeadDim), dim = -2)
//    val outAbsorb = kvUpProjWeights.slice(Seq(0, numHeads), Seq(qkNopeHeadDim, -1), dim = -2)
//
//    // 4. 应用旋转位置编码到与位置相关的部分
//    println(s"q_pe shape: ${qPe.shape}")
//
//    val (cos, sin) = rotaryEmb(qPe, Some(qLen))
//    val qPeEmbed = applyRotaryPosEmbV2(qPe, cos, sin, posIds)
//    
//    println(s"k_pe shape: ${kPeReshaped.shape}")
//    println(s"k pe mT shape: ${kPeReshaped.t.shape}")
//    println(s"compressed_kv shape: ${compressedKvSplit.shape}")
//    println(s"q_nope shape: ${qNope.shape}")
//
//    // 5. 应用矩阵吸收并计算注意力权重
//    val qNopeAbsorbed = torch.matmul(qNope, qAbsorb)
//    val attnWeights = (
//      torch.matmul(qPeEmbed, kPeReshaped.t) + 
//      torch.matmul(qNopeAbsorbed, compressedKvSplit.unsqueeze(-3).t)
//    ) / sqrt(qHeadDim).toFloat32
//
//    // 6. 应用 softmax
//    val attnWeightsSoftmax = softmax(
//      attnWeights, 
//      dim = -1, 
//      dtype = torch.float32
//    ).toType(qNope.dtype)
//
//    // 7. 计算注意力输出
//    // 使用爱因斯坦求和约定计算注意力输出
//    val attnOutput = torch.einsum(
//      "bhql,blc->bhqc", 
//      attnWeightsSoftmax, compressedKvSplit
//    )
//    
//    // 应用输出吸收矩阵
//    val attnOutputAbsorbed = torch.matmul(attnOutput, outAbsorb.t)
//    
//    // 重塑并应用输出投影
//    val output = attnOutputAbsorbed.transpose(1, 2).reshape(bsz, qLen, -1)
//    val finalOutput = oProj(output)
//
//    (finalOutput, attnWeightsSoftmax)
//
//  // 实现 apply 方法，默认调用 forward
//  def apply(
//      hiddenStates: Tensor[ParamType],
//      attentionMask: Option[Tensor[Int]] = None,
//      positionIds: Option[Tensor[Int]] = None,
//      compressedKv: Option[Tensor[ParamType]] = None
//  ): (Tensor[ParamType], Tensor[ParamType]) =
//    forward(hiddenStates, attentionMask, positionIds, compressedKv)
//
//// 损失函数 - 交叉熵损失
//def crossEntropyLoss[ParamType <: FloatNN: Default](
//    input: Tensor[ParamType],
//    target: Tensor[Int]
//): Tensor[ParamType] =
//  val logProbs = torch.log_softmax(input, dim = -1)
//  val nllLoss = -logProbs.gather(1, target.unsqueeze(1)).squeeze(1)
//  nllLoss.mean()
//
//// 训练循环示例
//def trainMLAV2[ParamType <: FloatNN: Default](
//    model: MLAV2[ParamType],
//    dataLoader: DataLoader[Tuple2[Tensor[ParamType], Tensor[Int]]],
//    optimizer: Optimizer[ParamType],
//    epochs: Int
//): Unit =
//  model.train()
//  for epoch <- 0 until epochs do
//    var totalLoss = 0.0
//    var count = 0
//    for (batch <- dataLoader) {
//      val (inputs, targets) = batch
//      optimizer.zeroGrad()
//      
//      // 生成位置 ID
//      val bsz = inputs.shape(0)
//      val qLen = inputs.shape(1)
//      val positionIds = Tensor.arange[Int](qLen).unsqueeze(0).expand(bsz, -1)
//      
//      // 生成 compressed_kv（简化版，实际应用中可能需要更复杂的处理）
//      val kvSeqLen = qLen // 可以根据实际情况调整
//      val compressedKv = Tensor.randn[ParamType](
//        bsz, kvSeqLen, model.kvLoraRank + model.qkRopeHeadDim
//      )
//      
//      // 前向传播
//      val (output, _) = model(inputs, None, Some(positionIds), Some(compressedKv))
//      
//      // 计算损失
//      val loss = crossEntropyLoss(output, targets)
//      
//      // 反向传播和优化
//      loss.backward()
//      optimizer.step()
//      
//      totalLoss += loss.item().toDouble
//      count += 1
//    }
//    println(s"Epoch $epoch, Average Loss: ${totalLoss / count}")
//
//// 测试函数
//object MLAV2Test {
//  def main(args: Array[String]): Unit =
//    // 设置随机种子
//    torch.manualSeed(42)
//    
//    // 创建配置
//    val config = DeepseekConfig(
//      hiddenSize = 7168,
//      numHeads = 16,
//      maxPositionEmbeddings = 1024,
//      ropeTheta = 128000.0,
//      attentionDropout = 0.1,
//      qLoraRank = 1536,
//      qkRopeHeadDim = 64,
//      kvLoraRank = 512,
//      vHeadDim = 128,
//      qkNopeHeadDim = 128,
//      attentionBias = false
//    )
//    
//    // 创建模型
//    val model = MLAV2[Float32](config)
//    
//    // 测试前向传播
//    val bsz = 2
//    val qLen = 1
//    val kvSeqLen = 12
//    val x = Tensor.randn[Float32](bsz, qLen, config.hiddenSize)
//    val positionIds = Tensor.full[Int](Seq(bsz, qLen), 12)
//    
//    // 生成 compressed_kv
//    val compressedKv = Tensor.randn[Float32](
//      bsz, kvSeqLen, config.kvLoraRank + config.qkRopeHeadDim
//    )
//    
//    println(s"compressed_kv shape: ${compressedKv.shape}")
//    
//    // 前向计算
//    val (output, attnWeights) = model(x, None, Some(positionIds), Some(compressedKv))
//    println(s"output shape: ${output.shape}")
//    println(s"attn_weights shape: ${attnWeights.shape}")
//    
//    // 这里可以添加完整的训练代码，需要准备数据集和优化器
//}
