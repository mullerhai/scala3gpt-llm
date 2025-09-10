package deepseek

import torch.{Float32, Tensor,::, Default, FloatNN,nn,Int64}
import torch.nn.functional as F
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import torch.nn.modules.{HasParams, TensorModule}
import torch.nn.functional.{dropout, softmax}
import torch.optim.Optimizer
import torch.utils.data.{DataLoader, Dataset}
import upack.MsgPackKeys.Int64

import scala.util.Random
import scala.math.sqrt

// RMS 归一化实现
class DeepseekV2RMSNorm[ParamType <: FloatNN: Default](hiddenSize: Int, eps: Double = 1e-6)
    extends HasParams[ParamType] with TensorModule[ParamType]:
  // 创建可训练参数
  val weight = register_parameter("weight", torch.ones(hiddenSize))
  val varianceEpsilon: Double = eps

  def forward(input: Tensor[ParamType]): Tensor[ParamType] =
    // 保存输入类型，用于之后转换回来
    val inputDtype = input.dtype
    // 转换为 Float32 进行计算
    val hiddenStates = input
    // 计算方差
    val variance = hiddenStates.pow(2).mean(-1, keepdim = true)
    // 归一化
    val normalized = hiddenStates * variance.add(varianceEpsilon).rsqrt()
    // 应用权重并转回原始类型
    (weight * normalized).to(inputDtype)

  // 实现 apply 方法，默认调用 forward
  def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

// 旋转位置编码实现
class DeepseekV2RotaryEmbedding[ParamType <: FloatNN: Default](
    dim: Int,
    maxPositionEmbeddings: Int = 2048,
    base: Double = 10000.0
) extends HasParams[ParamType]:
  // 计算逆频率
  private val invFreq: Tensor[ParamType] = {
    val arange = torch.arange(0, dim, 2).to(Float32)
    val freq =1.0/ (base ** (arange / dim.toFloat32))
    register_buffer("invFreq", freq)
  }

  // 初始化缓存
  private var cosCached: Tensor[ParamType] = _
  private var sinCached: Tensor[ParamType] = _
  private var maxSeqLenCached: Int = _

  // 设置余弦和正弦缓存
  private def setCosSinCache(seqLen: Int, device: torch.Device, dtype: torch.DType): Unit =
    if maxSeqLenCached != seqLen then
      maxSeqLenCached = seqLen
      val t = torch.arange(end= seqLen, device = device, dtype = invFreq.dtype)
      val freqs = torch.outer(t, invFreq.to(t.device))
      val emb = torch.cat(Seq(freqs, freqs), dim = -1)
      cosCached = register_buffer("cosCached", emb.cos().to(dtype))
      sinCached = register_buffer("sinCached", emb.sin().to(dtype))

  def forward(x: Tensor[ParamType], seqLen: Int = -1): (Tensor[ParamType], Tensor[ParamType]) =
    val seqLenValue = if seqLen > 0 then seqLen else x.shape(2)
    setCosSinCache(seqLenValue, x.device, x.dtype)
    (cosCached(0.::( seqLenValue)), sinCached(0.::(seqLenValue)))

  // 实现 apply 方法，默认调用 forward
  def apply(x: Tensor[ParamType], seqLen: Int = -1): (Tensor[ParamType], Tensor[ParamType]) = forward(x, seqLen)

// 旋转一半隐藏维度
object RotaryEmbeddingUtils {
  def rotateHalf[ParamType <: FloatNN: Default](x: Tensor[ParamType]): Tensor[ParamType] =
    val dim = x.shape(-1)
    val x1 = x(0.::(dim / 2), dim = -1)
    val x2 = x((dim / 2).::(dim), dim = -1)
    torch.cat(Seq(-x2, x1), dim = -1)

  // 应用旋转位置编码
  def applyRotaryPosEmb[ParamType <: FloatNN: Default](
      q: Tensor[ParamType],
      k: Tensor[ParamType],
      cos: Tensor[ParamType],
      sin: Tensor[ParamType],
      positionIds: Tensor[Int64],
      unsqueezeDim: Int = 1
  ): (Tensor[ParamType], Tensor[ParamType]) =
    val cosExpanded = cos.index_select(0, positionIds).unsqueeze(unsqueezeDim)
    val sinExpanded = sin.index_select(0, positionIds).unsqueeze(unsqueezeDim)

    // 重塑 q 和 k 以应用旋转
    val qShape = q.shape
    val qReshaped = q.view(qShape(0), qShape(1), qShape(2), qShape(3) / 2, 2)
      .transpose(3, 4)
      .reshape(qShape*)

    val kShape = k.shape
    val kReshaped = k.view(kShape(0), kShape(1), kShape(2), kShape(3) / 2, 2)
      .transpose(3, 4)
      .reshape(kShape*)

    // 应用旋转编码
    val qEmbed = qReshaped * cosExpanded + rotateHalf(qReshaped) * sinExpanded
    val kEmbed = kReshaped * cosExpanded + rotateHalf(kReshaped) * sinExpanded
    (qEmbed, kEmbed)
}

// 配置类
case class DeepseekConfig(
    hiddenSize: Int,
    numHeads: Int,
    maxPositionEmbeddings: Int,
    ropeTheta: Double,
    attentionDropout: Double,
    qLoraRank: Int,
    qkRopeHeadDim: Int,
    kvLoraRank: Int,
    vHeadDim: Int,
    qkNopeHeadDim: Int,
    attentionBias: Boolean
)

// MLA 核心实现
class MLA[ParamType <: FloatNN: Default](config: DeepseekConfig)
    extends HasParams[ParamType] with TensorModule[ParamType]:
  import RotaryEmbeddingUtils._

  // 基本参数
  val attentionDropout: Double = config.attentionDropout
  val hiddenSize: Int = config.hiddenSize
  val numHeads: Int = config.numHeads
  val vHeadDim: Int = config.vHeadDim

  // 输出投影层
  val outProj = register_module(
    "outProj",
    nn.Linear[ParamType](numHeads * vHeadDim, hiddenSize, bias = false)
  )

  // MLA 压缩部分 - 下采样
  val qkNopeHeadDim: Int = config.qkNopeHeadDim
  val qkRopeHeadDim: Int = config.qkRopeHeadDim
  val qLoraRank: Int = config.qLoraRank
  val kvLoraRank: Int = config.kvLoraRank

  // Q 下采样投影
  val qDownProj = register_module(
    "qDownProj",
    nn.Linear[ParamType](hiddenSize, qLoraRank, bias = config.attentionBias)
  )
  val qDownNorm: DeepseekV2RMSNorm[ParamType] = register_module(
    "qDownNorm",
    DeepseekV2RMSNorm[ParamType](qLoraRank)
  )

  // KV 下采样投影
  val kvDownProj = register_module(
    "kvDownProj",
    nn.Linear[ParamType](hiddenSize, kvLoraRank + config.qkRopeHeadDim, bias = config.attentionBias)
  )
  val kvDownNorm: DeepseekV2RMSNorm[ParamType] = register_module(
    "kvDownNorm",
    DeepseekV2RMSNorm[ParamType](kvLoraRank)
  )

  // 上采样部分
  val qHeadDim: Int = config.qkNopeHeadDim + config.qkRopeHeadDim
  val qUpProj = register_module(
    "qUpProj",
    nn.Linear[ParamType](qLoraRank, numHeads * qHeadDim, bias = config.attentionBias)
  )

  val kvUpProj = register_module(
    "kvUpProj",
    nn.Linear[ParamType](
      kvLoraRank,
      numHeads * (qHeadDim - config.qkRopeHeadDim + vHeadDim),
      bias = config.attentionBias
    )
  )

  // 旋转位置编码
  val rotaryEmb: DeepseekV2RotaryEmbedding[ParamType] = register_module(
    "rotaryEmb",
    DeepseekV2RotaryEmbedding[ParamType](
      config.qkRopeHeadDim,
      config.maxPositionEmbeddings,
      config.ropeTheta
    )
  )

  def forward(
      hiddenStates: Tensor[ParamType],
      positionIds: Tensor[Int64],
      attentionMask: Option[Tensor[Int64]] = None
  ): (Tensor[ParamType], Tensor[ParamType]) =
    val (bsz, qLen, _) = (hiddenStates.shape(0), hiddenStates.shape(1), hiddenStates.shape(2))

    // 1. Q 压缩处理
    var q = qDownProj(hiddenStates)
    q = qDownNorm(q)
    q = qUpProj(q)
    q = q.view(bsz, qLen, numHeads, qHeadDim).transpose(1, 2)

    // 分割 Q 为 nope 和 rope 部分
    val qNope_qRope = q.split(Seq(qkNopeHeadDim, qkRopeHeadDim), dim = -1)

    val qNope = qNope_qRope(0)
    val qRope = qNope_qRope(1)
    // 2. KV 处理
    var cKv = kvDownProj(hiddenStates)
    val cKvSplit_kRope = cKv.split(Seq(kvLoraRank, qkRopeHeadDim), dim = -1)
    val cKvSplit = cKvSplit_kRope(0)
    val kRope = cKvSplit_kRope(1)
    // 重塑 kRope 用于广播
    val kRopeReshaped = kRope.view(bsz, qLen, 1, qkRopeHeadDim).transpose(1, 2)

    var kv = kvDownNorm(cKvSplit)
    kv = kvUpProj(kv)
    kv = kv.view(bsz, qLen, numHeads, qkNopeHeadDim + vHeadDim).transpose(1, 2)

    // 分割 K 和 V
    val kNope_valueStates = kv.split(Seq(qkNopeHeadDim, vHeadDim), dim = -1)
    val kNope = kNope_valueStates(0)
    val valueStates = kNope_valueStates(1)

    // 3. 应用旋转位置编码
    val kvSeqLen = valueStates.shape(2)
    val (cos, sin) = rotaryEmb(valueStates, kvSeqLen)
    val (qRopeEmbed, kRopeEmbed) = applyRotaryPosEmb(qRope, kRopeReshaped, cos, sin, positionIds)

    // 4. 构建完整的 Query 和 Key
    val queryStates = torch.cat(Seq(qNope, qRopeEmbed), dim = -1)
    val expandDims = kRopeEmbed.expand(-1, numHeads, -1, -1).to(hiddenStates.dtype)
    val keyStates = torch.cat(Seq(kNope, expandDims), dim = -1)

    // 5. 计算注意力权重
    var attnWeights = torch.matmul(queryStates, keyStates.transpose(2, 3))
    attnWeights = attnWeights / sqrt(qHeadDim).toFloat32

    // 6. 应用注意力掩码
    if attentionMask.isDefined then
      attnWeights = attnWeights.masked_fill(attentionMask.get.toBoolean, Float.NegativeInfinity.toFloat32)

    // 7. Softmax 和 Dropout
    attnWeights = softmax(attnWeights, dim = -1)
    attnWeights = dropout(attnWeights, p = attentionDropout, training = this.training)

    // 8. 应用注意力权重到值
    var output = torch.matmul(attnWeights, valueStates)
    output = output.transpose(1, 2).reshape(bsz, qLen, -1)

    // 9. 输出投影
    output = outProj(output)

    (output, attnWeights)

  // 实现 apply 方法，默认调用 forward
  def apply(
      hiddenStates: Tensor[ParamType],
      positionIds: Tensor[Int64],
      attentionMask: Option[Tensor[Int64]] = None
  ): (Tensor[ParamType], Tensor[ParamType]) = forward(hiddenStates, positionIds, attentionMask)

// 为了完整性，添加一个简单的 Linear 层实现
class Linear[ParamType <: FloatNN: Default](
    inFeatures: Int,
    outFeatures: Int,
    bias: Boolean = true
) extends HasParams[ParamType] with TensorModule[ParamType]:
  val weight = register_parameter(
    "weight",
    Tensor.randn(outFeatures, inFeatures) * sqrt(1.0 / inFeatures)
  )
  val biasParam: Option[Tensor[ParamType]] =
    if bias then
      Some(register_parameter("bias", Tensor.zeros(outFeatures)))
    else None

  def forward(input: Tensor[ParamType]): Tensor[ParamType] =
    val output = torch.matmul(input, weight.t)
    biasParam.map(b => output + b).getOrElse(output)

  def apply(input: Tensor[ParamType]): Tensor[ParamType] = forward(input)

// 损失函数 - 交叉熵损失
def crossEntropyLoss[ParamType <: FloatNN: Default](
    input: Tensor[ParamType],
    target: Tensor[Int]
): Tensor[ParamType] =
  // 简化实现，实际使用时可能需要更复杂的处理
  val logProbs = torch.log_softmax(input, dim = -1)
  val nllLoss = -logProbs.gather(1l, target.unsqueeze(1).to(Int64)).squeeze(1)
  nllLoss.mean()

// 训练循环示例
def trainMLA[ParamType <: FloatNN: Default](
    model: MLA[ParamType],
    dataLoader: DataLoader[Tuple2[Tensor[ParamType], Tensor[Int64]]],
    optimizer: Optimizer,
    epochs: Int
): Unit =
  model.train()
  for epoch <- 0 until epochs do
    var totalLoss = 0.0
    var count = 0
    for (batch <- dataLoader) {
      val (inputs, targets) = batch
      optimizer.zeroGrad()

      // 生成位置 ID
      val seqLen = inputs.shape(1)
      val positionIds = torch.arange[Int](seqLen).unsqueeze(0).expand(inputs.shape(0), -1)

      // 前向传播
      val (output, _) = model(inputs, positionIds)

      // 计算损失
      val loss = crossEntropyLoss(output, targets)

      // 反向传播和优化
      loss.backward()
      optimizer.step()

      totalLoss += loss.item().toDouble
      count += 1
    }
    println(s"Epoch $epoch, Average Loss: ${totalLoss / count}")

// 测试函数
object MLATest {
  def main(args: Array[String]): Unit =
    // 设置随机种子
    torch.manualSeed(42)

    // 创建配置
    val config = DeepseekConfig(
      hiddenSize = 7168,
      numHeads = 16,
      maxPositionEmbeddings = 1024,
      ropeTheta = 128000.0,
      attentionDropout = 0.1,
      qLoraRank = 1536,
      qkRopeHeadDim = 64,
      kvLoraRank = 512,
      vHeadDim = 128,
      qkNopeHeadDim = 128,
      attentionBias = false
    )

    // 创建模型
    val model = MLA[Float32](config)

    // 测试前向传播
    val x = torch.randn[Float32](2, 1024, 7168)
    val positionIds = torch.arange(end = config.maxPositionEmbeddings)
      .unsqueeze(0).expand(x.shape(0), -1)

    val (attnOutput, attnWeights) = model(x, positionIds)
    println(s"Output shape: ${attnOutput.shape}")
    println(s"Attention weights shape: ${attnWeights.shape}")

    // 这里可以添加完整的训练代码，需要准备数据集和优化器
}
