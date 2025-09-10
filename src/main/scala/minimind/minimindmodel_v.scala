package minimind

import torch.*
import torch.nn.functional as F
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}
import torch.nn.modules.{HasParams, TensorModule}
import scala.collection.mutable.ListBuffer

// MiniMind 配置类
//class MiniMindConfig(
//    val dropout: Double = 0.0,
//    val bosTokenId: Int = 1,
//    val eosTokenId: Int = 2,
//    val hiddenAct: String = "silu",
//    val hiddenSize: Int = 512,
//    var intermediateSize: Option[Int] = None,
//    val maxPositionEmbeddings: Int = 32768,
//    val numAttentionHeads: Int = 8,
//    val numHiddenLayers: Int = 8,
//    val numKeyValueHeads: Int = 2,
//    val vocabSize: Int = 6400,
//    val rmsNormEps: Double = 1e-5,
//    val ropeTheta: Double = 1e6,
//    val flashAttn: Boolean = true,
//    val useMoe: Boolean = false,
//    val numExpertsPerTok: Int = 2,
//    val nRoutedExperts: Int = 4,
//    val nSharedExperts: Int = 1,
//    val scoringFunc: String = "softmax",
//    val auxLossAlpha: Double = 0.1,
//    val seqAux: Boolean = true,
//    val normTopkProb: Boolean = true
//)

// RMS 归一化层
//class RMSNorm[ParamType <: FloatNN: Default](dim: Int, eps: Double = 1e-5) extends HasParams[ParamType] with TensorModule[ParamType] {
//  val epsValue = eps
//  val weight =  torch.ones(dim)
//
//  register_parameter("weight", weight)
//
//  private def _norm(x: Tensor[ParamType]): Tensor[ParamType] = {
//    val xp = x.pow(2)
//    val xpMean = xp.mean(-1, keepdim = true)
//    val result = x * torch.rsqrt(xpMean + epsValue)
//    result.to(x.dtype)
//  }
//
//  override def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
//    weight.to(x.dtype) * _norm(x)//.to(x.dtype)
//  }
//
//  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
//}

//// 预计算旋转位置编码频率
//def precomputeFreqsCis[ParamType <: FloatNN: Default](dim: Int, end: Int = 32 * 1024, theta: Double = 1e6): (Tensor[ParamType], Tensor[ParamType]) = {
//  val freqs = (1.0 / (theta.pow(torch.arange[ParamType](0, dim, 2).slice(0, dim / 2) / dim.toParamType[ParamType]))).toType[ParamType]
//  val t = torch.arange[ParamType](end)
//  val outerFreqs = torch.outer(t, freqs).toType[ParamType]
//  val freqsCos = torch.cat(Seq(torch.cos(outerFreqs), torch.cos(outerFreqs)), dim = -1)
//  val freqsSin = torch.cat(Seq(torch.sin(outerFreqs), torch.sin(outerFreqs)), dim = -1)
//  (freqsCos, freqsSin)
//}
//
//// 应用旋转位置编码
//def applyRotaryPosEmb[ParamType <: FloatNN: Default](q: Tensor[ParamType], k: Tensor[ParamType], cos: Tensor[ParamType], sin: Tensor[ParamType], positionIds: Option[Tensor[ParamType]] = None, unsqueezeDim: Int = 1): (Tensor[ParamType], Tensor[ParamType]) = {
//  def rotateHalf(x: Tensor[ParamType]): Tensor[ParamType] = {
//    val halfSize = x.shape.last / 2
//    torch.cat(Seq(-x.slice(-halfSize, None), x.slice(0, halfSize)), dim = -1)
//  }
//
//  val qEmbed = (q * cos.unsqueeze(unsqueezeDim)) + (rotateHalf(q) * sin.unsqueeze(unsqueezeDim))
//  val kEmbed = (k * cos.unsqueeze(unsqueezeDim)) + (rotateHalf(k) * sin.unsqueeze(unsqueezeDim))
//  (qEmbed, kEmbed)
//}
//
//// 重复键值对
//def repeatKv[ParamType <: FloatNN: Default](x: Tensor[ParamType], nRep: Int): Tensor[ParamType] = {
//  val Array(bs, slen, numKeyValueHeads, headDim) = x.shape
//  if (nRep == 1) x
//  else {
//    x.unsqueeze(3).expand(bs, slen, numKeyValueHeads, nRep, headDim).reshape(bs, slen, numKeyValueHeads * nRep, headDim)
//  }
//}

// 注意力层
//class Attention[ParamType <: FloatNN: Default](args: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
//  val numKeyValueHeads = if (args.numKeyValueHeads == 0) args.numAttentionHeads else args.numKeyValueHeads
//  assert(args.numAttentionHeads % numKeyValueHeads == 0)
//  val nLocalHeads = args.numAttentionHeads
//  val nLocalKvHeads = numKeyValueHeads
//  val nRep = nLocalHeads / nLocalKvHeads
//  val headDim = args.hiddenSize / args.numAttentionHeads
//  val qProj = nn.Linear(args.hiddenSize, args.numAttentionHeads * headDim, bias = false)
//  val kProj = nn.Linear(args.hiddenSize, numKeyValueHeads * headDim, bias = false)
//  val vProj = nn.Linear(args.hiddenSize, numKeyValueHeads * headDim, bias = false)
//  val oProj = nn.Linear(args.numAttentionHeads * headDim, args.hiddenSize, bias = false)
//  val attnDropout = nn.Dropout(args.dropout)
//  val residDropout = nn.Dropout(args.dropout)
//  val dropoutValue = args.dropout
//  val flash = torch.nn.functional.hasScaledDotProductAttention && args.flashAttn
//
//  register_module("qProj", qProj)
//  register_module("kProj", kProj)
//  register_module("vProj", vProj)
//  register_module("oProj", oProj)
//  register_module("attnDropout", attnDropout)
//  register_module("residDropout", residDropout)
//
//  override def forward(x: Tensor[ParamType], positionEmbeddings: (Tensor[ParamType], Tensor[ParamType]), pastKeyValue: Option[(Tensor[ParamType], Tensor[ParamType])] = None, useCache: Boolean = false, attentionMask: Option[Tensor[ParamType]] = None): (Tensor[ParamType], Option[(Tensor[ParamType], Tensor[ParamType])]) = {
//    val (bsz, seqLen, _) = x.shape
//    val xq = qProj(x)
//    val xk = kProj(x)
//    val xv = vProj(x)
//    val xqReshaped = xq.view(bsz, seqLen, nLocalHeads, headDim)
//    val xkReshaped = xk.view(bsz, seqLen, nLocalKvHeads, headDim)
//    val xvReshaped = xv.view(bsz, seqLen, nLocalKvHeads, headDim)
//
//    val (cos, sin) = positionEmbeddings
//    val (qEmbed, kEmbed) = applyRotaryPosEmb(xqReshaped, xkReshaped, cos.slice(0, seqLen), sin.slice(0, seqLen))
//
//    var xkFinal = xkReshaped
//    var xvFinal = xvReshaped
//    if (pastKeyValue.isDefined) {
//      xkFinal = torch.cat(Seq(pastKeyValue.get._1, xkFinal), dim = 1)
//      xvFinal = torch.cat(Seq(pastKeyValue.get._2, xvFinal), dim = 1)
//    }
//    val pastKv = if (useCache) Some((xkFinal, xvFinal)) else None
//
//    val xqTransposed = xqReshaped.transpose(1, 2)
//    val xkTransposed = repeatKv(xkFinal, nRep).transpose(1, 2)
//    val xvTransposed = repeatKv(xvFinal, nRep).transpose(1, 2)
//
//    val output = if (flash && seqLen != 1) {
//      val dropoutP = if (isTraining) dropoutValue else 0.0
//      val attnMaskProcessed = attentionMask.map { mask =>
//        mask.view(bsz, 1, 1, -1).expand(bsz, nLocalHeads, seqLen, -1).bool()
//      }
//      torch.nn.functional.scaled_dot_product_attention(xqTransposed, xkTransposed, xvTransposed, attnMask = attnMaskProcessed, dropoutP = dropoutP, isCausal = true)
//    } else {
//      val scores = (xqTransposed @@ xkTransposed.transpose(-2, -1)) / math.sqrt(headDim).toParamType[ParamType]
//      val causalMask = torch.triu(torch.full(Seq(seqLen, seqLen), Double.NegativeInfinity.toParamType[ParamType], device = scores.device), diagonal = 1).unsqueeze(0).unsqueeze(0)
//      val scoresWithMask = scores + causalMask
//
//      attentionMask.foreach { mask =>
//        val extendedMask = mask.unsqueeze(1).unsqueeze(2)
//        val extendedMaskProcessed = (1.0.toParamType[ParamType] - extendedMask) * (-1e9).toParamType[ParamType]
//        scoresWithMask += extendedMaskProcessed
//      }
//
//      val softmaxScores = torch.softmax(scoresWithMask.toFloatNN, dim = -1).to(xqTransposed.dtype)
//      val droppedScores = attnDropout(softmaxScores)
//      droppedScores @ xvTransposed
//    }
//
//    val outputReshaped = output.transpose(1, 2).reshape(bsz, seqLen, -1)
//    val finalOutput = residDropout(oProj(outputReshaped))
//    (finalOutput, pastKv)
//  }
//
//  def apply(x: Tensor[ParamType], positionEmbeddings: (Tensor[ParamType], Tensor[ParamType]), pastKeyValue: Option[(Tensor[ParamType], Tensor[ParamType])] = None, useCache: Boolean = false, attentionMask: Option[Tensor[ParamType]] = None): (Tensor[ParamType], Option[(Tensor[ParamType], Tensor[ParamType])]) =
//    forward(x, positionEmbeddings, pastKeyValue, useCache, attentionMask)
//}

// 前馈网络
//class FeedForward[ParamType <: FloatNN: Default](config: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
//  val intermediateSize = config.intermediateSize.getOrElse {
//    val tempSize = (config.hiddenSize * 8 / 3).toInt
//    64 * ((tempSize + 64 - 1) / 64)
//  }
//  val gateProj = nn.Linear(config.hiddenSize, intermediateSize, bias = false)
//  val downProj = nn.Linear(intermediateSize, config.hiddenSize, bias = false)
//  val upProj = nn.Linear(config.hiddenSize, intermediateSize, bias = false)
//  val dropout = nn.Dropout(config.dropout)
//  val actFn = ACT2FN[ParamType](config.hiddenAct)
//
//  register_module("gateProj", gateProj)
//  register_module("downProj", downProj)
//  register_module("upProj", upProj)
//  register_module("dropout", dropout)
//
//  override def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
//    dropout(downProj(actFn(gateProj(x)) * upProj(x)))
//  }
//
//  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
//}

// MoE 门控层
//class MoEGate[ParamType <: FloatNN: Default](config: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
//  val topK = config.numExpertsPerTok
//  val nRoutedExperts = config.nRoutedExperts
//  val scoringFunc = config.scoringFunc
//  val alpha = config.auxLossAlpha
//  val seqAux = config.seqAux
//  val normTopkProb = config.normTopkProb
//  val gatingDim = config.hiddenSize
//  val weight: Tensor[ParamType] = torch.empty[ParamType](nRoutedExperts, gatingDim)
//
//  register_parameter("weight", weight)
//
//  resetParameters()
//
//  private def resetParameters(): Unit = {
//    KaimingUniform(weight, a = math.sqrt(5).toParamType[ParamType])
//  }
//
//  override def forward(hiddenStates: Tensor[ParamType]): (Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]) = {
//    val (bsz, seqLen, h) = hiddenStates.shape
//    val hiddenStatesFlattened = hiddenStates.view(-1, h)
//    val logits = torch.nn.functional.linear(hiddenStatesFlattened, weight)
//    val scores = scoringFunc match {
//      case "softmax" => torch.softmax(logits, dim = -1)
//      case _ => throw new NotImplementedError(s"unsupportable scoring function for MoE gating: $scoringFunc")
//    }
//
//    val (topkWeight, topkIdx) = torch.topk(scores, k = topK, dim = -1, sorted = false)
//
//    val normalizedTopkWeight = if (topK > 1 && normTopkProb) {
//      val denominator = topkWeight.sum(dim = -1, keepdim = true) + 1e-20.toParamType[ParamType]
//      topkWeight / denominator
//    } else {
//      topkWeight
//    }
//
//    val auxLoss = if (isTraining && alpha > 0.0) {
//      val scoresForAux = scores
//      val auxTopk = topK
//      val topkIdxForAuxLoss = topkIdx.view(bsz, -1)
//      if (seqAux) {
//        val scoresForSeqAux = scoresForAux.view(bsz, seqLen, -1)
//        val ce = torch.zeros[ParamType](bsz, nRoutedExperts, device = hiddenStates.device)
//        ce.scatterAdd(1, topkIdxForAuxLoss, torch.ones[ParamType](bsz, seqLen * auxTopk, device = hiddenStates.device)).div((seqLen * auxTopk / nRoutedExperts).toParamType[ParamType])
//        (ce * scoresForSeqAux.mean(dim = 1)).sum(dim = 1).mean() * alpha.toParamType[ParamType]
//      } else {
//        val maskCe = torch.nn.functional.oneHot(topkIdxForAuxLoss.view(-1), numClasses = nRoutedExperts).toType[ParamType]
//        val ce = maskCe.mean(0)
//        val Pi = scoresForAux.mean(0)
//        val fi = ce * nRoutedExperts.toParamType[ParamType]
//        (Pi * fi).sum() * alpha.toParamType[ParamType]
//      }
//    } else {
//      torch.tensor[ParamType](0.0, device = hiddenStates.device)
//    }
//
//    (topkIdx, normalizedTopkWeight, auxLoss)
//  }
//
//  def apply(hiddenStates: Tensor[ParamType]): (Tensor[ParamType], Tensor[ParamType], Tensor[ParamType]) = forward(hiddenStates)
//}

// MoE 前馈网络
//class MOEFeedForward[ParamType <: FloatNN: Default](config: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
//  val configValue = config
//  val experts = nn.ModuleList(Seq.fill(config.nRoutedExperts)(new FeedForward[ParamType](config)))
//  val gate = new MoEGate[ParamType](config)
//  val sharedExperts = if (config.nSharedExperts > 0) Some(ModuleList(Seq.fill(config.nSharedExperts)(new FeedForward[ParamType](config)))) else None
//  var auxLoss: Tensor[ParamType] = torch.tensor[ParamType](0.0)
//
//  registerModule("gate", gate)
//  registerModuleList("experts", experts)
//  sharedExperts.foreach(registerModuleList("sharedExperts", _))
//
//  override def forward(x: Tensor[ParamType]): Tensor[ParamType] = {
//    val identity = x
//    val origShape = x.shape
//    val (bsz, seqLen, _) = x.shape
//    val (topkIdx, topkWeight, auxLossValue) = gate(x)
//    val xFlattened = x.view(-1, x.shape.last)
//    val flatTopkIdx = topkIdx.view(-1)
//    auxLoss = auxLossValue
//
//    val y = if (isTraining) {
//      val repeatedX = xFlattened.repeatInterleave(config.numExpertsPerTok, dim = 0)
//      val yTensor = torch.emptyLike(repeatedX, dtype = TensorType.Float16)
//      experts.zipWithIndex.foreach { case (expert, i) =>
//        val mask = flatTopkIdx == i.toParamType[ParamType]
//        yTensor(mask) = expert(repeatedX(mask)).to(yTensor.dtype)
//      }
//      (yTensor.view(topkWeight.shape ++ Seq(-1)) * topkWeight.unsqueeze(-1)).sum(dim = 1).view(origShape)
//    } else {
//      moeInfer(xFlattened, flatTopkIdx, topkWeight.view(-1, 1)).view(origShape)
//    }
//
//    sharedExperts.foreach { experts =>
//      experts.foreach { expert =>
//        y += expert(identity)
//      }
//    }
//
//    y
//  }
//
//  private def moeInfer(x: Tensor[ParamType], flatExpertIndices: Tensor[ParamType], flatExpertWeights: Tensor[ParamType]): Tensor[ParamType] = {
//    torch.no_grad {
//      val expertCache = torch.zerosLike(x)
//      val idxs = flatExpertIndices.argsort()
//      val tokensPerExpert = flatExpertIndices.bincount().cpu().numpy().cumsum(0)
//      val tokenIdxs = idxs / configValue.numExpertsPerTok.toParamType[ParamType]
//
//      tokensPerExpert.zipWithIndex.foreach { case (endIdx, i) =>
//        val startIdx = if (i == 0) 0 else tokensPerExpert(i - 1)
//        if (startIdx != endIdx) {
//          val expert = experts(i)
//          val expTokenIdx = tokenIdxs.slice(startIdx, endIdx)
//          val expertTokens = x(expTokenIdx)
//          val expertOut = expert(expertTokens).to(expertCache.dtype)
//          expertOut.mul(flatExpertWeights(idxs.slice(startIdx, endIdx)))
//          expertCache.scatterAdd(0, expTokenIdx.view(-1, 1).repeat(1, x.shape.last), expertOut)
//        }
//      }
//
//      expertCache
//    }
//  }
//
//  def apply(x: Tensor[ParamType]): Tensor[ParamType] = forward(x)
//}

// MiniMind 块
//class MiniMindBlock[ParamType <: FloatNN: Default](layerId: Int, config: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
//  val numAttentionHeads = config.numAttentionHeads
//  val hiddenSize = config.hiddenSize
//  val headDim = config.hiddenSize / config.numAttentionHeads
//  val selfAttn = new Attention[ParamType](config)
//  val layerIdValue = layerId
//  val inputLayernorm = new RMSNorm[ParamType](config.hiddenSize, eps = config.rmsNormEps)
//  val postAttentionLayernorm = new RMSNorm[ParamType](config.hiddenSize, eps = config.rmsNormEps)
//  val mlp = if (!config.useMoe) new FeedForward[ParamType](config) else new MOEFeedForward[ParamType](config)
//
//  registerModule("selfAttn", selfAttn)
//  registerModule("inputLayernorm", inputLayernorm)
//  registerModule("postAttentionLayernorm", postAttentionLayernorm)
//  registerModule("mlp", mlp)
//
//  override def forward(hiddenStates: Tensor[ParamType], positionEmbeddings: (Tensor[ParamType], Tensor[ParamType]), pastKeyValue: Option[(Tensor[ParamType], Tensor[ParamType])] = None, useCache: Boolean = false, attentionMask: Option[Tensor[ParamType]] = None): (Tensor[ParamType], Option[(Tensor[ParamType], Tensor[ParamType])]) = {
//    val residual = hiddenStates
//    val (attnOutput, presentKeyValue) = selfAttn(inputLayernorm(hiddenStates), positionEmbeddings, pastKeyValue, useCache, attentionMask)
//    val attnOutputWithResidual = attnOutput + residual
//    val finalOutput = attnOutputWithResidual + mlp(postAttentionLayernorm(attnOutputWithResidual))
//    (finalOutput, presentKeyValue)
//  }
//
//  def apply(hiddenStates: Tensor[ParamType], positionEmbeddings: (Tensor[ParamType], Tensor[ParamType]), pastKeyValue: Option[(Tensor[ParamType], Tensor[ParamType])] = None, useCache: Boolean = false, attentionMask: Option[Tensor[ParamType]] = None): (Tensor[ParamType], Option[(Tensor[ParamType], Tensor[ParamType])]) =
//    forward(hiddenStates, positionEmbeddings, pastKeyValue, useCache, attentionMask)
//}

// MiniMind 模型
//class MiniMindModel[ParamType <: FloatNN: Default](config: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
//  val configValue = config
//  val vocabSize = config.vocabSize
//  val numHiddenLayers = config.numHiddenLayers
//  val embedTokens = nn.Embedding(config.vocabSize, config.hiddenSize)
//  val dropout = nn.Dropout(config.dropout)
//  val layers = nn.ModuleList(Seq.tabulate(config.numHiddenLayers)(l => new MiniMindBlock[ParamType](l, config)))
//  val norm = new RMSNorm[ParamType](config.hiddenSize, eps = config.rmsNormEps)
//
//  registerModule("embedTokens", embedTokens)
//  registerModule("dropout", dropout)
////  registerModuleList("layers", layers)
//  registerModule("norm", norm)
//
//  val (freqsCos, freqsSin) = precomputeFreqsCis[ParamType](dim = config.hiddenSize / config.numAttentionHeads, end = config.maxPositionEmbeddings, theta = config.ropeTheta)
//  register_buffer("freqsCos", freqsCos, persistent = false)
//  register_buffer("freqsSin", freqsSin, persistent = false)
//
//  override def forward(inputIds: Tensor[ParamType], attentionMask: Option[Tensor[ParamType]] = None, pastKeyValues: Option[List[(Tensor[ParamType], Tensor[ParamType])]] = None, useCache: Boolean = false, kwargs: Map[String, Any] = Map.empty): (Tensor[ParamType], List[(Tensor[ParamType], Tensor[ParamType])], Tensor[ParamType]) = {
//    val (batchSize, seqLength) = inputIds.shape
//    val pastKeyValuesList = pastKeyValues.getOrElse(List.fill(layers.size)(null.asInstanceOf[(Tensor[ParamType], Tensor[ParamType])]))
//    val startPos = if (pastKeyValuesList.head != null) pastKeyValuesList.head._1.shape(1) else 0
//
//    val hiddenStates = dropout(embedTokens(inputIds))
//
//    val positionEmbeddings = (
//      freqsCos.slice(startPos, startPos + seqLength),
//      freqsSin.slice(startPos, startPos + seqLength)
//    )
//
//    val presents = new ListBuffer[(Tensor[ParamType], Tensor[ParamType])]()
//    var currentHiddenStates = hiddenStates
//    layers.zip(pastKeyValuesList).foreach { case (layer, pastKeyValue) =>
//      val (newHiddenStates, present) = layer(currentHiddenStates, positionEmbeddings, Option(pastKeyValue), useCache, attentionMask)
//      currentHiddenStates = newHiddenStates
//      present.foreach(presents.append(_))
//    }
//
//    val finalHiddenStates = norm(currentHiddenStates)
//
//    val auxLoss = layers.foldLeft(torch.tensor[ParamType](0.0)) { (acc, layer) =>
//      layer.mlp match {
//        case moe: MOEFeedForward[ParamType] => acc + moe.auxLoss
//        case _ => acc
//      }
//    }
//
//    (finalHiddenStates, presents.toList, auxLoss)
//  }
//
//  def apply(inputIds: Tensor[ParamType], attentionMask: Option[Tensor[ParamType]] = None, pastKeyValues: Option[List[(Tensor[ParamType], Tensor[ParamType])]] = None, useCache: Boolean = false, kwargs: Map[String, Any] = Map.empty): (Tensor[ParamType], List[(Tensor[ParamType], Tensor[ParamType])], Tensor[ParamType]) =
//    forward(inputIds, attentionMask, pastKeyValues, useCache, kwargs)
//}

// MiniMind 因果语言模型
//class MiniMindForCausalLM[ParamType <: FloatNN: Default](config: Option[MiniMindConfig] = None) extends HasParams[ParamType] with TensorModule[ParamType] {
//  val configValue = config.getOrElse(new MiniMindConfig())
//  val model = new MiniMindModel[ParamType](configValue)
//  val lmHead = nn.Linear(configValue.hiddenSize, configValue.vocabSize, bias = false)
//
//  register_module("model", model)
//  register_module("lmHead", lmHead)
//
//  // 共享权重
//  model.embedTokens.weight = lmHead.weight
//
//  override def forward(inputIds: Tensor[ParamType], attentionMask: Option[Tensor[ParamType]] = None, pastKeyValues: Option[List[(Tensor[ParamType], Tensor[ParamType])]] = None, useCache: Boolean = false, logitsToKeep: Any = 0, args: Map[String, Any] = Map.empty): CausalLMOutputWithPast[ParamType] = {
//    val (h, pastKvs, auxLoss) = model(inputIds, attentionMask, pastKeyValues, useCache, args)
//    val sliceIndices = logitsToKeep match {
//      case i: Int => if (i > 0) -i until h.shape(1) else 0 until h.shape(1)
//      case t: Tensor[ParamType] => ??? // 需根据具体情况实现
//    }
//    val logits = lmHead(h.slice(sliceIndices, ::))
//    CausalLMOutputWithPast(
//      lastHiddenState = h,
//      logits = logits,
//      auxLoss = auxLoss,
//      pastKeyValues = pastKvs
//    )
//  }
//
//  def apply(inputIds: Tensor[ParamType], attentionMask: Option[Tensor[ParamType]] = None, pastKeyValues: Option[List[(Tensor[ParamType], Tensor[ParamType])]] = None, useCache: Boolean = false, logitsToKeep: Any = 0, args: Map[String, Any] = Map.empty): CausalLMOutputWithPast[ParamType] =
//    forward(inputIds, attentionMask, pastKeyValues, useCache, logitsToKeep, args)
//}

//// 因果语言模型输出类
//case class CausalLMOutputWithPast[ParamType <: FloatNN: Default](
//    lastHiddenState: Tensor[ParamType],
//    logits: Tensor[ParamType],
//    auxLoss: Tensor[ParamType],
//    pastKeyValues: List[(Tensor[ParamType], Tensor[ParamType])]
//)

//// 激活函数映射
//object ACT2FN {
//  def apply[ParamType <: FloatNN: Default](actName: String): Tensor[ParamType] => Tensor[ParamType] = actName match {
//    case "silu" => torch.nn.functional.silu[ParamType]
//    case _ => throw new NotImplementedError(s"Unsupported activation function: $actName")
//  }
//}
