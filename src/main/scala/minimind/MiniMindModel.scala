package minimind

import torch.{Default, FloatNN, Tensor, nn,::}
import torch.nn.modules.{HasParams, TensorModule}

import scala.collection.mutable.ListBuffer

class MiniMindModel[ParamType <: FloatNN: Default](config: MiniMindConfig) extends HasParams[ParamType] with TensorModule[ParamType] {
  val configValue = config
  val vocabSize = config.vocabSize
  val numHiddenLayers = config.numHiddenLayers
  val embedTokens = nn.Embedding(config.vocabSize, config.hiddenSize)
  val dropout = nn.Dropout(config.dropout.toFloat)
  val layers = nn.ModuleList(Seq.tabulate(config.numHiddenLayers)(l => new MiniMindBlock[ParamType](l, config)))
  val norm = new RMSNorm[ParamType](config.hiddenSize, eps = config.rmsNormEps)

  register_module("embedTokens", embedTokens)
  register_module("dropout", dropout)
  //  registerModuleList("layers", layers)
  register_module("norm", norm)

  val (freqsCos, freqsSin) = precomputeFreqsCis[ParamType](dim = config.hiddenSize / config.numAttentionHeads, end = config.maxPositionEmbeddings, theta = config.ropeTheta)
  register_buffer("freqsCos", freqsCos)//, persistent = false)
  register_buffer("freqsSin", freqsSin)//, persistent = false)

  override def forward(inputIds: Tensor[ParamType], attentionMask: Option[Tensor[ParamType]] = None, pastKeyValues: Option[List[(Tensor[ParamType], Tensor[ParamType])]] = None, useCache: Boolean = false, kwargs: Map[String, Any] = Map.empty): (Tensor[ParamType], List[(Tensor[ParamType], Tensor[ParamType])], Tensor[ParamType]) = {
//    val (batchSize, seqLength) = inputIds.shape
    val batchSize = inputIds.shape(0)
    val seqLength = inputIds.shape(1)
    val pastKeyValuesList = pastKeyValues.getOrElse(List.fill(layers.size)(null.asInstanceOf[(Tensor[ParamType], Tensor[ParamType])]))
    val startPos = if (pastKeyValuesList.head != null) pastKeyValuesList.head._1.shape(1) else 0

    val hiddenStates = dropout(embedTokens(inputIds))

    val positionEmbeddings = (
      freqsCos(startPos.::(startPos + seqLength)),
      freqsSin(startPos.::(startPos + seqLength))
    )

    val presents = new ListBuffer[(Tensor[ParamType], Tensor[ParamType])]()
    var currentHiddenStates = hiddenStates
    layers.zip(pastKeyValuesList).foreach { case (layer, pastKeyValue) =>
      val (newHiddenStates, present) = layer(currentHiddenStates, positionEmbeddings, Option(pastKeyValue), useCache, attentionMask)
      currentHiddenStates = newHiddenStates
      present.foreach(presents.append(_))
    }

    val finalHiddenStates = norm(currentHiddenStates)

    val auxLoss = layers.foldLeft(torch.tensor(0.0)) { (acc, layer) =>
      layer.mlp match {
        case moe: MOEFeedForward[ParamType] => acc + moe.auxLoss
        case _ => acc
      }
    }

    (finalHiddenStates, presents.toList, auxLoss)
  }

  def apply(inputIds: Tensor[ParamType], attentionMask: Option[Tensor[ParamType]] = None, pastKeyValues: Option[List[(Tensor[ParamType], Tensor[ParamType])]] = None, useCache: Boolean = false, kwargs: Map[String, Any] = Map.empty): (Tensor[ParamType], List[(Tensor[ParamType], Tensor[ParamType])], Tensor[ParamType]) =
    forward(inputIds, attentionMask, pastKeyValues, useCache, kwargs)
}

