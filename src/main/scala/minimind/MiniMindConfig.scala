package minimind

import torch.{Default, Tensor, nn, FloatNN}
import torch.nn.modules.{HasParams, TensorModule}

class MiniMindConfig(
                      val dropout: Double = 0.0,
                      val bosTokenId: Int = 1,
                      val eosTokenId: Int = 2,
                      val hiddenAct: String = "silu",
                      val hiddenSize: Int = 512,
                      var intermediateSize: Option[Int] = None,
                      val maxPositionEmbeddings: Int = 32768,
                      val numAttentionHeads: Int = 8,
                      val numHiddenLayers: Int = 8,
                      val numKeyValueHeads: Int = 2,
                      val vocabSize: Int = 6400,
                      val rmsNormEps: Double = 1e-5,
                      val ropeTheta: Double = 1e6,
                      val flashAttn: Boolean = true,
                      val useMoe: Boolean = false,
                      val numExpertsPerTok: Int = 2,
                      val nRoutedExperts: Int = 4,
                      val nSharedExperts: Int = 1,
                      val scoringFunc: String = "softmax",
                      val auxLossAlpha: Double = 0.1,
                      val seqAux: Boolean = true,
                      val normTopkProb: Boolean = true
                    )
