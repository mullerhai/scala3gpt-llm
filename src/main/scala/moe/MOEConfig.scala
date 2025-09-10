package moe

case class MOEConfig(
                      hiddenDim: Int,
                      expertNumber: Int,
                      topK: Int,
                      sharedExpertsNumber: Int = 2
                    )
