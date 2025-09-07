ThisBuild / version := "0.1.0"

ThisBuild / scalaVersion := "3.7.2"

lazy val root = (project in file("."))
  .settings(
    name := "scala3gpt-llm"
  )

libraryDependencies += "io.github.mullerhai" % "storch_core_3" % "0.4.6-1.15.2"


ThisBuild / assemblyMergeStrategy := {
  case v if v.contains("main$package$.class")                => MergeStrategy.first
  case v if v.contains("main.class")                         => MergeStrategy.first
  case v if v.contains("main$package.class")                 => MergeStrategy.first
  case v if v.contains("main$package.tasty")                 => MergeStrategy.first
  case v if v.contains("main.tasty")                         => MergeStrategy.first
  case v if v.contains("main.class")                         => MergeStrategy.discard
  case v if v.contains("module-info.class")                  => MergeStrategy.discard
  case v if v.contains("UnusedStub")                         => MergeStrategy.first
  case v if v.contains("aopalliance")                        => MergeStrategy.first
  case v if v.contains("inject")                             => MergeStrategy.first
  case v if v.contains("jline")                              => MergeStrategy.discard
  case v if v.contains("scala-asm")                          => MergeStrategy.discard
  case v if v.contains("asm")                                => MergeStrategy.discard
  case v if v.contains("scala-compiler")                     => MergeStrategy.deduplicate
  case v if v.contains("reflect-config.json")                => MergeStrategy.discard
  case v if v.contains("jni-config.json")                    => MergeStrategy.discard
  case v if v.contains("git.properties")                     => MergeStrategy.discard
  case v if v.contains("reflect.properties")                 => MergeStrategy.discard
  case v if v.contains("compiler.properties")                => MergeStrategy.discard
  case v if v.contains("scala-collection-compat.properties") => MergeStrategy.discard
  case x =>
    val oldStrategy = (assembly / assemblyMergeStrategy).value
    oldStrategy(x)
}
