(defproject clj-djl/clj-djl "0.1.9"
  :description "A clojure lib wraps deep java learning(DJL.ai)"
  :url "http://github.com/scicloj/clj-djl"
  :license {:name "Apache License, Version 2.0"
            :url "http://www.apache.org/licenses/LICENSE-2.0.html"}
  :dependencies [[org.clojure/clojure "1.10.3"]
                 [org.slf4j/slf4j-api "1.7.32"]
                 [ai.djl/api "0.16.0"]
                 [ai.djl/model-zoo "0.16.0"]
                 [ai.djl/basicdataset "0.16.0"]
                 [ai.djl.mxnet/mxnet-engine "0.16.0"]
                 [ai.djl.mxnet/mxnet-native-auto "1.8.0"]

                 [net.mikera/core.matrix "0.62.0"]]
                 ;;[ai.djl.pytorch/pytorch-engine "0.8.0"]
                 ;;[ai.djl.pytorch/pytorch-native-auto "1.6.0"]

                 ;;[ai.djl.tensorflow/tensorflow-engine "0.8.0"]
                 ;;[ai.djl.tensorflow/tensorflow-native-auto "2.3.0"]

  :plugins [[lein-localrepo "0.5.4"]]
  :source-paths ["src"]
  :main ^:skip-aot clj-djl.core
  :repl-options {:init-ns clj-djl.core}
  :profiles {:dev
             {:dependencies [[org.slf4j/slf4j-simple "1.7.32"]]}
             :codox
             {:dependencies [[codox-theme-rdash "0.1.2"]]
              :plugins [[lein-codox "0.10.7"]]
              :codox {:project {:name "clj-djl"}
                      :themes [:rdash]
                      :metadata {:doc/format :markdown}
                      :source-paths ["src"]
                      :source-uri "https://github.com/kimim/clj-djl/blob/master/{filepath}#L{line}"
                      :output-path "docs"}}}
  :aliases {"codox" ["with-profile" "codox" "codox"]}
  :repositories [["sonatype" "https://oss.sonatype.org/content/repositories/snapshots/"]])
