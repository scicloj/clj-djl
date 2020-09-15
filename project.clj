(defproject clj-djl "0.1.0"
  :description "A clojure lib wraps deep java learning(DJL.ai)"
  :url "http://github.com/kimim/clj-djl"
  :license {:name "Apache License, Version 2.0"
            :url "http://www.apache.org/licenses/LICENSE-2.0.html"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 ;; https://mvnrepository.com/artifact/ai.djl.mxnet/mxnet-engine
                 [ai.djl/api "0.7.0"]
                 [ai.djl/model-zoo "0.7.0"]
                 [ai.djl/basicdataset "0.7.0"]
                 [ai.djl.mxnet/mxnet-engine "0.7.0"]]
  :repl-options {:init-ns clj-djl.core})
