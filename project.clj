(defproject clj-djl "0.1.0-SNAPSHOT"
  :description "A clojure lib wraps deep java learning(DJL.ai)"
  :url "http://github.com/kimim/clj-djl"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 ;; https://mvnrepository.com/artifact/ai.djl.mxnet/mxnet-engine
                 [ai.djl.mxnet/mxnet-engine "0.7.0"]]
  :repl-options {:init-ns clj-djl.core})
