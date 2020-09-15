(ns clj-djl.model-zoo
  (:import [ai.djl.basicmodelzoo.basic Mlp]))

(defn mlp [input output hidden]
  (Mlp. input output hidden))
