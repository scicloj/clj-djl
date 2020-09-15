(ns clj-djl.training.dataset
  (:import [ai.djl.training.dataset Dataset]))

(defn set-sampling [builder batch-size drop-last]
  (.setSampling builder batch-size drop-last)
  builder)

(defn build [builder]
  (.build builder))

(defn prepare [ds progress]
  (.prepare ds progress)
  ds)
