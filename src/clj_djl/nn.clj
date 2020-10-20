(ns clj-djl.nn
  (:import [ai.djl.nn Activation SequentialBlock]
           [ai.djl.nn.core Linear]))

(defn relu-block []
  (Activation/reluBlock))

(defn relu [data]
  (Activation/relu data))

(defn sigmoid-block []
  (Activation/sigmoidBlock))

(defn sigmoid [data]
  (Activation/sigmoid data))

(defn sequential-block []
  (SequentialBlock.))

(defn new-linear-builder []
  (Linear/builder))

(defn opt-bias [builder bias]
  (.optBias builder bias))

(defn set-units [builder unit]
  (.setUnits builder unit))

(defn build [builder]
  (.build builder))

(defn add [net block]
  (.add net block))
