(ns clj-djl.nn
  (:import [ai.djl.nn Activation]))

(defn relu-block []
  (Activation/reluBlock))

(defn relu [data]
  (Activation/relu data))

(defn sigmoid-block []
  (Activation/sigmoidBlock))

(defn sigmoid [data]
  (Activation/sigmoid data))
